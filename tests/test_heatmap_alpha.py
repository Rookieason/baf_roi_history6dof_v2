"""Behavioral tests for actual per-step alpha fusion, not endpoint mixing."""
import json
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from baf_roi.pose import ypr_to_quat, quat_to_ypr, interpolate_predictions
from baf_roi.runtime import Predictor
from baf_roi.upstream_ops import apply_decision_fusion
from baf_roi_heatmap.contracts import InputConfig
from baf_roi_heatmap.data import HeatmapSeries, encode_window
from baf_roi_heatmap.losses import numpy_prediction, task_loss
from baf_roi.roi import Scene
from baf_roi_heatmap_alpha.model import (AlphaConfig, AlphaFusion, HeatmapAlphaGate,
    correct_alpha, heatmap_statistics, quaternion_to_ypr, ypr_to_quaternion)
from baf_roi_heatmap_alpha.runtime import HeatmapAlphaPredictor


class ToyConfidence(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(.2))

    def forward(self, x):
        return {'confidence': torch.sigmoid(x[:, :, 0].mean(1)*self.weight-.4)}


class ToyExpert(nn.Module):
    def __init__(self, offset):
        super().__init__(); self.offset = nn.Parameter(torch.tensor(offset))

    def forward(self, x, aux=None):
        p = x[:, -1, :3]+self.offset*torch.tensor([1., .2, -.1], device=x.device)
        q = x[:, -1, 3:]
        return {'p_hat_tr': p, 'q_hat_norm': q/q.norm(dim=-1, keepdim=True)}


def fixture(batch=2, horizons=(50, 250, 1000)):
    torch.set_num_threads(2)
    base = Predictor({'confidence': ToyConfidence(), 'focused': ToyExpert(.002), 'explorer': ToyExpert(.02)},
                     {'mean': [0., 0., 0.], 'std': [1., 1., 1.]}, .6)
    t = np.arange(30)/30
    h = np.stack((1+t*.1, t*.01, np.ones(30), t*10, t*2, t), -1)
    histories = np.tile(h[None], (batch, 1, 1))
    initial = np.linspace(.2, .8, batch).astype(np.float32)
    ref = base.forecast_batch(histories, horizons, initial_alpha=initial)
    x = {'maps': torch.randn(batch, 8, 3, 8, 8), 'mask': torch.ones(batch, 8, dtype=torch.bool),
         'timing': torch.zeros(batch, 8, 2), 'rf_available': torch.ones(batch, dtype=torch.bool),
         'alpha': torch.tensor(initial), 'horizon_s': torch.tensor(horizons).float()[None].expand(batch, -1)/1000,
         'base_p': torch.tensor(ref['positions_m']).float(), 'base_q': torch.tensor(ref['quaternions_wxyz']).float()}
    return base, histories, x, ref


def reference_rollout(base, histories, horizons, initial_alpha, delta):
    """Independent NumPy execution through the original author's fusion calls."""
    h = histories.copy(); alpha = initial_alpha.copy()
    ps, qs, weights = [h[:, -1, :3]], [ypr_to_quat(h[:, -1, 3:])], []
    for step in range(math.ceil(max(horizons)*30/1000-1e-10)):
        raw, fp, fq, ep, eq = base._step(h)
        prior = alpha if step == 0 else base.mu*raw+(1-base.mu)*alpha
        multiplier = np.exp(delta)
        alpha = prior*multiplier/(1-prior+prior*multiplier)
        p, q, _ = apply_decision_fusion('full_slerp', alpha, raw, fp, ep, fq, eq, .5, .5)
        ps.append(p); qs.append(q); weights.append(alpha.copy())
        h = np.concatenate((h[:, 1:], np.concatenate((p, quat_to_ypr(q)), -1)[:, None]), 1)
    p, q = interpolate_predictions(np.stack(ps, 1), np.stack(qs, 1), horizons)
    return p, q, np.stack(weights, 1)


def test_zero_correction_is_bit_exact_baseline_including_saturated_alpha():
    base, h, x, ref = fixture()
    model = AlphaFusion(base).eval()
    pred = model(x, torch.tensor(h))
    p, q = numpy_prediction(pred, ref['positions_m'], ref['quaternions_wxyz'])
    np.testing.assert_array_equal(p, ref['positions_m'])
    np.testing.assert_array_equal(q, ref['quaternions_wxyz'])
    assert not pred['applied'].any()
    a = torch.tensor([0., .2, .5, .8, 1.])
    torch.testing.assert_close(correct_alpha(a, torch.zeros_like(a)), a, rtol=0, atol=0)
    assert correct_alpha(a, torch.ones_like(a))[[0, -1]].tolist() == [0., 1.]


def test_nonzero_gate_matches_actual_per_step_numpy_rollout_and_changes_intermediates():
    base, h, x, ref = fixture()
    model = AlphaFusion(base).eval()
    model.gate.head[-1].bias.data.fill_(.4)
    pred = model(x, torch.tensor(h))
    delta = pred['logit_delta'].detach().numpy()
    p, q, alpha = reference_rollout(base, h, [50, 250, 1000], x['alpha'].numpy(), delta)
    np.testing.assert_allclose(pred['p'].detach().numpy(), p, atol=4e-6)
    np.testing.assert_allclose(pred['q'].detach().numpy(), q, atol=4e-6)
    np.testing.assert_allclose(pred['rollout_alpha'].detach().numpy(), alpha, atol=1e-6)
    assert not np.allclose(pred['rollout_positions_m'].detach().numpy(), ref['rollout_positions_m'])
    assert np.all(pred['rollout_alpha'][:, 0].detach().numpy() > x['alpha'].numpy())


def test_missing_rf_for_one_sample_restores_exact_original_trajectory():
    base, h, x, ref = fixture()
    model = AlphaFusion(base).eval(); model.gate.head[-1].bias.data.fill_(1.)
    x['rf_available'][1] = False
    pred = model(x, torch.tensor(h))
    p, q = numpy_prediction(pred, ref['positions_m'], ref['quaternions_wxyz'])
    np.testing.assert_array_equal(p[1], ref['positions_m'][1])
    np.testing.assert_array_equal(q[1], ref['quaternions_wxyz'][1])
    assert pred['applied'][0].all() and not pred['applied'][1].any()


def test_fallback_gate_diagnostic_is_invariant_to_batch_composition():
    base, h, x, _ = fixture()
    model = AlphaFusion(base).eval(); model.gate.head[-1].bias.data.fill_(.4)
    x['rf_available'][1] = False
    mixed = model(x, torch.tensor(h))
    alone = model({key: value[1:] for key, value in x.items()}, torch.tensor(h[1:]))
    torch.testing.assert_close(mixed['gate'][1:], alone['gate'], atol=0, rtol=0)
    assert torch.equal(alone['gate'], x['alpha'][1:, None].expand_as(alone['gate']))


@pytest.mark.parametrize('bound', [0., -1., float('nan'), float('inf'), 100.])
def test_invalid_logit_bounds_fail_before_rollout(bound):
    with pytest.raises(ValueError):
        AlphaConfig(max_logit_delta=bound).validate()


def test_largest_supported_logit_bound_has_finite_gradients():
    alpha = torch.tensor([0., .01, .5, .99, 1.], requires_grad=True)
    for value in (-10., 10.):
        delta = torch.full_like(alpha, value, requires_grad=True)
        corrected = correct_alpha(alpha, delta)
        assert torch.isfinite(corrected).all() and torch.all((corrected >= 0) & (corrected <= 1))
        corrected.sum().backward()
        assert torch.isfinite(delta.grad).all() and torch.isfinite(alpha.grad).all()


@pytest.mark.parametrize('mode', ['legacy_center_cone', 'rectangular_center_frustum'])
def test_training_changes_only_gate_and_has_finite_gradients(mode):
    base, h, x, ref = fixture(horizons=(50, 250))
    model = AlphaFusion(base).train()
    frozen = {name: p.detach().clone() for name, p in model.experts.named_parameters()}
    scene = Scene(np.array([[4., 0, 1], [3., 2, 1]]), ['a', 'b'],
                  {'visibility_mode': mode, 'fov_deg': 110., 'horizontal_fov_deg': 110.,
                   'vertical_fov_deg': 100., 'near_m': .01})
    y = {'target_p': x['base_p']+.1, 'target_q': x['base_q'], 'target_mask': torch.ones(2, 2, 2)}
    optimizer = torch.optim.Adam(model.gate.parameters(), lr=.01)
    pred = model(x, torch.tensor(h)); loss = task_loss(pred, y, scene)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.gate.parameters())
    assert model.gate.head[-1].bias.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.experts.parameters())
    optimizer.step()
    for name, p in model.experts.named_parameters():
        assert torch.equal(p, frozen[name])
    assert not model.experts.training and model.gate.training


def test_statistics_ignore_masked_frames_even_nonfinite_values():
    _, _, x, _ = fixture()
    x['mask'][:, 1:3] = False
    a = heatmap_statistics(x)
    x['maps'][:, 1:3] = float('nan')
    x['timing'][:, 1:3] = float('nan')
    torch.testing.assert_close(heatmap_statistics(x), a, rtol=0, atol=0)
    assert a.shape == (2, 8)


def test_quaternion_coordinate_contract_matches_scipy_and_gimbal_gradients_are_finite():
    angles = np.array([[33., -21., 7.], [235., 6., -13.], [20., 90., 30.], [20., -90., 30.]])
    q = ypr_to_quaternion(torch.tensor(angles))
    np.testing.assert_allclose(q.numpy(), ypr_to_quat(angles), atol=1e-14)
    np.testing.assert_allclose(quaternion_to_ypr(q).numpy(), quat_to_ypr(q.numpy()), atol=1e-6)
    q = q.detach().requires_grad_()
    quaternion_to_ypr(q).sum().backward()
    assert torch.isfinite(q.grad).all()


def test_future_rf_cannot_change_alpha(tmp_path):
    cfg = InputConfig(size=8)
    end = np.arange(.2, 2., .05)
    norm = {'median': 0., 'scale': 1.}
    windows = []
    for i in range(2):
        maps = np.arange(len(end)*5*4).reshape(len(end), 5, 4).astype(float)+1
        maps[end > 1.2] *= 1+1000000*i
        metadata = {'format_version': 1, 'unit': 'linear', 'representation': 'aoa_tof',
                    'physical_axes_verified': True, 'clock_domain': 'unix_seconds', 'availability_source': 'test'}
        path = tmp_path/f'{i}.npz'
        np.savez(path, maps=maps, valid=np.ones(len(end), bool), support_start_s=end-.2,
                 support_end_s=end, available_at_s=end+.02, theta_deg=np.linspace(-90, 90, 5),
                 tau_s=np.linspace(0, 3e-9, 4), metadata_json=np.array(json.dumps(metadata)))
        window = encode_window(HeatmapSeries(path), 1.2, norm, cfg)
        windows.append({k: torch.tensor(np.asarray(v)[None]) for k, v in window.items()})
    gate = HeatmapAlphaGate()
    nn.init.normal_(gate.head[-1].weight)
    torch.testing.assert_close(gate(windows[0]), gate(windows[1]), rtol=0, atol=0)


def test_online_does_not_commit_forecast_alpha_and_outage_returns_original(monkeypatch):
    import baf_roi_heatmap_alpha.runtime as runtime
    base, h, x, _ = fixture(batch=1)
    baseline, _, _, _ = fixture(batch=1)
    online = HeatmapAlphaPredictor.__new__(HeatmapAlphaPredictor)
    online.base = base; online.model = AlphaFusion(base).eval()
    online.model.gate.head[-1].bias.data.fill_(1.)
    online.cfg = InputConfig(size=8); online.device = 'cpu'
    online.bundle = {'horizons_ms': [50, 250, 1000], 'grid_id': 'synthetic',
                     'allow_exploratory': False, 'normalizer': {'median': 0., 'scale': 1.}}
    rf = SimpleNamespace(grid_id='synthetic', metadata={'physical_axes_verified': True})
    window = {k: x[k][0].numpy() for k in ('maps', 'mask', 'timing', 'rf_available')}
    def encode(series, *args):
        result = window.copy()
        result['rf_available'] = np.bool_(series is not None)
        return result
    monkeypatch.setattr(runtime, 'encode_window', encode)
    for i in range(2):
        ts = np.arange(30)/30+i/30
        ref = baseline.predict_observations(ts, h[0, :, :3], h[0, :, 3:],
                                            horizons_ms=[50, 250, 1000], stream_id='a')
        pred = online.predict_observations(ts, h[0, :, :3], h[0, :, 3:], rf if i == 0 else None, stream_id='a')
        assert online.base.observed_state == baseline.observed_state
        if i == 0:
            assert pred['adapted']
        else:
            np.testing.assert_array_equal(pred['positions_m'], ref['positions_m'])
            np.testing.assert_array_equal(pred['quaternions_wxyz'], ref['quaternions_wxyz'])
    old = online.base.observed_state.copy()
    with pytest.raises(ValueError):
        online.predict_observations(ts, h[0, :, :3], h[0, :, 3:], rf, stream_id='a')
    assert online.base.observed_state == old
    online.reset('a'); assert 'a' not in online.base.observed_state


def test_actual_baf_torch_rollout_matches_baseline_before_fallback_and_backpropagates():
    from baf_roi.models import build_models
    from baf_roi.pose import quaternion_error_deg
    torch.set_num_threads(2); torch.manual_seed(47)
    base = Predictor(build_models(), {'mean': [1., .2, 1.], 'std': [.12, .2, .3]}, .6)
    _, h, x, _ = fixture()
    h[1, :, 3] += 235  # Keep the initial, unwrapped observed Euler angles.
    reference = base.forecast_batch(h, [50, 250, 1000], initial_alpha=x['alpha'].numpy())
    x['base_p'] = torch.tensor(reference['positions_m']).float()
    x['base_q'] = torch.tensor(reference['quaternions_wxyz']).float()
    model = AlphaFusion(base).train()  # Do not take the eval-mode zero-delta shortcut.
    with torch.no_grad():
        prediction = model(x, torch.tensor(h))
    np.testing.assert_allclose(prediction['p'].numpy(), reference['positions_m'], atol=1e-5, rtol=0)
    assert np.max(quaternion_error_deg(prediction['q'].numpy(), reference['quaternions_wxyz'])) < .01
    # Several recurrent steps exercise gradients through the frozen LSTM/TCN.
    x['horizon_s'] = torch.tensor([[.05, .15], [.05, .15]])
    x['base_p'], x['base_q'] = x['base_p'][:, :2], x['base_q'][:, :2]
    prediction = model(x, torch.tensor(h))
    loss = prediction['p'].square().sum()+prediction['q'][..., 0].sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.gate.head[-1].bias.grad.abs().sum() > 0
