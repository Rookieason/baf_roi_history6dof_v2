"""Tests of CSI's position inside BAF, causal rollout, and deployment parity.

These checks use the author's actual expert modules. Synthetic motion is only a
software fixture; passing them says nothing about held-out research accuracy.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from baf_roi.models import build_models, focused, explorer
from baf_roi.pose import quaternion_error_deg
from baf_roi.runtime import Predictor
from baf_roi_heatmap.contracts import InputConfig
from baf_roi_heatmap.data import HeatmapSeries, encode_window
from baf_roi_heatmap_residual.model import ResidualConfig, InternalResidualFusion
from baf_roi_heatmap_residual.runtime import HeatmapResidualPredictor, reference_prediction


class HistoryConfidence(nn.Module):
    """Make changes to generated history observable in subsequent alpha values."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.2))

    def forward(self, history):
        return {'confidence': torch.sigmoid(history[:, :, 0].mean(1)*self.weight-.4)}


def make_fixture(batch=2, horizons=(50, 250, 1000), full_models=False):
    torch.set_num_threads(2)
    torch.manual_seed(47)
    models = build_models() if full_models else {
        'confidence': HistoryConfidence(),
        'focused': focused.FocusedResidualLSTM(focused.FocusedBranchConfig(
            hidden_dim=8, num_layers=1, dropout=0.)),
        'explorer': explorer.ExplorerTCNBranch(explorer.ExplorerBranchConfig(
            aux_dim=0, hidden_dim=8, num_tcn_blocks=1, dropout=0.)),
    }
    base = Predictor(models, {'mean': [1., .2, 1.], 'std': [.12, .2, .3]}, .6)
    t = np.arange(30)/30
    history = np.stack((1+t*.1, .2+t*.01, 1+t*.015, t*10, t*2, t), -1)
    histories = np.tile(history[None], (batch, 1, 1))
    initial = np.linspace(.2, .8, batch).astype(np.float32)
    inputs = {
        'maps': torch.randn(batch, 8, 3, 8, 8),
        'mask': torch.ones(batch, 8, dtype=torch.bool),
        'timing': torch.zeros(batch, 8, 2),
        'rf_available': torch.ones(batch, dtype=torch.bool),
        'alpha': torch.tensor(initial),
        'horizon_s': torch.tensor(horizons).float()[None].expand(batch, -1)/1000,
    }
    return base, histories, inputs


def activate(model, *, spatial=False):
    """Nonzero, deterministic test weights; never presented as a trained model."""
    with torch.no_grad():
        for i, head in enumerate(model.residual.heads.values()):
            if spatial:
                nn.init.normal_(head.head[-1].weight, std=.05)
            else:
                head.head[-1].bias.fill_(.4 if i == 0 else -.3)
    assert not model.is_zero_residual()
    return model


def test_zero_residual_matches_actual_author_rollout_without_completed_forecast_input():
    base, history, inputs = make_fixture(full_models=True)
    history[1, :, 3] += 235  # Also retain the author's unwrapped observed angles.
    reference = base.forecast_batch(history, [50, 250, 1000], initial_alpha=inputs['alpha'].numpy())
    model = InternalResidualFusion(base).eval()
    assert model.is_zero_residual()
    assert not any(key.startswith('base_') for key in inputs)
    with torch.no_grad():
        prediction = model(inputs, torch.tensor(history))
    np.testing.assert_allclose(prediction['p'].numpy(), reference['positions_m'], atol=1e-5, rtol=0)
    assert np.max(quaternion_error_deg(prediction['q'].numpy(), reference['quaternions_wxyz'])) < .01
    assert not prediction['applied'].any()
    # Reports restore exact original float64 values on the runtime's global
    # zero-initialization fallback, rather than treating all applied=False alike.
    p, q = reference_prediction(model, prediction, inputs['rf_available'],
                                reference['positions_m'], reference['quaternions_wxyz'])
    np.testing.assert_array_equal(p, reference['positions_m'])
    np.testing.assert_array_equal(q, reference['quaternions_wxyz'])


def test_residual_reaches_original_heads_before_decode_and_changes_next_history():
    base, history, inputs = make_fixture(batch=1, horizons=(50, 100))
    model = InternalResidualFusion(base).eval()
    captures = {'focused': [], 'explorer': [], 'encoder': []}
    hooks = []
    for name in ('focused', 'explorer'):
        hooks.append(model.experts[name].translation_head.register_forward_pre_hook(
            lambda module, args, name=name: captures[name].append(args[0].detach().clone())))
    hooks.append(model.experts['focused'].encoder.register_forward_pre_hook(
        lambda module, args: captures['encoder'].append(args[0].detach().clone())))
    try:
        with torch.no_grad():
            original = model(inputs, torch.tensor(history))
        before = {name: list(values) for name, values in captures.items()}
        for values in captures.values():
            values.clear()
        activate(model)
        with torch.no_grad():
            assisted = model(inputs, torch.tensor(history))
    finally:
        for hook in hooks:
            hook.remove()
    for name, bias in (('focused', .4), ('explorer', -.3)):
        latent = before[name][0]
        scale = latent.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        expected = latent + model.config.max_relative_delta*scale*np.tanh(bias)
        torch.testing.assert_close(captures[name][0], expected, rtol=1e-5, atol=1e-6)
    # CSI starts at the latent: the first history is identical, but its decoded
    # prediction is fed back into the next LSTM input within the same forecast.
    torch.testing.assert_close(captures['encoder'][0], before['encoder'][0], rtol=0, atol=0)
    assert not torch.equal(captures['encoder'][1], before['encoder'][1])
    assert not torch.equal(assisted['rollout_positions_m'], original['rollout_positions_m'])
    torch.testing.assert_close(assisted['rollout_alpha'][:, 0], inputs['alpha'], rtol=0, atol=0)
    assert assisted['applied'].all()


def test_gradients_cross_frozen_decoder_and_rollout_while_only_residual_weights_train():
    base, history, inputs = make_fixture(horizons=(150,))
    model = InternalResidualFusion(base).train()
    frozen = {name: value.detach().clone() for name, value in model.experts.state_dict().items()}
    optimizer = torch.optim.Adam(model.residual.parameters(), lr=.01)
    # At zero initialization, there must already be a learning signal for the
    # final residual projection despite every original BAF parameter being frozen.
    prediction = model(inputs, torch.tensor(history))
    loss = prediction['p'].square().sum()+prediction['q'][..., 0].sum()
    loss.backward()
    assert all(head.head[-1].weight.grad is not None for head in model.residual.heads.values())
    assert sum(float(head.head[-1].weight.grad.abs().sum()) for head in model.residual.heads.values()) > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.experts.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    inputs['maps'].requires_grad_(True)
    first_latents = []
    def retain(module, args):
        args[0].retain_grad()
        first_latents.append(args[0])
    hook = model.experts['focused'].translation_head.register_forward_pre_hook(retain)
    try:
        prediction = model(inputs, torch.tensor(history))
        # Only the last horizon is supervised; the first step still needs a gradient.
        (prediction['p'][:, -1].square().sum()+prediction['q'][:, -1, 0].sum()).backward()
    finally:
        hook.remove()
    assert first_latents[0].grad is not None and first_latents[0].grad.abs().sum() > 0
    assert inputs['maps'].grad is not None and inputs['maps'].grad.abs().sum() > 0
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    for name, value in model.experts.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    assert not model.experts.training and model.residual.training


def test_prefix_horizons_do_not_depend_on_other_requested_horizons():
    base, history, inputs = make_fixture(batch=1)
    model = activate(InternalResidualFusion(base).eval(), spatial=True)
    with torch.no_grad():
        full = model(inputs, torch.tensor(history))
        selected = dict(inputs, horizon_s=inputs['horizon_s'][:, :2])
        prefix = model(selected, torch.tensor(history))
    for key in ('p', 'q', 'gate'):
        torch.testing.assert_close(prefix[key], full[key][:, :2], rtol=0, atol=0)
    torch.testing.assert_close(prefix['rollout_positions_m'],
                               full['rollout_positions_m'][:, :prefix['rollout_positions_m'].shape[1]], rtol=0, atol=0)


def test_heatmap_content_changes_expert_features_and_is_encoded_once_per_forecast():
    base, history, inputs = make_fixture(batch=1, horizons=(50, 150))
    model = activate(InternalResidualFusion(base).eval(), spatial=True)
    contexts, head_inputs = [], []
    hooks = [model.residual.encoder.register_forward_hook(
        lambda module, args, output: contexts.append(output.detach().clone())),
        model.experts['focused'].translation_head.register_forward_pre_hook(
        lambda module, args: head_inputs.append(args[0].detach().clone()))]
    try:
        with torch.no_grad():
            first = model(inputs, torch.tensor(history))
            first_head = head_inputs[0]
            head_inputs.clear()
            second = model(dict(inputs, maps=inputs['maps']*3+2), torch.tensor(history))
    finally:
        for hook in hooks:
            hook.remove()
    assert len(contexts) == 2  # Two calls, never one RF encoding per rollout step.
    assert not torch.equal(contexts[0], contexts[1])
    assert not torch.equal(first_head, head_inputs[0])
    assert not torch.equal(first['p'], second['p'])


def test_relative_feature_bound_holds_at_every_rollout_step():
    base, history, inputs = make_fixture(horizons=(50, 250))
    model = InternalResidualFusion(base).eval()
    with torch.no_grad():
        for head in model.residual.heads.values():
            head.head[-1].bias.fill_(1000.)
        prediction = model(inputs, torch.tensor(history))
    assert torch.all(prediction['rollout_relative_residual_norm'] <= model.config.max_relative_delta+1e-7)
    torch.testing.assert_close(prediction['rollout_relative_residual_norm'],
                               torch.full_like(prediction['rollout_relative_residual_norm'], .1))


def test_masked_maps_and_timing_cannot_contaminate_forward_or_backward():
    base, history, inputs = make_fixture(horizons=(50, 150))
    model = activate(InternalResidualFusion(base).train(), spatial=True)
    inputs['mask'][:, 1:3] = False
    clean = model(inputs, torch.tensor(history))
    dirty = {key: value.clone() for key, value in inputs.items()}
    dirty['maps'][:, 1:3] = float('nan')
    dirty['timing'][:, 1:3] = float('nan')
    contaminated = model(dirty, torch.tensor(history))
    for key in ('p', 'q', 'penalty'):
        torch.testing.assert_close(contaminated[key], clean[key], rtol=0, atol=0)
    contaminated['p'].square().sum().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_missing_rf_row_is_unassisted_independently_of_other_batch_rows():
    base, history, inputs = make_fixture(horizons=(50, 250))
    model = activate(InternalResidualFusion(base).eval(), spatial=True)
    inputs['rf_available'][1] = False
    reference = base.forecast_batch(history, [50, 250], initial_alpha=inputs['alpha'].numpy())
    with torch.no_grad():
        prediction = model(inputs, torch.tensor(history))
        alone = model({key: value[1:] for key, value in inputs.items()}, torch.tensor(history[1:]))
    assert prediction['applied'][0].all() and not prediction['applied'][1].any()
    assert not alone['applied'].any()
    p, q = reference_prediction(model, prediction, inputs['rf_available'],
                                reference['positions_m'], reference['quaternions_wxyz'])
    np.testing.assert_array_equal(p[1], reference['positions_m'][1])
    np.testing.assert_array_equal(q[1], reference['quaternions_wxyz'][1])
    torch.testing.assert_close(prediction['p'][1:], alone['p'], rtol=1e-5, atol=1e-6)


def test_hmd_capacity_control_ignores_all_rf_content_and_availability():
    base, history, inputs = make_fixture(horizons=(50, 150))
    model = activate(InternalResidualFusion(base, ResidualConfig(modality='hmd')).eval(), spatial=True)
    changed = {key: value.clone() for key, value in inputs.items()}
    changed['maps'].fill_(1234.)
    changed['timing'].fill_(.7)
    changed['mask'].zero_()
    changed['rf_available'].zero_()
    with torch.no_grad():
        a, b = model(inputs, torch.tensor(history)), model(changed, torch.tensor(history))
    torch.testing.assert_close(a['p'], b['p'], rtol=0, atol=0)
    torch.testing.assert_close(a['q'], b['q'], rtol=0, atol=0)
    assert b['applied'].all()


def test_future_rf_cannot_change_internal_residual(tmp_path):
    base, history, inputs = make_fixture(batch=1, horizons=(50, 150))
    model = activate(InternalResidualFusion(base).eval(), spatial=True)
    end = np.arange(.2, 2., .05)
    predictions = []
    for version in range(2):
        maps = np.arange(len(end)*5*4).reshape(len(end), 5, 4).astype(float)+1
        maps[end > 1.2] *= 1+1000000*version
        metadata = {'format_version': 1, 'unit': 'linear', 'representation': 'aoa_tof',
                    'physical_axes_verified': True, 'clock_domain': 'unix_seconds', 'availability_source': 'test'}
        path = tmp_path/f'{version}.npz'
        np.savez(path, maps=maps, valid=np.ones(len(end), bool), support_start_s=end-.2,
                 support_end_s=end, available_at_s=end+.02, theta_deg=np.linspace(-90, 90, 5),
                 tau_s=np.linspace(0, 3e-9, 4), metadata_json=np.array(json.dumps(metadata)))
        window = encode_window(HeatmapSeries(path), 1.2, {'median': 0., 'scale': 1.}, InputConfig(size=8))
        current = dict(inputs, **{key: torch.tensor(np.asarray(value)[None]) for key, value in window.items()})
        with torch.no_grad():
            predictions.append(model(current, torch.tensor(history)))
    for key in ('p', 'q', 'rollout_alpha'):
        torch.testing.assert_close(predictions[0][key], predictions[1][key], rtol=0, atol=0)


def online_fixture(monkeypatch, *, active=True):
    import baf_roi_heatmap_residual.runtime as runtime
    base, history, inputs = make_fixture(batch=1, horizons=(50, 150))
    reference, _, _ = make_fixture(batch=1, horizons=(50, 150))
    online = HeatmapResidualPredictor.__new__(HeatmapResidualPredictor)
    online.base, online.device, online.cfg = base, 'cpu', InputConfig(size=8)
    online.model = InternalResidualFusion(base).eval()
    if active:
        activate(online.model)
    online.bundle = {'horizons_ms': [50, 150], 'grid_id': 'synthetic',
                     'allow_exploratory': False, 'normalizer': {'median': 0., 'scale': 1.},
                     'rf_age_basis': 'support_end'}
    rf = SimpleNamespace(grid_id='synthetic', metadata={'physical_axes_verified': True})
    window = {key: inputs[key][0].numpy() for key in ('maps', 'mask', 'timing', 'rf_available')}
    def encode(series, *args):
        return dict(window, rf_available=np.bool_(series is not None))
    monkeypatch.setattr(runtime, 'encode_window', encode)
    return online, reference, history[0], rf


def test_online_active_rf_runs_one_assisted_rollout_then_outage_is_exact_baseline(monkeypatch):
    online, reference, history, rf = online_fixture(monkeypatch)
    calls = {'baseline': 0, 'assisted': 0}
    baseline_forecast = online.base.forecast_batch
    def count_baseline(*args, **kwargs):
        calls['baseline'] += 1
        return baseline_forecast(*args, **kwargs)
    monkeypatch.setattr(online.base, 'forecast_batch', count_baseline)
    hook = online.model.register_forward_hook(lambda *args: calls.__setitem__('assisted', calls['assisted']+1))
    try:
        for index, offset in enumerate((0., 1/30, .5)):
            times = np.arange(30)/30+offset
            expected = reference.predict_observations(times, history[:, :3], history[:, 3:],
                                                       horizons_ms=[50, 150], stream_id='a')
            actual = online.predict_observations(times, history[:, :3], history[:, 3:],
                                                 rf if index == 0 else None, stream_id='a')
            assert online.base.observed_state == reference.observed_state
            if index == 0:
                assert actual['adapted'] and calls == {'baseline': 0, 'assisted': 1}
            else:
                assert not actual['adapted']
                np.testing.assert_array_equal(actual['positions_m'], expected['positions_m'])
                np.testing.assert_array_equal(actual['quaternions_wxyz'], expected['quaternions_wxyz'])
        assert calls == {'baseline': 2, 'assisted': 1}
    finally:
        hook.remove()
    saved = online.base.observed_state.copy()
    with pytest.raises(ValueError):
        online.predict_observations(times, history[:, :3], history[:, 3:], rf, stream_id='a')
    assert online.base.observed_state == saved
    online.reset('a')
    assert 'a' not in online.base.observed_state


def test_online_zero_initialization_is_exact_and_fails_transactionally(monkeypatch):
    online, reference, history, rf = online_fixture(monkeypatch, active=False)
    times = np.arange(30)/30
    expected = reference.predict_observations(times, history[:, :3], history[:, 3:], [50, 150], stream_id='a')
    actual = online.predict_observations(times, history[:, :3], history[:, 3:], rf, stream_id='a')
    assert not actual['adapted']
    np.testing.assert_array_equal(actual['positions_m'], expected['positions_m'])
    np.testing.assert_array_equal(actual['quaternions_wxyz'], expected['quaternions_wxyz'])
    saved = online.base.observed_state.copy()
    activate(online.model)
    def fail(*args, **kwargs):
        raise FloatingPointError('injected internal decoder error')
    monkeypatch.setattr(online.model, 'forward', fail)
    with pytest.raises(FloatingPointError):
        online.predict_observations(times+1/30, history[:, :3], history[:, 3:], rf, stream_id='a')
    assert online.base.observed_state == saved


def test_online_hmd_capacity_control_remains_active_without_rf(monkeypatch):
    online, _, history, _ = online_fixture(monkeypatch, active=False)
    online.model = activate(InternalResidualFusion(
        online.base, ResidualConfig(modality='hmd')).eval())
    def forbidden(*args, **kwargs):
        raise AssertionError('An active HMD control must run its residual inside BAF.')
    monkeypatch.setattr(online.base, 'forecast_batch', forbidden)
    prediction = online.predict_observations(np.arange(30)/30, history[:, :3], history[:, 3:],
                                             rf_series=None)
    assert not prediction['rf_available'] and prediction['adapted']
    assert prediction['execution_path'] == 'internal_residual_single_rollout'


@pytest.mark.parametrize('modality', ['rf', 'hmd'])
@pytest.mark.parametrize('global_zero', [True, False])
def test_reference_prediction_uses_runtime_fallback_decision_not_applied_flag(modality, global_zero):
    base, history, inputs = make_fixture(horizons=(50,))
    model = InternalResidualFusion(base, ResidualConfig(modality=modality)).eval()
    if not global_zero:
        activate(model)
    # Distinct reference values make an accidental restore detectable even when
    # an implementation happens to produce numerically identical no-op poses.
    predicted_p = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)
    predicted_q = torch.tensor([[[1., 0., 0., 0.]], [[1., 0., 0., 0.]]])
    prediction = {'p': predicted_p, 'q': predicted_q,
                  'applied': torch.zeros(2, 1, dtype=torch.bool)}
    original_p = predicted_p.numpy().astype(np.float64)+10.
    original_q = np.array([[[0., 1., 0., 0.]], [[0., 1., 0., 0.]]])
    available = torch.tensor([True, False])
    p, q = reference_prediction(model, prediction, available, original_p, original_q)
    for row in range(2):
        uses_reference = global_zero or (modality == 'rf' and not available[row])
        np.testing.assert_array_equal(p[row], original_p[row] if uses_reference else predicted_p[row].numpy())
        np.testing.assert_array_equal(q[row], original_q[row] if uses_reference else predicted_q[row].numpy())


def test_exact_input_specific_cancellation_retains_single_torch_online_and_evaluation_path(monkeypatch):
    import baf_roi_heatmap_residual.runtime as runtime
    online, baseline, history, radio = online_fixture(monkeypatch, active=False)
    times = np.arange(30)/30
    window = runtime.encode_window(radio)
    inputs = {key: torch.tensor(np.asarray(value)[None]) for key, value in window.items()}
    with torch.no_grad():
        context = online.model.residual.encode(inputs)
        different = online.model.residual.encode(dict(inputs, maps=inputs['maps']*3+2))
        coordinate = int((different-context).abs().argmax())
        assert different[0, coordinate] != context[0, coordinate]
        for head in online.model.residual.heads.values():
            # This nonzero learned projection cancels exactly for this RF
            # context, but not for another one. Its output is zero at every
            # rollout step because the causal RF context is held fixed.
            first, last = head.head[0], head.head[-1]
            first.weight.zero_()
            first.bias.zero_()
            first.weight[0, last.out_features+coordinate] = 1.
            first.bias[0] = -context[0, coordinate]
            last.weight.zero_()
            last.bias.zero_()
            last.weight[:, 0] = 1.
    assert not online.model.is_zero_residual()
    original = baseline.predict_observations(times, history[:, :3], history[:, 3:], [50, 150])
    def forbidden(*args, **kwargs):
        raise AssertionError('Input-specific cancellation must not rerun the original BAF.')
    monkeypatch.setattr(online.base, 'forecast_batch', forbidden)
    captured = []
    hook = online.model.register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        actual = online.predict_observations(times, history[:, :3], history[:, 3:], radio)
    finally:
        hook.remove()
    assert len(captured) == 1 and not captured[0]['applied'].any()
    assert not actual['adapted']
    assert actual['execution_path'] == 'internal_residual_single_rollout'
    p, q = reference_prediction(online.model, captured[0], np.array([True]),
                                original['positions_m'], original['quaternions_wxyz'])
    np.testing.assert_array_equal(p, actual['positions_m'])
    np.testing.assert_array_equal(q, actual['quaternions_wxyz'])
    inputs.update(alpha=torch.tensor(actual['alpha_t']),
                  horizon_s=torch.tensor([[.05, .15]]))
    with torch.no_grad():
        changed = online.model(dict(inputs, maps=inputs['maps']*3+2), torch.tensor(history[None]))
    assert changed['applied'].all()


def test_loading_rejects_wrong_baseline_before_constructing_model(tmp_path):
    base = tmp_path/'base.pt'
    base.write_bytes(b'identity check happens before loading a baseline')
    checkpoint = tmp_path/'residual.pt'
    torch.save({'format_version': 1, 'method': 'baf_roi_heatmap_residual',
                'baseline_sha256': 'different'}, checkpoint)
    with pytest.raises(ValueError, match='baseline|Baseline|identity'):
        HeatmapResidualPredictor(base, checkpoint)


@pytest.mark.parametrize('value', [0., -.1, float('nan'), float('inf')])
def test_invalid_relative_residual_bound_is_rejected(value):
    with pytest.raises(ValueError):
        ResidualConfig(max_relative_delta=value).validate()
