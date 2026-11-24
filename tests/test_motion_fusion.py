"""Internal CSI motion prediction, causal routing, controls and selection limits."""
import json

import numpy as np
import pytest
import torch

from baf_roi.models import build_models
from baf_roi.pose import model_inputs
from baf_roi.runtime import Predictor
from baf_roi_heatmap.contracts import ModelConfig
from baf_roi_heatmap.losses import numpy_prediction
from baf_roi_heatmap.motion import MotionFusion, fuse_experts
from baf_roi_heatmap.motion_training import DEFAULT_LIMITS, validation_eligibility
from baf_roi_heatmap.training_utils import CheckpointSelection


def fixture(experts=3, modality='rf', conditioning='joint', batch=2, hs=(50, 100, 250)):
    torch.set_num_threads(2)
    torch.manual_seed(19)
    base = Predictor(build_models(), {'mean': [1., 0., 1.5], 'std': [.1, .1, .1]})
    t = np.arange(30)/30
    history = np.stack((1+.05*t, .02*t, 1.5+.01*t, 5*t, t, -t), -1)
    histories = np.tile(history[None], (batch, 1, 1))
    initial = np.linspace(.3, .7, batch).astype(np.float32)
    ref = base.forecast_batch(histories, hs, initial_alpha=initial)
    _, pose = model_inputs(histories, base.stats)
    inputs = {'hmd': torch.tensor(pose), 'maps': torch.randn(batch, 8, 3, 8, 8),
              'mask': torch.ones(batch, 8, dtype=torch.bool), 'timing': torch.zeros(batch, 8, 2),
              'rf_available': torch.ones(batch, dtype=torch.bool), 'alpha': torch.tensor(initial),
              'horizon_s': torch.tensor(hs).float()[None].expand(batch, -1)/1000,
              'base_p': torch.tensor(ref['positions_m']).float(), 'base_q': torch.tensor(ref['quaternions_wxyz']).float()}
    config = ModelConfig(method='motion_moe', experts=experts, hidden=8, modality=modality, conditioning=conditioning)
    model = MotionFusion(base, config).eval()
    return base, model, torch.tensor(histories), inputs, ref


def activate(model):
    model.baseline_only.fill_(False)
    # Make latent conditioning nonzero as after training, to test actual sensing
    # rather than falsely concluding RF is ignored from zero initialization.
    with torch.no_grad():
        for expert in model.motion_experts:
            expert.film[-1].weight.normal_(std=.15)
        model.router[-1].weight.normal_(std=.1)
    return model


@pytest.mark.parametrize('experts', [2, 3, 4])
def test_epoch_zero_and_missing_rf_keep_exact_baseline(experts):
    _, model, h, x, ref = fixture(experts)
    for missing in (False, True):
        if missing:
            activate(model); x['rf_available'][:] = False
        pred = model(x, h)
        p, q = numpy_prediction(pred, ref['positions_m'], ref['quaternions_wxyz'])
        np.testing.assert_array_equal(p, ref['positions_m'])
        np.testing.assert_array_equal(q, ref['quaternions_wxyz'])
        assert not pred['applied'].any()


@pytest.mark.parametrize('experts', [2, 3, 4])
def test_csi_changes_expert_predictions_before_fusion_and_changes_next_history(experts):
    _, model, h, x, _ = fixture(experts=experts)
    activate(model)
    # Hold routing fixed. CSI must still alter the expert's next-pose candidate.
    for parameter in model.router.parameters(): parameter.data.zero_()
    first = model(x, h)
    altered = {**x, 'maps': x['maps']*0+4}
    second = model(altered, h)
    assert not torch.allclose(first['rollout_positions_m'][:, 1], second['rollout_positions_m'][:, 1], atol=1e-8, rtol=0)
    assert not torch.allclose(first['p'], second['p'], atol=1e-8, rtol=0)
    # A completed BAF endpoint is not an input to active motion prediction.
    poisoned = {**x, 'base_p': x['base_p']+100, 'base_q': -x['base_q']}
    torch.testing.assert_close(first['p'], model(poisoned, h)['p'], rtol=0, atol=0)
    assert first['rollout_expert_weights'].shape == (2, 8, experts)
    torch.testing.assert_close(first['expert_weights'].sum(-1), torch.ones(2, 3))
    torch.testing.assert_close(first['q'].norm(dim=-1), torch.ones(2, 3, dtype=torch.float64))


def test_router_only_holds_experts_fixed_and_has_no_conditioning_effect_when_router_fixed():
    _, model, h, x, _ = fixture(conditioning='router_only', experts=2)
    activate(model)
    assert all(not p.requires_grad for e in model.motion_experts for p in e.parameters())
    for parameter in model.router.parameters(): parameter.data.zero_()
    first = model(x, h)
    second = model({**x, 'maps': x['maps']*0+100}, h)
    torch.testing.assert_close(first['p'], second['p'], rtol=0, atol=0)
    torch.testing.assert_close(first['q'], second['q'], rtol=0, atol=0)


@pytest.mark.parametrize('experts', [2, 3, 4])
def test_active_warm_start_preserves_baf_prediction_before_learning(experts):
    _, model, h, x, ref = fixture(experts=experts)
    model.baseline_only.fill_(False)
    pred = model(x, h)
    np.testing.assert_allclose(pred['p'].detach().numpy(), ref['positions_m'], atol=2e-5)
    # Quaternion signs represent the same rotation.
    q = pred['q'].detach().numpy()
    dot = np.abs((q*ref['quaternions_wxyz']).sum(-1))
    np.testing.assert_allclose(dot, 1., atol=2e-5)


def test_hmd_control_ignores_every_rf_input_and_availability():
    _, model, h, x, _ = fixture(modality='hmd')
    activate(model)
    first = model(x, h)
    altered = {**x, 'maps': torch.full_like(x['maps'], float('nan')), 'timing': x['timing']+1e6,
               'mask': torch.zeros_like(x['mask']), 'rf_available': torch.zeros_like(x['rf_available'])}
    second = model(altered, h)
    torch.testing.assert_close(first['p'], second['p'], rtol=0, atol=0)
    assert second['applied'].all()


def test_mixed_batch_fallback_and_invalid_rf_pixels_do_not_change_valid_rows():
    _, model, h, x, ref = fixture()
    activate(model)
    x['rf_available'][1] = False
    x['mask'][:, :3] = False
    first = model(x, h)
    altered = {**x, 'maps': x['maps'].clone(), 'timing': x['timing'].clone()}
    altered['maps'][:, :3] = float('nan'); altered['timing'][:, :3] = float('nan')
    second = model(altered, h)
    torch.testing.assert_close(first['p'], second['p'], rtol=0, atol=0)
    p, q = numpy_prediction(second, ref['positions_m'], ref['quaternions_wxyz'])
    np.testing.assert_array_equal(p[1], ref['positions_m'][1])
    np.testing.assert_array_equal(q[1], ref['quaternions_wxyz'][1])


def test_joint_training_reaches_radio_conditioners_router_and_native_pose_heads():
    _, model, h, x, _ = fixture()
    activate(model).train()
    pred = model(x, h)
    loss = (pred['p']-x['base_p']-.01).square().mean()+.01*(pred['q']-x['base_q']).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.context.spatial[0].weight.grad.abs().sum() > 0
    assert model.router[-1].weight.grad.abs().sum() > 0
    for expert in model.motion_experts:
        assert expert.film[-1].weight.grad.abs().sum() > 0
        assert expert.branch.translation_head[-1].weight.grad.abs().sum() > 0
        assert expert.branch.rotation_head[-1].weight.grad.abs().sum() > 0
        assert not expert.branch.training
    assert all(p.grad is None for p in model.confidence.parameters())
    assert all(p.grad is None for p in model.motion_experts[0].branch.encoder.parameters())


def test_quaternion_fusion_respects_sign_and_has_finite_gradients():
    p = torch.randn(2, 4, 3, requires_grad=True)
    q = torch.randn(2, 4, 4, requires_grad=True)
    w = torch.softmax(torch.randn(2, 4), -1)
    out_p, out_q = fuse_experts(p, q, w)
    signs = torch.tensor([1., -1., 1., -1.])[None, :, None]
    _, signed = fuse_experts(p, q*signs, w)
    torch.testing.assert_close((out_q*signed).sum(-1).abs(), torch.ones(2))
    (out_p.square().mean()+out_q.square().mean()).backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()


def test_each_horizon_validation_limit_is_required_even_if_mean_iou_improves(tmp_path):
    baseline = {str(h): {'iou_mean': .7, 'coverage_micro': .9, 'translation_mean_m': .02,
                        'rotation_mean_deg': 2.} for h in (50, 1000)}
    good = {h: {**v, 'iou_mean': .8} for h, v in baseline.items()}
    assert validation_eligibility(good, baseline, DEFAULT_LIMITS) == (True, [])
    for field, bad_value in [('coverage_micro', .89), ('translation_mean_m', .04), ('rotation_mean_deg', 3.)]:
        bad = {h: dict(v) for h, v in good.items()}; bad['50'][field] = bad_value
        eligible, reasons = validation_eligibility(bad, baseline, DEFAULT_LIMITS)
        assert not eligible and reasons[0].startswith('50ms:')
    selection = CheckpointSelection(tmp_path, 'adapter.pt')
    selection.update({'selected_epoch': 0, 'validation_recording_macro_iou': .7})
    result = selection.update({'selected_epoch': 1, 'validation_recording_macro_iou': .9, 'selection_eligible': False})
    assert result['selected_epoch'] == 0 and result['best_trained_epoch'] == 1
    result = selection.update({'selected_epoch': 2, 'validation_recording_macro_iou': .8, 'selection_eligible': True})
    assert result['selected_epoch'] == 2


@pytest.mark.parametrize('experts', [1, 5, 2.5, True])
def test_invalid_expert_counts_are_rejected(experts):
    with pytest.raises(ValueError): ModelConfig(method='motion_moe', experts=experts).validate()


def test_cli_defaults_select_internal_motion_and_explicit_controls():
    from roi_experiments.runner import parser
    args = parser().parse_args(['--output', 'unused'])
    assert args.heatmap_adapter == 'motion_moe' and args.motion_experts == 3
    assert args.heatmap_objective is None  # resolves to pose_roi for motion_moe
    args = parser().parse_args(['--output', 'unused', '--motion-experts', '2', '--motion-conditioning', 'router_only', '--modality', 'hmd'])
    assert args.motion_experts == 2 and args.motion_conditioning == 'router_only' and args.modality == 'hmd'


def test_online_runs_internal_experts_without_completed_baf_forecast_and_keeps_observed_state(tmp_path, monkeypatch):
    from dataclasses import asdict
    from baf_roi_heatmap.contracts import InputConfig, file_hash
    from baf_roi_heatmap.runtime import HeatmapPredictor
    import baf_roi_heatmap.motion_runtime as runtime
    base, model, h, x, _ = fixture(batch=1)
    activate(model)
    baseline = tmp_path/'baf.pt'
    torch.save({'format_version': 2, 'models': {k: m.state_dict() for k, m in base.models.items()},
                'normalization': base.stats, 'config': {'data': base.data_config, 'model': {'temporal_mu': base.mu}}}, baseline)
    checkpoint = tmp_path/'adapter.pt'
    torch.save({'format_version': 1, 'model': model.state_dict(), 'model_config': asdict(model.config),
                'baseline_sha256': file_hash(baseline), 'horizons_ms': [50, 100, 250],
                'input_config': asdict(InputConfig(size=8)), 'grid_id': 'test', 'allow_exploratory': False,
                'normalizer': {'median': 0., 'scale': 1.}}, checkpoint)
    predictor = HeatmapPredictor(baseline, checkpoint)
    window = {key: x[key][0].numpy() for key in ('maps', 'mask', 'timing', 'rf_available')}
    monkeypatch.setattr(runtime, 'encode_window', lambda *args: window)
    ts = 100.+np.arange(30)/30
    history = h[0].numpy()
    original_forecast = predictor.base.forecast_batch
    monkeypatch.setattr(predictor.base, 'forecast_batch', lambda *a, **k: pytest.fail('Active CSI must not first complete BAF rollout'))
    result = predictor.predict_observations(ts, history[:, :3], history[:, 3:], stream_id='s')
    assert result['adapted'] and result['expert_weights'].shape == (1, 3, 3)
    observed = float(base.confidence(history[None])[0])
    assert predictor.base.observed_state['s'][1] == pytest.approx(observed)
    state = predictor.base.observed_state.copy()
    with pytest.raises(ValueError, match='horizons'):
        predictor.predict_observations(ts+.03, history[:, :3], history[:, 3:], horizons_ms=[75], stream_id='s')
    assert predictor.base.observed_state == state
    with pytest.raises(ValueError, match='strictly increase'):
        predictor.predict_observations(ts, history[:, :3], history[:, 3:], stream_id='s')
    assert predictor.base.observed_state == state
    monkeypatch.setattr(predictor.base, 'forecast_batch', original_forecast)
    window['rf_available'] = False
    base.observed_state = state.copy()
    reference = base.predict_observations(ts+1/30, history[:, :3], history[:, 3:], [50, 100, 250], 's')
    fallback = predictor.predict_observations(ts+1/30, history[:, :3], history[:, 3:], stream_id='s')
    assert not fallback['adapted']
    np.testing.assert_array_equal(fallback['positions_m'], reference['positions_m'])
    np.testing.assert_array_equal(fallback['quaternions_wxyz'], reference['quaternions_wxyz'])
    predictor.reset('s'); assert 's' not in predictor.base.observed_state


def test_training_plot_does_not_label_validation_ineligible_epoch_as_best(tmp_path):
    from roi_experiments.visualize import training_histories
    directory = tmp_path/'models/baf_roi_heatmap'; directory.mkdir(parents=True)
    (directory/'training.json').write_text(json.dumps({'history': [
        {'epoch': 0, 'validation_recording_macro_iou': .7, 'selection_eligible': True},
        {'epoch': 1, 'validation_recording_macro_iou': .9, 'selection_eligible': False},
        {'epoch': 2, 'validation_recording_macro_iou': .8, 'selection_eligible': True}]}))
    history = training_histories(tmp_path, ['baf_roi_heatmap'])[0]
    assert history['best_history_epoch'] == 2 and history['best_validation'] == .8
    assert history['eligibility_filtered']
