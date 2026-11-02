import json
from dataclasses import replace
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from baf_roi.roi import Scene
from baf_roi_heatmap.contracts import InputConfig, ModelConfig
from baf_roi_heatmap.data import HeatmapSeries, encode_window, to_db, fit_normalizer
from baf_roi_heatmap.music import smooth_covariance, steering_rows, music_spectrum, validate_csi
from baf_roi_heatmap.model import ForecastAdapter, slerp, rotation_vector_quaternion, rotation_matrix, bounded_vector, ablate_maps
from baf_roi_heatmap.losses import task_loss, numpy_prediction
from baf_roi_heatmap.evaluation import margin_scene, roi_counts, summarize
from baf_roi_heatmap.prepare import observed_weights


def maps_file(path, shift=0., future_scale=1., unit='linear', bad_support=False):
    end = np.arange(.2, 2., .05)
    maps = np.ones((len(end), 5, 4)) * (2+np.arange(len(end)))[:, None, None]
    maps[end > 1.2] *= future_scale
    metadata = {'format_version': 1, 'unit': unit, 'representation': 'aoa_tof',
                'physical_axes_verified': True, 'clock_domain': 'unix_seconds', 'availability_source': 'test'}
    np.savez(path, maps=maps+shift, valid=np.ones(len(end), bool), support_start_s=end-(10 if bad_support else .2),
             support_end_s=end, available_at_s=end+.02, theta_deg=np.linspace(-90, 90, 5), tau_s=np.linspace(0, 3e-9, 4),
             metadata_json=np.array(json.dumps(metadata)))
    return HeatmapSeries(path)


def inputs(batch=2, horizons=3):
    q = torch.zeros(batch, horizons, 4); q[..., 0] = 1
    return {'hmd': torch.randn(batch, 30, 7), 'maps': torch.randn(batch, 8, 3, 24, 24),
            'mask': torch.ones(batch, 8, dtype=torch.bool), 'timing': torch.zeros(batch, 8, 2),
            'rf_available': torch.ones(batch, dtype=torch.bool), 'horizon_s': torch.linspace(.05, 1., horizons)[None].expand(batch, -1),
            'base_p': torch.zeros(batch, horizons, 3), 'base_q': q.clone(),
            'focused_p': torch.zeros(batch, horizons, 3), 'focused_q': q.clone(),
            'explorer_p': torch.ones(batch, horizons, 3)*.1, 'explorer_q': q.clone(), 'alpha': torch.ones(batch)*.3}


def scene(mode='legacy_center_cone'):
    cfg = {'visibility_mode': mode, 'fov_deg': 110, 'horizontal_fov_deg': 100, 'vertical_fov_deg': 90, 'near_m': .01}
    return Scene(np.array([[1., 0, 0], [1, 2, 0], [-1, 0, 0], [1, .5, .5]]), ['a', 'b', 'c', 'd'], cfg)


@pytest.mark.parametrize('field,value', [('frames', 1), ('size', 2), ('raw_history_s', 2), ('min_valid_fraction', 1.1), ('clip', float('nan')), ('max_age_s', -1)])
def test_bad_input_config(field, value):
    with pytest.raises(ValueError): replace(InputConfig(), **{field: value}).validate()


@pytest.mark.parametrize('method', ['residual', 'expert_gate'])
@pytest.mark.parametrize('modality', ['rf', 'hmd'])
def test_model_shapes_and_finite_gradients(method, modality):
    torch.set_num_threads(2)
    model = ForecastAdapter(ModelConfig(method, modality))
    x = inputs(); y = {'target_p': x['base_p']+.03, 'target_q': x['base_q'], 'target_mask': torch.ones(2, 3, 4)}
    pred = model(x)
    loss = task_loss(pred, y, scene())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert pred['p'].shape == (2, 3, 3)
    assert torch.allclose(pred['q'].norm(dim=-1), torch.ones(2, 3))


def test_zero_initialization_and_exact_numpy_fallback():
    model = ForecastAdapter(); x = inputs()
    pred = model(x)
    p0 = np.full((2, 3, 3), 1.123456789123)
    q0 = np.tile([1., 0, 0, 0], (2, 3, 1))
    p, q = numpy_prediction(pred, p0, q0)
    np.testing.assert_array_equal(p, p0); np.testing.assert_array_equal(q, q0)


def test_missing_rf_fallback_even_after_nonzero_head():
    model = ForecastAdapter(); model.head[-1].bias.data.fill_(1.)
    x = inputs(); x['rf_available'][:] = False
    pred = model(x)
    assert torch.equal(pred['p'], x['base_p']) and torch.equal(pred['q'], x['base_q'])
    assert not pred['applied'].any()


def test_hmd_control_ignores_rf_pixels_and_mask():
    model = ForecastAdapter(ModelConfig(modality='hmd')); model.head[-1].bias.data.fill_(.2)
    x = inputs(); first = model(x)['p']
    x['maps'] *= 100; x['mask'][:] = False; x['rf_available'][:] = False
    assert torch.equal(first, model(x)['p'])


def test_quaternion_sign_and_small_angle_gradients():
    q = torch.tensor([[1., 0, 0, 0]], requires_grad=True)
    mixed = slerp(q, -q, torch.tensor([.7]))
    assert torch.allclose(mixed, q)
    mixed.sum().backward(); assert torch.isfinite(q.grad).all()
    v = torch.zeros(2, 3, requires_grad=True)
    rotation_vector_quaternion(v).sum().backward()
    assert torch.isfinite(v.grad).all()


def test_rotation_matrix_matches_scipy():
    v = np.array([[.1, -.2, .3], [0, 0, 0]])
    q = rotation_vector_quaternion(torch.tensor(v, dtype=torch.float64))
    np.testing.assert_allclose(rotation_matrix(q).numpy(), Rotation.from_rotvec(v).as_matrix(), atol=1e-12)


def test_bounded_vector_norm():
    result = bounded_vector(torch.ones(100, 3)*100, .2)
    assert torch.all(result.norm(dim=-1) <= .200001)


@pytest.mark.parametrize('ablation', ['full', 'spatial_mean', 'angle_only', 'delay_only', 'last_frame', 'abs_change', 'metadata_only'])
def test_ablation_shapes(ablation):
    x = inputs(); m, mask = ablate_maps(x['maps'], x['mask'], ablation)
    assert m.shape == x['maps'].shape and mask.shape == x['mask'].shape
    if ablation == 'last_frame':
        assert not mask[:, :-1].any() and not m[:, :, 1:].any()


def test_db_contract():
    np.testing.assert_allclose(to_db([1, 10, 100], 'linear'), [0, 10, 20])
    np.testing.assert_array_equal(to_db([-50, -10], 'db'), [-50, -10])
    with pytest.raises(ValueError): to_db([-1], 'linear')
    with pytest.raises(ValueError): to_db([np.nan], 'db')


def test_future_mutation_cannot_change_inputs(tmp_path):
    a, b = maps_file(tmp_path/'a.npz'), maps_file(tmp_path/'b.npz', future_scale=100000)
    norm = {'median': 0., 'scale': 1.}
    x, y = (encode_window(v, 1.2, norm, InputConfig()) for v in (a, b))
    for k in x: np.testing.assert_array_equal(x[k], y[k])
    assert x['rf_available']


def test_no_future_delivery_or_excess_raw_history(tmp_path):
    series = maps_file(tmp_path/'maps.npz')
    queries, ids = series.indices(1.2, InputConfig())
    good = ids >= 0
    assert np.all(series.ready[ids[good]] <= queries[good]+1e-9)
    assert np.all(series.start[ids[good]] >= 1.2-29/30-1e-9)
    assert np.all(series.end[ids[good]] <= queries[good]+1e-9)
    assert not encode_window(series, 1.2, {'median': 0., 'scale': 1.}, InputConfig(), 2.)['rf_available']
    bad = maps_file(tmp_path/'bad.npz', bad_support=True)
    assert not encode_window(bad, 1.2, {'median': 0., 'scale': 1.}, InputConfig())['mask'].any()


def test_uncertain_clock_and_negative_delay(tmp_path):
    series = maps_file(tmp_path/'maps.npz'); series.uncertainty = 1.
    assert not (series.indices(1.2, InputConfig())[1] >= 0).any()
    with pytest.raises(ValueError): series.indices(1.2, InputConfig(), -1)


def test_training_only_normalization(tmp_path):
    maps_file(tmp_path/'train.npz'); maps_file(tmp_path/'test.npz', shift=1000)
    records = [{'key': k, 'split': k, 'file': str(tmp_path/(k+'.npz')), 'clock_offset_s': 0., 'clock_uncertainty_s': 0.} for k in ('train', 'test')]
    first = fit_normalizer({'records': records})
    maps_file(tmp_path/'test.npz', shift=100000000)
    second = fit_normalizer({'records': records})
    assert first == second and first['record_keys'] == ['train']


def test_covariance_matches_explicit_snapshot_construction():
    rng = np.random.default_rng(42)
    w = rng.normal(size=(3, 2, 4, 7)) + 1j*rng.normal(size=(3, 2, 4, 7))
    snapshots = [w[t, tx, r:r+3][:, f+np.array([0, 2, 4])].reshape(-1)
                 for t in range(3) for tx in range(2) for r in range(2) for f in range(3)]
    x = np.stack(snapshots, 1)
    np.testing.assert_allclose(smooth_covariance(w, 3, 3, 2), x@x.conj().T/x.shape[1], atol=1e-12)


@pytest.mark.parametrize('projection', ['sin', 'legacy_cos'])
def test_music_single_path_peak(projection):
    freq = 5.57e9+np.arange(8)*5e6
    theta = np.array([-60., -30., 0., 30., 60.]) if projection == 'sin' else np.array([0., 30., 60., 90., 120.])
    tau = np.arange(0, 10e-9, 2e-9)
    path = steering_rows(theta[2:3], tau[2:3], 5.57e9, freq, 4, .02, projection).reshape(4, 8)
    data = np.exp(1j*np.arange(8))[:, None, None, None] * path[None, None]
    cov = smooth_covariance(data, 3, 3, 2)
    steering = steering_rows(theta, tau, 5.57e9, freq[np.array([0, 2, 4])], 3, .02, projection)
    spectrum = music_spectrum(cov, steering, (len(theta), len(tau)), 1)
    assert np.unravel_index(np.argmax(spectrum), spectrum.shape) == (2, 2)


def test_csi_requires_timestamps_and_frequency_grid():
    csi = np.ones((4, 1, 3, 4), complex)
    ts = np.arange(4)*.01
    with pytest.raises(ValueError): validate_csi(csi.real, ts, ts, [1, 2, 3, 4])
    with pytest.raises(ValueError): validate_csi(csi, ts, ts, [1, 2, 4, 5])
    with pytest.raises(ValueError): validate_csi(csi, ts, ts-.1, [1, 2, 3, 4])


def test_observed_ema_resets_on_gaps_and_rejects_reverse_time():
    alpha, state = observed_weights([0., 1., 1.], [0., 1/30, 1.], .6, .1)
    np.testing.assert_allclose(alpha, [0, .6, 1])
    with pytest.raises(ValueError): observed_weights([0.], [.5], .6, .1, state)


@pytest.mark.parametrize('mode', ['legacy_center_cone', 'rectangular_center_frustum'])
def test_margin_never_changes_ground_truth(mode):
    base = scene(mode); snapshot = base.snapshot()
    widened = margin_scene(base, 10.)
    p, q = np.zeros((1, 3)), np.array([[1., 0, 0, 0]])
    assert base.snapshot() == snapshot
    assert np.all(~base.mask(p, q) | widened.mask(p, q))


def test_roi_counts():
    tp, fp, fn, tn = roi_counts([[1, 1, 0, 0]], [[1, 0, 1, 0]])
    assert (tp[0], fp[0], fn[0], tn[0]) == (1, 1, 1, 1)
