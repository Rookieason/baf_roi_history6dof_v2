from pathlib import Path
import copy
import csv
import importlib.util
import json
import numpy as np
import pytest
import torch
from baf_roi.audit import verify
from baf_roi.config import load_config
from baf_roi.data import Series, parse_timestamp, read_recording, discover_recordings, make_split, prepare, Cache, OriginDataset
from baf_roi.pose import ypr_to_quat, quat_to_ypr, ypr_to_xyz, model_inputs, interpolate_predictions, quaternion_error_deg
from baf_roi.models import build_models, focused, explorer
from baf_roi.runtime import Predictor
from baf_roi.roi import Scene, grid_centers, load_scene
from baf_roi.supervision import score_resampled_index
from baf_roi.upstream_ops import slerp_quaternion_wxyz, compute_temporal_optimized_alpha, weighted_bce_loss
from baf_roi.train import stage_losses, train
from baf_roi.evaluate import Metrics, binary_counts
from scripts.smoke_test import make_fixture

ROOT = Path(__file__).resolve().parents[1]
STATS = {'mean': [0, 0, 0], 'std': [1, 1, 1]}

@pytest.fixture(autouse=True, scope='session')
def torch_threads():
    torch.set_num_threads(2)

@pytest.fixture(scope='module')
def fixture_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp('recordings')
    cfg = make_fixture(root)
    manifest = prepare(cfg, root / 'cache')
    return root, cfg, manifest

@pytest.fixture(scope='module')
def predictor():
    torch.manual_seed(17)
    return Predictor(build_models(), STATS)


def histories(n=2):
    t = np.arange(30) / 30
    h = np.zeros((n, 30, 6))
    for i in range(n):
        h[i, :, :3] = np.column_stack((.2 * t + .1 * i, .1 * t, 1.5 + .01 * t))
        h[i, :, 3] = 20 * t + i * 4
        h[i, :, 4] = 2 * np.sin(t)
    return h


def test_upstream_sources_and_functions_are_exact():
    result = verify()
    assert result['verified_source_files'] == 67
    assert result['verified_verbatim_functions'] == 4


def test_actual_explorer_run_configuration(predictor):
    assert predictor.models['explorer'].config.aux_dim == 0
    assert predictor.models['focused'].config.input_dim == 7
    total = sum(p.numel() for m in predictor.models.values() for p in m.parameters())
    assert 600000 < total < 630000


@pytest.mark.parametrize('angles', [[0, 0, 0], [90, 0, 0], [0, 45, 0], [20, -30, 70], [179, -60, 89]])
def test_dataset_rotation_conversion(angles):
    q = ypr_to_quat(angles)
    restored = quat_to_ypr(q)
    assert quaternion_error_deg(ypr_to_quat(restored), q) < 1e-8
    assert np.allclose(ypr_to_xyz(angles), [angles[2], -angles[1], angles[0]])


def test_quaternion_sign_is_same_rotation():
    q = ypr_to_quat([70, 50, 20])
    assert quaternion_error_deg(q, -q) == pytest.approx(0)
    assert np.isclose(np.linalg.norm(slerp_quaternion_wxyz(q[None], -q[None], np.array([.5]))), 1.)


def test_slerp_crosses_short_arc():
    q0, q1 = ypr_to_quat([[170, 0, 0], [-170, 0, 0]])
    mid = slerp_quaternion_wxyz(q0[None], q1[None], np.array([.5]))
    assert quaternion_error_deg(mid[0], ypr_to_quat([180, 0, 0])) < 1e-4


def test_fractional_horizons_are_exact():
    ts = np.arange(9) / 30
    p = np.zeros((1, 9, 3)); p[0, :, 0] = 3 * ts
    q = ypr_to_quat(np.column_stack((60 * ts, ts * 0, ts * 0)))[None]
    h = [50, 100, 150, 200, 250]
    ph, qh = interpolate_predictions(p, q, h)
    assert np.allclose(ph[0, :, 0], 3 * np.array(h) / 1000)
    expected = ypr_to_quat(np.column_stack((60 * np.array(h) / 1000, np.zeros(5), np.zeros(5))))
    assert np.max(quaternion_error_deg(qh[0], expected)) < 1e-4


def test_online_history_cannot_see_future():
    t = np.arange(0, 3, 1 / 90)
    p = np.column_stack((t, t * .3, t * 0, t * 20, t * 0, t * 0))
    i = 137
    a = Series(t, p)
    changed = p.copy(); changed[i + 1:] += 50
    b = Series(t, changed)
    assert np.array_equal(a.history(i), b.history(i))
    assert not np.array_equal(a.at([t[i] + .1]), b.at([t[i] + .1]))
    assert np.array_equal(a.history(i), Series(t[:i + 1], p[:i + 1]).history(i))


def test_interpolation_refuses_future_support():
    t = np.arange(0, 2, .01)
    series = Series(t, np.zeros((len(t), 6)))
    with pytest.raises(ValueError, match='beyond'):
        series.at([.505], cutoff_index=50)


def test_gap_and_extrapolation_are_rejected():
    t = np.array([0., .01, .5, .51])
    series = Series(t, np.zeros((4, 6)), .1)
    with pytest.raises(ValueError, match='gap'):
        series.at([.3])
    with pytest.raises(ValueError, match='extrapolation'):
        series.at([.6])


def test_angle_interpolation_unwraps():
    p = np.zeros((2, 6)); p[:, 3] = [179, -179]
    series = Series(np.array([0, .1]), p)
    assert abs(abs(series.at([.05])[0, 3]) - 180) < 1e-8


@pytest.mark.parametrize('value,unit,expected', [('1700000000', 's', 1700000000),
    ('1700000000000', 'ms', 1700000000), ('2023-11-14T22:13:20Z', 's', 1700000000),
    ('2023-11-14T22-13-20.000Z', 's', 1700000000)])
def test_timestamps(value, unit, expected):
    assert parse_timestamp(value, unit) == pytest.approx(expected)


def test_naive_time_is_not_local_timezone_guess():
    with pytest.raises(ValueError, match='timezone'):
        parse_timestamp('2023-11-14T22:13:20')


def write_rows(path, payloads):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['payload_json']); w.writeheader()
        for p in payloads:
            w.writerow({'payload_json': json.dumps(p)})


def test_missing_rotation_is_not_invented(tmp_path):
    file = tmp_path / 'x.csv'
    write_rows(file, [{'ts_utc': 1, 'pos': [1, 2, 3]}])
    with pytest.raises(ValueError, match='rot_deg'):
        read_recording(file, {'position_unit': 'cm'})


def test_duplicate_conflicts_are_rejected(tmp_path):
    file = tmp_path / 'x.csv'
    write_rows(file, [{'ts_utc': 1, 'pos': [i, 0, 0], 'rot_deg': [0, 0, 0]} for i in range(2)])
    with pytest.raises(ValueError, match='conflicting'):
        read_recording(file, {'position_unit': 'cm'})


def test_pose_only_no_heatmaps_and_no_artifacts(fixture_cache):
    root, cfg, manifest = fixture_cache
    assert not (root / 'recordings/artifacts').exists()
    assert len(discover_recordings(root / 'recordings')) == 3
    assert len(manifest['trajectories']) == 3


def test_cache_all_clean_filters_and_units(fixture_cache):
    root, cfg, manifest = fixture_cache
    cache = Cache(root / 'cache'); arrays = cache.arrays(0)
    assert 1.9 < arrays['pose'][0, 0] < 2.1
    for mode, key in [('all', 'all_next'), ('confidence', 'confidence'),
                      ('focused_clean', 'focused_clean'), ('explorer_clean', 'explorer_clean')]:
        ds = OriginDataset(root / 'cache', 'train', mode)
        assert len(ds) == manifest['trajectories'][0]['counts'][key]
        item = ds[0]
        assert (item['conf'] if mode == 'confidence' else item['pose']).shape[0] == 30
    assert manifest['trajectories'][0]['counts']['unlabeled'] > 0


def test_clean_means_current_and_next_not_entire_history(fixture_cache):
    root, _, _ = fixture_cache
    ds = OriginDataset(root / 'cache', 'train', 'explorer_clean')
    for ti, oi in ds.index:
        a = ds.cache.arrays(int(ti))
        assert a['labels'][oi] == 1 and a['next_labels'][oi] == 1


def test_split_overlap_rejected(tmp_path):
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'train': ['a'], 'valid': ['a'], 'test': ['b']}))
    with pytest.raises(ValueError, match='exactly once'):
        make_split(['a', 'b'], {'manifest': str(split)})


def test_split_is_reproducible_and_disjoint():
    ids = [f's{i}' for i in range(20)]
    a = make_split(ids, {'seed': 57})
    assert a == make_split(list(reversed(ids)), {'seed': 57})
    assert (len(a['train']), len(a['valid']), len(a['test'])) == (14, 3, 3)
    assert not set(a['train']) & set(a['test'])


def test_statistics_do_not_use_test_poses(tmp_path):
    cfg = make_fixture(tmp_path)
    a = prepare(cfg, tmp_path / 'a')
    test_file = Path(cfg['data']['dataset_root']) / 'db/session_2/agent6_head/head.csv'
    with test_file.open() as f:
        rows = list(csv.DictReader(f))
    payloads = []
    for row in rows:
        payload = json.loads(row['payload_json']); payload['pos'] = [x + 100000 for x in payload['pos']]
        payloads.append(payload)
    write_rows(test_file, payloads)
    b = prepare(cfg, tmp_path / 'b')
    assert a['normalization'] == b['normalization']
    assert a['cache_id'] != b['cache_id']


def test_unit_contract_must_be_confirmed(tmp_path):
    cfg = load_config(ROOT / 'configs/default.json')
    with pytest.raises(ValueError, match='Confirm position_unit'):
        prepare(cfg, tmp_path / 'not_created')


def test_behavior_is_motion_only():
    t = np.arange(31) / 30
    x = np.zeros((31, 6)); x[:, 2] = 1.6
    assert score_resampled_index(t, x, 15)['Label'] == 'Focused'
    x[:, 0] = .4 * t; x[:, 3] = 200 * t
    result = score_resampled_index(t, x, 15)
    assert result['Label'] == 'Explorer'
    assert result['speed_mean'] == pytest.approx(.4)
    assert np.rad2deg(result['head_rate_yaw_q90']) == pytest.approx(200)
    assert result['yaw_zero_cross_per_s'] == 0
    assert result['head_rate_yaw_hi_prop'] == 1


def test_weighted_bce_matches_released_run_sh():
    logits = torch.zeros(2); targets = torch.tensor([0., 1.])
    loss = weighted_bce_loss(logits, targets, focused_weight=3., explorer_weight=1.)
    assert loss.item() == pytest.approx(2 * np.log(2))


@pytest.mark.parametrize('stage', ['confidence', 'focused_pretrain', 'focused_finetune', 'explorer'])
def test_all_original_loss_paths_have_finite_gradients(stage):
    from baf_roi.train import create_component
    torch.manual_seed(3)
    m = create_component(stage)
    conf, pose = model_inputs(histories(), STATS)
    batch = {'conf': torch.tensor(conf), 'pose': torch.tensor(pose), 'label': torch.tensor([0., 1.]),
             'target_tr': torch.tensor(pose[:, -1, :3] + .01), 'target_q': torch.tensor(pose[:, -1, 3:])}
    loss = stage_losses(stage, m, batch)['loss_total']; loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_history_only_model_input_shapes():
    conf, pose = model_inputs(histories(), STATS)
    assert conf.shape == (2, 30, 6) and pose.shape == (2, 30, 7)
    assert np.allclose(np.linalg.norm(pose[:, :, 3:], axis=-1), 1)


def test_rollout_uses_predicted_history_not_teacher_forcing(monkeypatch, predictor):
    seen = []
    def step(h):
        seen.append(h.copy())
        p = h[:, -1, :3] + np.array([1, 0, 0])
        q = ypr_to_quat(h[:, -1, 3:])
        return np.full(len(h), .7), p, q, p, q
    monkeypatch.setattr(predictor, '_step', step)
    h = histories(1)
    out = predictor.forecast_batch(h)
    assert len(seen) == 8
    assert seen[1][0, -1, 0] == pytest.approx(h[0, -1, 0] + 1)
    assert seen[7][0, -1, 0] == pytest.approx(h[0, -1, 0] + 7)
    assert out['positions_m'][0, -1, 0] == pytest.approx(h[0, -1, 0] + 7.5)


def test_batch_independence_and_horizon_order(predictor):
    h = histories()
    batch = predictor.forecast_batch(h)
    one = predictor.forecast_batch(h[:1], [250, 50, 150, 100, 200])
    indices = [4, 0, 2, 1, 3]
    assert np.allclose(one['positions_m'][0], batch['positions_m'][0, indices], atol=2e-5)
    assert np.max(quaternion_error_deg(one['quaternions_wxyz'][0], batch['quaternions_wxyz'][0, indices])) < .01


def test_temporal_smoothing_is_invariant_to_batch_boundaries():
    raw = np.array([.1, .9, .2, .7], np.float32)
    meta = [{'file_name': 'a'}] * 4
    whole = compute_temporal_optimized_alpha(raw, meta, {}, .6)
    state = {}
    parts = np.r_[compute_temporal_optimized_alpha(raw[:2], meta[:2], state, .6),
                  compute_temporal_optimized_alpha(raw[2:], meta[2:], state, .6)]
    assert np.array_equal(whole, parts)
    assert whole[1] == pytest.approx(.6 * .9 + .4 * .1)


def test_temporal_state_is_isolated_per_stream():
    raw = np.array([.1, .9, .2])
    meta = [{'file_name': 'a'}, {'file_name': 'b'}, {'file_name': 'a'}]
    out = compute_temporal_optimized_alpha(raw, meta, {}, .6)
    assert out[1] == pytest.approx(.9)
    assert out[2] == pytest.approx(.6 * .2 + .4 * .1)


def test_forecast_ema_not_committed_to_real_stream(monkeypatch, predictor):
    def step(h):
        p = h[:, -1, :3] + np.array([1, 0, 0])
        q = ypr_to_quat(h[:, -1, 3:])
        return np.minimum(.99, h[:, -1, 0] / 10 + .1), p, q, p, q
    monkeypatch.setattr(predictor, '_step', step)
    predictor.reset()
    t = np.arange(0, 1.2, .01)
    pos = np.column_stack((t * .1, t * 0, t * 0 + 1.5))
    rot = np.zeros_like(pos)
    result = predictor.predict_observations(t, pos, rot, stream_id='a')
    assert result['rollout_alpha'][0, -1] != pytest.approx(result['alpha_t'][0])
    assert predictor.observed_state['a'][1] == pytest.approx(result['alpha_t'][0])
    with pytest.raises(ValueError, match='strictly increase'):
        predictor.predict_observations(t, pos, rot, stream_id='a')
    predictor.reset()


def load_reference_roi():
    path = ROOT / 'reference/roi_baseline/experiments/roi_heatmap_ablation/data/roi_label.py'
    spec = importlib.util.spec_from_file_location('_test_legacy_roi', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_legacy_roi_is_identical_after_unit_conversion():
    original = load_reference_roi()
    rng = np.random.default_rng(5)
    centers = rng.uniform(-5, 5, (500, 3))
    scene = Scene(centers, [str(i) for i in range(len(centers))],
                  {'visibility_mode': 'legacy_center_cone', 'fov_deg': 110, 'max_distance_m': 4})
    for _ in range(30):
        pos = rng.uniform(-1, 1, 3); ypr = rng.uniform(-160, 160, 3)
        ours = scene.mask(pos, ypr_to_quat(ypr))
        theirs = original.pose_to_roi_mask(pos * 100, ypr, centers * 100, 110, max_distance_cm=400)
        assert np.array_equal(ours, theirs.astype(bool))


def test_grid_order_matches_original_x_fastest():
    original = load_reference_roi()
    a = grid_centers(np.array([1., 2, 3]), [1, 2, 3], 8)
    b = original.generate_partition_centers(np.array([100., 200, 300]), [100, 200, 300], 8) / 100
    assert np.allclose(a, b)


def test_rectangular_geometry_not_cone_and_uses_roll():
    cfg = {'visibility_mode': 'rectangular_center_frustum', 'horizontal_fov_deg': 120,
           'vertical_fov_deg': 40, 'near_m': .01, 'far_m': 5}
    scene = Scene(np.array([[1, 0, .8], [1, .8, 0], [-1, 0, 0], [.001, 0, 0], [6, 0, 0]]),
                  ['a', 'b', 'c', 'd', 'e'], cfg)
    assert np.array_equal(scene.mask([0, 0, 0], ypr_to_quat([0, 0, 0])), [False, True, False, False, False])
    assert np.array_equal(scene.mask([0, 0, 0], ypr_to_quat([0, 0, 90])), [True, False, False, False, False])


def test_geometry_is_required_not_fabricated():
    with pytest.raises(ValueError, match='Confirm'):
        load_scene({'geometry_confirmed': False})
    with pytest.raises(ValueError, match='no scene'):
        load_scene({'geometry_confirmed': True})


def test_binary_metrics_empty_and_nonempty():
    m = Metrics()
    pred = np.array([[0, 0], [1, 0], [1, 1]], bool)
    true = np.array([[0, 0], [1, 0], [1, 0]], bool)
    counts, iou = m.add(pred, true, [0, 0, 0], [0, 0, 0])
    result = m.result()
    assert np.allclose(iou, [1, 1, .5])
    assert result['both_empty_count'] == 1
    assert result['iou_mean_nonempty_union'] == pytest.approx(.75)
    assert result['exact_match'] == pytest.approx(2 / 3)
    assert result['tp'] == 2 and result['fp'] == 1


def test_packed_masks_roundtrip_without_padding_cells():
    rng = np.random.default_rng(2)
    masks = rng.random((3, 5, 513)) > .5
    packed = np.packbits(masks, axis=-1, bitorder='little')
    restored = np.unpackbits(packed, axis=-1, count=513, bitorder='little').astype(bool)
    assert np.array_equal(masks, restored)
