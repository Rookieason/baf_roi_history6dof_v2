"""Regression tests for the newly supplied ORIGINAL data preparation source."""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from baf_roi.data import Series, Cache, prepare, OriginDataset
from baf_roi.models import build_models, load_original
from baf_roi.pose import model_inputs
from baf_roi.runtime import Predictor
from baf_roi.supervision import (author_module, score_resampled_index, centered_pair,
                                 semantic_supervisor_rotations, protocol_metadata,
                                 label_code, numeric_record)
from scripts.smoke_test import make_fixture

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def same_dict(a, b):
    assert (a is None) == (b is None)
    if a is None:
        return
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], str):
            assert a[key] == b[key]
        else:
            np.testing.assert_allclose(a[key], b[key], rtol=0, atol=0, equal_nan=True,
                                       err_msg=f'Original field mismatch: {key}')


def synthetic(n=151, seed=9):
    rng = np.random.default_rng(seed)
    timer = np.arange(n, dtype=float) / 30
    x = np.zeros((n, 6))
    x[:, :3] = np.cumsum(rng.normal(0, .004, (n, 3)), axis=0)
    rate = rng.choice([-240., -30., 0., 0., 30., 240.], n)
    x[:, 3] = (np.cumsum(rate / 30) + 180) % 360 - 180
    x[:, 4:] = rng.normal(0, 2, (n, 2))
    return timer, x


@pytest.mark.parametrize('seed', [1, 13, 88])
def test_all_original_fields_match_author_function_at_every_row(seed):
    timer, x = synthetic(seed=seed)
    rot = x[:, [4, 3, 5]]  # independent, explicit semantic yaw mapping
    m = author_module()
    for i in range(len(timer)):
        reference = m.score_one_index(timer, x[:, :3], rot, np.deg2rad(rot), i)
        same_dict(score_resampled_index(timer, x, i), reference)


def test_step3_and_step5_label_scores_match_on_same_flat_input():
    m = load_module('_test_step3', ROOT / 'vendor/data_preparation/csv_label_generator.py')
    timer, x = synthetic(91)
    df = pd.DataFrame(np.column_stack([timer, x[:, :3], x[:, [4, 3, 5]]]),
                      columns=['Timer', 'HeadX', 'HeadY', 'HeadZ', 'HeadRX', 'HeadRY', 'HeadRZ'])
    labeled = m.label_dataframe(df)
    for i in range(len(df)):
        ours = score_resampled_index(timer, x, i)
        assert labeled.at[i, 'Label'] == ours['Label']
        for field in m.OUTPUT_COLUMNS:
            if field != 'Label':
                assert labeled.at[i, field] == pytest.approx(ours[field], rel=1e-12, abs=1e-12)


def test_centered_label_can_change_while_causal_model_history_is_identical():
    timer = np.arange(121, dtype=float) / 30
    still = np.zeros((121, 6)); still[:, 2] = 1.5
    moving = still.copy()
    future = np.maximum(0, timer - timer[45])
    moving[:, 0] += .8 * future
    moving[:, 3] += 220 * future
    a, b = Series(timer, still), Series(timer, moving)
    assert np.array_equal(a.history(45), b.history(45))
    assert centered_pair(a, 45)[0]['Label'] == 'Focused'
    assert centered_pair(b, 45)[0]['Label'] == 'Explorer'
    assert centered_pair(b, 45)[0]['supervision_is_centered_window'] == 1


@pytest.mark.parametrize('anchor', [0, 20, 73, 149])
def test_centered_pair_matches_unmodified_author_on_same_reanchored_grid(anchor):
    # Nonuniform native observations: keep v1's actual-origin protocol.
    timer = np.arange(150, dtype=float) / 75 + np.arange(150) * .00007
    x = np.zeros((150, 6)); x[:, 0] = .27 * timer
    x[:, 3] = 150 * timer; x[:, 4] = 7 * np.sin(timer)
    s = Series(timer, x)
    off = np.arange(-16, 18)
    query = timer[anchor] + off / 30
    mask = (query >= timer[0]) & (query <= timer[-1])
    query, off = query[mask], off[mask]
    p = s.at(query)
    r = p[:, [4, 3, 5]]
    ref = []
    for k in (0, 1):
        ids = np.flatnonzero(off == k)
        ref.append(author_module().score_one_index(query, p[:, :3], r, np.deg2rad(r), int(ids[0]))
                   if len(ids) else None)
    got = centered_pair(s, anchor)
    for a, b in zip(got, ref):
        same_dict(a, b)


def test_exact_zero_rate_matches_author_not_v1_remove_zeros():
    assert author_module().zero_crossing_count(np.array([-1., 0., -1.])) == 2
    timer = np.arange(7) / 30
    x = np.zeros((7, 6)); x[:, 3] = np.r_[0, np.cumsum([-1., 0., -1., 0., -1., 0.])]
    result = score_resampled_index(timer, x, 3)
    assert result['yaw_zero_cross_per_s'] == pytest.approx(5 / (6 / 30))
    v1 = load_module('_test_v1', ROOT / 'reference/v1_supervision.py')
    old = np.zeros((30, 6)); old[:, 3] = np.r_[0, np.cumsum(np.resize([-1., 0.], 29))]
    assert v1.motion_features(old)[2] == 0


def test_partial_centered_windows_use_author_minimum_five_not_full_thirty():
    timer, x = synthetic(5)
    assert score_resampled_index(timer[:4], x[:4], 2) is None
    result = score_resampled_index(timer, x, 2)
    assert result is not None and result['score_valid'] == 1
    assert result['supervision_window_duration_sec'] == 1
    assert timer[-1] - timer[0] < 1  # nominal duration is NOT actual coverage


def test_scoring_yaw_axis_is_not_geometric_xyz_middle_channel():
    t = np.arange(31) / 30
    yaw = np.zeros((31, 6)); yaw[:, 0] = .3 * t; yaw[:, 3] = 200 * t
    pitch = yaw.copy(); pitch[:, 3] = 0; pitch[:, 4] = 200 * t
    a, b = score_resampled_index(t, yaw, 15), score_resampled_index(t, pitch, 15)
    assert np.rad2deg(a['head_rate_yaw_q90']) == pytest.approx(200)
    assert b['head_rate_yaw_q90'] == 0
    assert a['Label'] == 'Explorer' and b['Label'] == 'Focused'
    # Neural inputs retain v1's physical transform, not scoring-only column order.
    conf, _ = model_inputs(yaw[:30], {'mean': [0, 0, 0], 'std': [1, 1, 1]})
    assert np.allclose(conf[:, -1], yaw[:30, 3])
    assert np.all(conf[:, -2] == 0)


def test_metadata_units_and_wraparound_debug_are_not_silently_corrected():
    timer = np.arange(7) / 30
    x = np.zeros((7, 6)); x[:, 3] = [173, 175, 177, 179, -179, -177, -175]
    out = score_resampled_index(timer, x, 3)
    assert out['head_rate_yaw_q90'] == pytest.approx(np.deg2rad(60))
    # Source intentionally uses raw np.diff for debug columns; keep this behavior.
    assert out['delta_ry_deg_s'] > 1000
    assert protocol_metadata()['head_rate_yaw_q90_unit'].startswith('rad/s')


def test_time_weighted_translation_and_strict_high_rate_threshold():
    timer = np.array([0, .01, .03, .06, .1])
    x = np.zeros((5, 6)); x[:, 0] = [0, .01, .011, .012, .02]
    assert score_resampled_index(timer, x, 2)['speed_mean'] == pytest.approx(.2)
    for deg_rate, expected in ((59., 0.), (61., 1.)):
        x[:, 3] = timer * deg_rate
        assert score_resampled_index(timer, x, 2)['head_rate_yaw_hi_prop'] == expected


def test_unlabeled_defaults_are_nan_and_excluded():
    m = author_module()
    assert np.isnan(m.label_to_binary_target('Unlabeled'))
    assert m.label_to_valid_for_confidence('Unlabeled') == 0
    assert label_code(None) == -1
    defaults = numeric_record(None)
    assert defaults['valid_for_confidence'] == defaults['score_valid'] == 0
    assert np.isnan(defaults['binary_target'])


def test_inference_does_not_call_oracle_supervisor(monkeypatch):
    torch.set_num_threads(2)
    predictor = Predictor(build_models(), {'mean': [0, 0, 0], 'std': [1, 1, 1]})
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline supervisor was called during inference')
    monkeypatch.setattr(author_module(), 'score_one_index', forbidden)
    t = np.arange(40) / 30
    out = predictor.predict_observations(t, np.zeros((40, 3)), np.zeros((40, 3)))
    assert out['positions_m'].shape == (1, 5, 3)
    assert out['rollout_positions_m'].shape == (1, 9, 3)


def test_source_confidence_reader_drops_nan_labels_before_windowing(tmp_path):
    # Document the upstream behavior we deliberately do NOT propagate into v2.
    m = load_original('confidence_reader_test', 'confidence/utility.py')
    t = np.arange(45) / 30
    df = pd.DataFrame({'Timer': t, **{c: np.zeros(45) for c in m.POSE_COLS},
                       'binary_target': np.ones(45), 'valid_for_confidence': np.ones(45)})
    df['HeadX'] = t
    df.loc[20, ['binary_target', 'valid_for_confidence']] = [np.nan, 0]
    path = tmp_path / 'conf.csv'; df.to_csv(path, index=False)
    samples = m.ConfidenceDatasetBuilder().build_from_file(str(path))
    sample = next(s for s in samples if abs(s.time_t - t[35]) < 1e-12)
    assert len(sample.x_window) == 30
    assert np.max(np.diff(sample.x_window[:, 0])) > 1.5 / 30
    intact = Series(t, np.column_stack((t, np.zeros((45, 5))))).history(35)
    assert np.allclose(np.diff(intact[:, 0]), 1 / 30)
    assert intact[-1, 0] - intact[0, 0] == pytest.approx(29 / 30)


def test_cache_stores_centered_current_and_next_metadata_and_keeps_inputs(tmp_path):
    cfg = make_fixture(tmp_path)
    manifest = prepare(cfg, tmp_path / 'cache')
    cache = Cache(tmp_path / 'cache')
    assert manifest['format_version'] == 2
    assert manifest['supervision']['id'] == 'author_centered_supervision_v2'
    arrays = cache.arrays(0)
    with np.load(cache.path / cache.records[0]['key'] / 'supervision_metadata.npz') as meta:
        for oi, anchor in enumerate(arrays['anchors']):
            cur, nxt = centered_pair(arrays['series'], int(anchor))
            assert arrays['labels'][oi] == label_code(cur)
            assert meta['current_supervision_window_center_time'][oi] == pytest.approx(arrays['t'][anchor])
            if nxt is not None:
                assert meta['next_supervision_window_center_time'][oi] == pytest.approx(arrays['t'][anchor] + 1 / 30)
            if arrays['next_labels'][oi] != -2:
                assert np.array_equal(arrays['targets'][oi], arrays['series'].future_history(int(anchor))[-1])
    ds = OriginDataset(cache.path, 'train', 'confidence')
    ti, oi = ds.index[len(ds) // 2]
    a = ds.cache.arrays(int(ti))
    expected, _ = model_inputs(a['series'].history(int(a['anchors'][oi])), cache.stats)
    assert np.array_equal(ds[len(ds) // 2]['conf'], expected)
    assert len(a['pose']) == 720  # Unlabeled poses have NOT been removed.


def test_v1_cache_and_checkpoint_are_rejected(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'format_version': 1}))
    with pytest.raises(ValueError, match='retrain'):
        Cache(tmp_path)
    checkpoint = tmp_path / 'v1.pt'; torch.save({'format_version': 1}, checkpoint)
    with pytest.raises(ValueError, match='retrain'):
        Predictor.load(checkpoint)
