"""Regression cases where segmentation or irregular timing changed comparisons."""
import csv
import importlib
import json
from types import SimpleNamespace

import numpy as np
import torch

from baf_roi.data import Series
from baf_roi.roi import Scene
from baf_roi.runtime import Predictor, observed_weights
from baf_roi_heatmap.data import ForecastDataset, fit_normalizer
from baf_roi_heatmap.prepare import observed_weights as prepare_observed_weights
from baf_roi_heatmap.train import validation_iou


def irregular_series():
    # Every raw gap is valid, but snapping 30-Hz decisions skips t=1.020.
    times = np.r_[np.arange(31) / 30, 1.020, 1.118, np.arange(35, 70) / 30]
    poses = np.zeros((len(times), 6))
    poses[:, 0] = times
    return Series(times, poses, .1)


def test_observed_ema_is_batch_invariant_and_resets_at_decision_gaps():
    series = irregular_series()
    times = series.t[series.origin_indices()]
    assert np.diff(series.t).max() < .1 < np.diff(times).max()
    confidence = (times > 1.05).astype(float)
    whole, state = observed_weights(confidence, times, .6, .1)
    prefix, intermediate = observed_weights(confidence[:2], times[:2], .6, .1)
    suffix, final = observed_weights(confidence[2:], times[2:], .6, .1, intermediate)
    np.testing.assert_array_equal(whole, np.r_[prefix, suffix])
    np.testing.assert_array_equal(whole[:3], [0., 0., 1.])
    assert final == state
    assert prepare_observed_weights is observed_weights


def test_legacy_and_prepared_baf_share_gap_reset_and_predictions(tmp_path, monkeypatch):
    legacy = importlib.import_module('baf_roi.evaluate')
    preparation = importlib.import_module('baf_roi_heatmap.prepare')
    series = irregular_series()
    records = [dict(key=f'trajectory_{i}', session=split, split=split,
                    source_sha256=f'source-{i}', epoch_start_s=1000. * i)
               for i, split in enumerate(('train', 'valid', 'test'))]
    cache = SimpleNamespace(
        manifest={'cache_id': 'shared', 'config': {'evaluation': {'horizons_ms': [50.]},
                                                  'scene': {}},
                  'supervision': {'id': 'fixture'},
                  'split': {r['split']: [r['session']] for r in records}},
        records=records, stats={'mean': [0., 0., 0.], 'std': [1., 1., 1.]},
        arrays=lambda index: {'series': series, 'anchors': series.origin_indices()},
    )
    scene = Scene(np.array([[3., 0., 0.], [1., 1., 0.]]), ['a', 'b'],
                  {'visibility_mode': 'legacy_center_cone', 'fov_deg': 110.})

    class DeterministicPredictor(Predictor):
        def __init__(self):
            super().__init__({}, cache.stats, mu=.6)
            self.bundle = {'cache_id': 'shared'}

        @classmethod
        def load(cls, *args, **kwargs):
            return cls()

        def confidence(self, histories):
            return (histories[:, -1, 0] > 1.05).astype(np.float32)

        def _step(self, histories):
            focused = histories[:, -1, :3] + [0.01, 0., 0.]
            explorer = histories[:, -1, :3] + [0.05, 0., 0.]
            q = np.tile([1., 0., 0., 0.], (len(histories), 1))
            return self.confidence(histories), focused, q, explorer, q

    for module in (legacy, preparation):
        monkeypatch.setattr(module, 'Cache', lambda path: cache)
        monkeypatch.setattr(module, 'load_scene', lambda config: scene)
    monkeypatch.setattr(preparation, 'Predictor', DeterministicPredictor)
    monkeypatch.setattr(preparation, 'load_rf_manifest', lambda *args: {
        'records': [dict(key=r['key'], split=r['split'], file=None,
                         group_id=r['session'], scenario='example') for r in records]})
    checkpoint = tmp_path / 'base.pt'
    checkpoint.write_bytes(b'fixture checkpoint; fake model supplies deterministic predictions')
    prepared = tmp_path / 'prepared'
    preparation.prepare('cache', checkpoint, 'rf.json', prepared, horizons=[50.], batch_size=3)
    legacy.evaluate('cache', checkpoint, tmp_path / 'legacy', horizons_ms=[50.], batch_size=2,
                    predictor_class=DeterministicPredictor)
    with (tmp_path / 'legacy' / 'per_origin_metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with np.load(prepared / 'frozen' / 'trajectory_2.npz') as frozen:
        np.testing.assert_array_equal([float(row['alpha_t']) for row in rows], frozen['alpha'])
        np.testing.assert_allclose([float(row['origin_epoch_s']) for row in rows], frozen['origin_s'])
        predicted = []
        for path in sorted((tmp_path / 'legacy' / 'predictions').glob('*.npz')):
            with np.load(path) as batch:
                predicted.extend(batch['positions_m'])
        np.testing.assert_allclose(predicted, frozen['base_p'], rtol=0, atol=0)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    assert manifest['records'][2]['recording_id'] == 'test:source-2'


def segmented_dataset():
    dataset = ForecastDataset.__new__(ForecastDataset)
    dataset.records = [{'cache_index': index} for index in range(3)]
    dataset.cache = SimpleNamespace(records=[
        {'session': 's1', 'source_sha256': 'source-a'},
        {'session': 's1', 'source_sha256': 'source-a'},
        {'session': 's1', 'source_sha256': 'source-b'},
    ])
    return dataset


def test_recording_identity_joins_legacy_segments_but_keeps_sessions_distinct():
    dataset = segmented_dataset()
    assert dataset.recording_id(0) == dataset.recording_id(1) == 's1:source-a'
    assert dataset.recording_id(2) == 's1:source-b'
    dataset.cache.records[1]['session'] = 's2'
    assert dataset.recording_id(1) == 's2:source-a'
    dataset.records[0]['recording_id'] = 'saved:identity'
    assert dataset.recording_id(0) == 'saved:identity'


def test_validation_iou_weights_original_recordings_not_segments():
    dataset = segmented_dataset()
    dataset.scene = Scene(np.array([[1., 0., 0.]]), ['a'],
                          {'visibility_mode': 'legacy_center_cone', 'fov_deg': 110.})
    dataset.arrays = lambda index: {
        'base_p': np.array([[[0. if index < 2 else 2., 0., 0.]]]),
        'base_q': np.array([[[1., 0., 0., 0.]]]),
    }

    class UnchangedModel(torch.nn.Module):
        def forward(self, inputs):
            return {'p': torch.zeros(3, 1, 3), 'q': torch.tensor([1., 0., 0., 0.]).expand(3, 1, 4),
                    'applied': torch.zeros(3, 1, dtype=torch.bool)}

    loader = [{'inputs': {'hmd': torch.zeros(3, 30, 7)},
               'record_index': torch.arange(3), 'origin_index': torch.zeros(3, dtype=torch.long),
               'supervision': {'target_mask': torch.ones(3, 1, 1)}}]
    # A has two perfect segments; B has one missed segment. Recording mean is .5, not 2/3.
    assert validation_iou(UnchangedModel(), dataset, loader, 'cpu') == .5


def test_rf_normalizer_does_not_reweight_a_file_referenced_by_more_segments(tmp_path):
    entries = []
    for name, value in [('a', 1.), ('b', 100.)]:
        path = tmp_path / (name + '.npz')
        metadata = {'format_version': 1, 'unit': 'linear', 'representation': 'aoa_tof',
                    'physical_axes_verified': True, 'clock_domain': 'unix_seconds',
                    'availability_source': 'fixture'}
        np.savez(path, maps=np.full((4, 2, 2), value), valid=np.ones(4, bool),
                 support_start_s=np.arange(4) * .1, support_end_s=np.arange(4) * .1 + .05,
                 available_at_s=np.arange(4) * .1 + .06, theta_deg=[0., 1.], tau_s=[0., 1e-9],
                 metadata_json=np.array(json.dumps(metadata)))
        entries.append(dict(key=name, split='train', file=str(path),
                            clock_offset_s=0., clock_uncertainty_s=0.))
    original = fit_normalizer({'records': entries})
    repeated = fit_normalizer({'records': [entries[0], dict(entries[0], key='a-second-segment'), entries[1]]})
    assert repeated == original
    assert original['median'] == 10.
    assert original['sample_count'] == 32
