"""Failures that would invalidate the scientific comparison, not only tensor shapes."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from baf_roi_heatmap.contracts import InputConfig, write_json
from baf_roi_heatmap.data import load_rf_manifest, HeatmapSeries
from baf_roi_heatmap.music import generate
from baf_roi_heatmap.evaluation import compare, select_policy
from .test_core import maps_file


def manifest_fixture(tmp_path):
    records, entries = [], []
    for index, split in enumerate(('train', 'valid', 'test')):
        key = str(index)
        path = tmp_path/(key+'.npz'); maps_file(path, shift=index)
        records.append({'key': key, 'split': split, 'session': key})
        entries.append({'key': key, 'file': str(path), 'clock_offset_s': 0., 'clock_uncertainty_s': 0.,
                        'source_clock_domain': 'unix_seconds', 'group_id': key})
    cache = SimpleNamespace(manifest={'cache_id': 'known'}, records=records)
    manifest = {'format_version': 1, 'hmd_cache_id': 'known', 'hmd_clock_domain': 'unix_seconds',
                'synchronization_confirmed': True, 'records': entries}
    return cache, manifest


@pytest.mark.parametrize('mutation', ['group', 'duplicate', 'missing', 'clock', 'cache', 'uncertainty', 'sync'])
def test_rf_manifest_rejects_invalid_protocol(tmp_path, mutation):
    cache, m = manifest_fixture(tmp_path)
    if mutation == 'group': m['records'][1]['group_id'] = '0'
    if mutation == 'duplicate': m['records'][1]['file'] = m['records'][0]['file']
    if mutation == 'missing': m['records'].pop()
    if mutation == 'clock': m['records'][0]['source_clock_domain'] = 'wrong'
    if mutation == 'cache': m['hmd_cache_id'] = 'wrong'
    if mutation == 'uncertainty': m['records'][0]['clock_uncertainty_s'] = .1
    if mutation == 'sync': m['synchronization_confirmed'] = False
    path = tmp_path/'rf.json'; write_json(path, m)
    with pytest.raises(ValueError): load_rf_manifest(path, cache, InputConfig())


def test_explicit_missing_rf_is_accepted(tmp_path):
    cache, m = manifest_fixture(tmp_path); m['records'][0]['file'] = None
    path = tmp_path/'rf.json'; write_json(path, m)
    result = load_rf_manifest(path, cache, InputConfig())
    assert result['records'][0]['file'] is None and result['grid_id']


def test_exploratory_axes_require_opt_in(tmp_path):
    p = tmp_path/'maps.npz'; maps_file(p)
    with np.load(p) as d: arrays = {k: d[k] for k in d.files}
    meta = json.loads(str(arrays['metadata_json'].item())); meta['physical_axes_verified'] = False
    arrays['metadata_json'] = np.array(json.dumps(meta)); np.savez(p, **arrays)
    with pytest.raises(ValueError): HeatmapSeries(p)
    assert not HeatmapSeries(p, allow_exploratory=True).metadata['physical_axes_verified']


def test_unverified_zero_covariance_not_a_valid_peak(tmp_path):
    ts = np.arange(10)*.01
    np.savez(tmp_path/'raw.npz', csi=np.ones((10, 1, 3, 4), complex), timestamps_s=ts,
             available_at_s=ts, frequencies_hz=5e9+np.arange(4)*1e6)
    cfg = {'window_packets': 4, 'stride_packets': 2, 'rx_window': 2, 'frequency_points': 2,
           'frequency_hop': 1, 'signal_dim': 1, 'theta_deg': [-30, 0, 30], 'tau_s': [0, 1e-9],
           'carrier_hz': 5e9, 'antenna_spacing_m': .02, 'max_raw_gap_s': .02,
           'projection': 'sin', 'preprocess': 'window_mean', 'delivery_delay_s': .1,
           'clock_domain': 'unix_seconds'}
    write_json(tmp_path/'cfg.json', cfg)
    report = generate(tmp_path/'raw.npz', tmp_path/'cfg.json', tmp_path/'maps.npz')
    assert not report['physical_axes_verified'] and report['valid'] == 0
    with np.load(tmp_path/'maps.npz') as d:
        assert not d['valid'].any()
        np.testing.assert_allclose(d['available_at_s']-d['support_end_s'], .1)
        np.testing.assert_allclose(d['support_end_s']-d['support_start_s'], .03)
    with pytest.raises(ValueError): generate(tmp_path/'raw.npz', tmp_path/'cfg.json', tmp_path/'no_extension')


def test_future_raw_csi_cannot_change_already_generated_maps(tmp_path):
    rng = np.random.default_rng(91); ts = np.arange(30)*.01
    csi = rng.normal(size=(30, 1, 3, 4))+1j*rng.normal(size=(30, 1, 3, 4))
    cfg = {'window_packets': 4, 'stride_packets': 2, 'rx_window': 2, 'frequency_points': 2,
           'frequency_hop': 1, 'signal_dim': 1, 'theta_deg': [-30, 0, 30], 'tau_s': [0, 1e-9],
           'carrier_hz': 5e9, 'antenna_spacing_m': .02, 'max_raw_gap_s': .02,
           'projection': 'sin', 'delivery_delay_s': .01, 'clock_domain': 'unix_seconds'}
    write_json(tmp_path/'cfg.json', cfg)
    for suffix in ('a', 'b'):
        if suffix == 'b': csi[ts > .15] *= 500
        np.savez(tmp_path/(suffix+'.npz'), csi=csi, timestamps_s=ts, available_at_s=ts, frequencies_hz=5e9+np.arange(4)*1e6)
        generate(tmp_path/(suffix+'.npz'), tmp_path/'cfg.json', tmp_path/(suffix+'_maps.npz'))
    with np.load(tmp_path/'a_maps.npz') as a, np.load(tmp_path/'b_maps.npz') as b:
        before = a['support_end_s'] <= .15
        np.testing.assert_array_equal(a['maps'][before], b['maps'][before])


def test_policy_cannot_be_tuned_on_test(tmp_path):
    write_json(tmp_path/'report.json', {'split': 'test'})
    with pytest.raises(ValueError, match='validation'): select_policy(tmp_path, tmp_path/'policy.json')


def test_comparison_rejects_unmatched_experiments(tmp_path):
    for name in ('a', 'b'):
        write_json(tmp_path/name/'report.json', {'experiment_id': name})
    with pytest.raises(ValueError, match='experiment_id'): compare(tmp_path/'a', tmp_path/'b', tmp_path/'delta.json')


def test_runtime_rolls_back_observed_state_on_error():
    from baf_roi_heatmap.runtime import HeatmapPredictor
    predictor = HeatmapPredictor.__new__(HeatmapPredictor)
    predictor.base = SimpleNamespace(observed_state={'stream': (1., .3)})
    def fail(*args):
        predictor.base.observed_state['stream'] = (2., .8)
        raise ValueError('injected adapter failure')
    predictor._predict = fail
    with pytest.raises(ValueError): predictor.predict_observations([], [], [])
    assert predictor.base.observed_state == {'stream': (1., .3)}


def test_scene_config_paths_are_relative_to_config_file(tmp_path):
    from baf_roi_heatmap.contracts import read_scene_config
    write_json(tmp_path/'scene.json', {'scene': {'partition_centers_file': 'centers.npz'}})
    result = read_scene_config(tmp_path/'scene.json')
    assert result['partition_centers_file'] == str(tmp_path/'centers.npz')


def test_manifest_template_requires_human_clock_review(tmp_path, monkeypatch):
    import baf_roi_heatmap.manifest as module
    cache = SimpleNamespace(manifest={'cache_id': 'known'}, records=[
        {'key': 'trajectory_00000', 'source': 'head.csv', 'epoch_start_s': 100., 'session': 'session1'}])
    monkeypatch.setattr(module, 'Cache', lambda path: cache)
    path = tmp_path/'rf.json'
    module.manifest_template('unused', tmp_path/'maps', path)
    saved = json.loads(path.read_text())
    assert not saved['synchronization_confirmed']
    assert saved['records'][0]['clock_uncertainty_s'] > .01
    assert saved['records'][0]['file'] is None
    with pytest.raises(ValueError): module.manifest_template('unused', tmp_path, path)


def test_reprocessed_same_raw_recording_cannot_cross_splits(tmp_path):
    cache, manifest = manifest_fixture(tmp_path)
    for entry in manifest['records']:
        path = Path(entry['file'])
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        metadata = json.loads(str(arrays['metadata_json'].item()))
        metadata['raw_sha256'] = 'same-raw-acquisition-with-different-processing'
        arrays['metadata_json'] = np.array(json.dumps(metadata))
        np.savez(path, **arrays)
    path = tmp_path/'rf.json'; write_json(path, manifest)
    with pytest.raises(ValueError, match='same raw CSI'):
        load_rf_manifest(path, cache, InputConfig())


def test_motion_transformer_cli_is_exclusive():
    from baf_roi_heatmap.__main__ import parser
    args = parser().parse_args(['evaluate', '--prepared', 'p', '--output', 'o', '--motion-transformer', 'm.pt'])
    assert args.motion_transformer == 'm.pt' and args.checkpoint is None
    with pytest.raises(SystemExit):
        parser().parse_args(['evaluate', '--prepared', 'p', '--output', 'o', '--motion-transformer', 'm.pt', '--checkpoint', 'a.pt'])
