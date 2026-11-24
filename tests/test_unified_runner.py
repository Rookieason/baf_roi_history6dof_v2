"""The shared command must not silently omit methods or compare unlike samples."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest

from roi_experiments import runner
from roi_experiments.methods import METHODS, resolve_methods


def config():
    path = Path(__file__).resolve().parents[1] / 'configs/default.json'
    return json.loads(path.read_text())


def report(method='baf'):
    pooled = {'n_origins': 3, 'iou_mean': .5, 'f1_micro': .6, 'coverage_micro': .7,
              'precision_micro': .6, 'selected_mean': 8., 'required_mean': 7.,
              'fp_mean': 3., 'fn_mean': 2., 'translation_mean_m': .1, 'rotation_mean_deg': 2.}
    return {'method': method, 'protocol_id': 'same-protocol', 'origin_keys_sha256': 'same-origins',
            'targets_sha256': 'same-targets', 'rf_cohort_sha256': 'same-rf-cohort', 'delivery_delay_s': 0.,
            'scene_id': 'same-scene', 'split': 'test', 'horizons_ms': [50],
            'summaries': {'50ms_margin0': {'all': {'pooled': pooled}}}}


def test_summary_explains_identical_baseline_retained_adapter(tmp_path):
    baseline = report()
    adapter = report('heatmap')
    adapter['checkpoint_selection'] = {'selected_epoch': 0, 'status': 'baseline_retained'}
    adapter['correction_applied_fraction'] = 0.
    result = runner.write_summary({'baf_roi': baseline, 'baf_roi_heatmap': adapter}, tmp_path)
    row = result['rows'][1]
    assert row['selected_epoch'] == 0 and row['selection_status'] == 'baseline_retained'
    assert row['correction_applied_fraction'] == 0. and row['delta_iou_mean_vs_baf'] == 0.


def test_all_selection_is_explicit_and_no_unknown_is_silently_dropped():
    assert resolve_methods() == list(METHODS)
    assert resolve_methods(['all']) == list(METHODS)
    for value in (['all', 'baf_roi'], ['baf_roi', 'baf_roi'], ['unknown']):
        with pytest.raises(ValueError):
            resolve_methods(value)


def test_generic_heatmap_run_fails_before_any_output_without_rf(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config()))
    args = runner.parser().parse_args(['--config', str(path), '--output', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match='require --rf-manifest'):
        runner.run(args)
    assert not (tmp_path / 'out').exists()


def test_pose_only_dry_run_does_not_require_recorded_rf_inputs(tmp_path):
    cfg = config()
    cfg['recorded_heatmaps'] = {'epochs': 2}
    path = tmp_path / 'config.json'; path.write_text(json.dumps(cfg))
    args = runner.parser().parse_args(['--config', str(path), '--output', str(tmp_path / 'out'),
                                      '--method', 'motion_transformer_roi', '--dry-run'])
    plan = runner.run(args)
    assert plan['methods'] == ['motion_transformer_roi']
    assert not plan['baf_dependency'] and not plan['recorded_heatmap_import']
    assert not plan['paired_evaluation'] and not (tmp_path / 'out').exists()


@pytest.mark.parametrize('field', ['protocol_id', 'origin_keys_sha256', 'targets_sha256',
                                  'rf_cohort_sha256', 'delivery_delay_s', 'scene_id', 'split', 'horizons_ms'])
def test_comparison_rejects_equal_counts_with_different_identity(field):
    a, b = report(), report('other')
    b[field] = [100] if field == 'horizons_ms' else 'different'
    with pytest.raises(ValueError, match=field):
        runner.assert_matched_reports({'baf_roi': a, 'baf_roi_heatmap': b})


def test_summary_uses_pooled_mean_iou_and_has_no_fabricated_ssim(tmp_path):
    a, b = report(), report('rf')
    b['summaries']['50ms_margin0']['all']['pooled']['iou_mean'] = .55
    b['summaries']['50ms_margin0']['all']['recording_macro'] = {'iou_mean': .99}
    result = runner.write_summary({'baf_roi': a, 'baf_roi_heatmap': b}, tmp_path)
    assert result['rows'][1]['iou_mean'] == .55
    assert result['rows'][1]['delta_iou_mean_vs_baf'] == pytest.approx(.05)
    assert result['quality']['ssim'] is None and result['quality']['status'] == 'not_evaluated'


def _fake_cache(tmp_path, monkeypatch):
    import baf_roi.data as data
    cfg = config(); cfg['training']['batch_size'] = 64
    path = tmp_path / 'cache'; path.mkdir()
    manifest = {'config': cfg, 'cache_id': 'cache', 'split': {'train': [], 'valid': [], 'test': []}}
    (path / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(data, 'Cache', lambda p: SimpleNamespace(manifest=manifest, stats={'mean': [0.], 'std': [1.]}))
    monkeypatch.setattr(runner, 'resolve_device', lambda value: 'cpu')
    return path


def _trainer(calls, method):
    def train(source, output, *args, **kwargs):
        calls.append(('train', method, str(source)))
        out = Path(output); out.mkdir(parents=True)
        path = out / (method + '.pt'); path.write_text(method)
        return path
    return train


def test_motion_transformer_alone_needs_no_baf_or_rf(tmp_path, monkeypatch):
    import motion_transformer_roi.train as mt
    import baf_roi.train as baf
    cache = _fake_cache(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(mt, 'train', _trainer(calls, 'motion_transformer_roi'))
    monkeypatch.setattr(baf, 'train', lambda *a, **k: pytest.fail('MT-only must not train BAF'))
    def evaluate_pose(cache_path, checkpoint, output, **kwargs):
        calls.append(('evaluate', kwargs['method'], str(cache_path)))
        return report('mt')
    monkeypatch.setitem(sys.modules, 'roi_experiments.evaluation', SimpleNamespace(
        evaluate=lambda *a, **k: pytest.fail('HMD-only must not require an RF prepared manifest'),
        evaluate_pose=evaluate_pose))
    out = tmp_path / 'out'
    args = runner.parser().parse_args(['--cache', str(cache), '--output', str(out),
                                      '--method', 'motion_transformer_roi', '--margins', '0'])
    result = runner.run(args)
    saved = json.loads((out / 'experiment.json').read_text())
    assert result['methods'] == ['motion_transformer_roi']
    assert list(saved['checkpoints']) == ['motion_transformer_roi'] and saved['prepared'] is None
    assert saved['complete'] and len(calls) == 2


def test_default_run_trains_four_methods_once_and_evaluates_one_prepared(tmp_path, monkeypatch):
    import baf_roi.train as baf
    import motion_transformer_roi.train as mt
    import baf_roi_heatmap.train as heat
    import baf_roi_heatmap.prepare as prepare
    cache = _fake_cache(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(baf, 'train', _trainer(calls, 'baf_roi'))
    monkeypatch.setattr(mt, 'train', _trainer(calls, 'motion_transformer_roi'))
    monkeypatch.setattr(heat, 'train', _trainer(calls, 'baf_roi_heatmap'))
    monkeypatch.setitem(sys.modules, 'baf_roi_heatmap_alpha.train', SimpleNamespace(
        train=_trainer(calls, 'baf_roi_heatmap_alpha')))
    monkeypatch.setattr(runner, '_check_checkpoint', lambda *args: None)
    monkeypatch.setattr(prepare, 'prepare', lambda *args: calls.append(('prepare', str(args[0]), str(args[3]))))
    def evaluate(prepared, output, **kwargs):
        method = Path(output).name
        calls.append(('evaluate', method, str(prepared)))
        if method == 'baf_roi_heatmap_alpha': assert 'alpha_checkpoint' in kwargs
        if method == 'baf_roi_heatmap': assert 'checkpoint' in kwargs
        if method == 'motion_transformer_roi': assert 'motion_transformer_checkpoint' in kwargs
        return report(method)
    monkeypatch.setitem(sys.modules, 'roi_experiments.evaluation', SimpleNamespace(
        evaluate=evaluate, evaluate_pose=lambda *a, **k: pytest.fail('All four must share the prepared evaluator')))
    out = tmp_path / 'out'
    (tmp_path / 'rf.json').write_text('{}')
    args = runner.parser().parse_args(['--cache', str(cache), '--rf-manifest', str(tmp_path / 'rf.json'),
                                      '--output', str(out), '--smoke-epochs', '1', '--margins', '0'])
    result = runner.run(args)
    trained = [row[1] for row in calls if row[0] == 'train']
    evaluated = [row for row in calls if row[0] == 'evaluate']
    assert trained == list(METHODS)
    assert len([row for row in calls if row[0] == 'prepare']) == 1
    assert [row[1] for row in evaluated] == list(METHODS)
    assert len({row[2] for row in evaluated}) == 1
    assert result['smoke_test'] and (out / 'summary.csv').is_file()


def test_existing_prepared_horizons_cannot_be_silently_changed(tmp_path):
    cache = tmp_path / 'cache'; cache.mkdir()
    (cache / 'manifest.json').write_text(json.dumps({'config': config()}))
    prepared = tmp_path / 'prepared'; prepared.mkdir()
    (prepared / 'manifest.json').write_text(json.dumps({'pose_cache': str(cache), 'horizons_ms': [50]}))
    args = runner.parser().parse_args(['--prepared', str(prepared), '--output', str(tmp_path / 'out'),
                                      '--horizons-ms', '100', '--dry-run'])
    with pytest.raises(ValueError, match='horizons-ms differs'):
        runner.run(args)


def test_reused_legacy_checkpoint_is_not_reported_as_default_motion_moe(tmp_path, monkeypatch):
    import torch
    import baf_roi_heatmap.prepare as prepare
    cache = _fake_cache(tmp_path, monkeypatch)
    baseline = tmp_path/'baf.pt'; baseline.write_text('baseline')
    adapter = tmp_path/'legacy.pt'
    torch.save({'model_config': {'method': 'residual', 'modality': 'rf'},
                'objective': 'roi_iou', 'aggregation': 'origin'}, adapter)
    rf = tmp_path/'rf.json'; rf.write_text('{}')
    monkeypatch.setattr(runner, '_check_checkpoint', lambda *args: None)
    monkeypatch.setattr(prepare, 'prepare', lambda *args: None)
    monkeypatch.setitem(sys.modules, 'roi_experiments.evaluation', SimpleNamespace(
        evaluate=lambda *a, **k: report('legacy'), evaluate_pose=lambda *a, **k: report()))
    out = tmp_path/'out'
    args = runner.parser().parse_args(['--cache', str(cache), '--rf-manifest', str(rf),
                                      '--method', 'baf_roi_heatmap', '--baseline-checkpoint', str(baseline),
                                      '--heatmap-checkpoint', str(adapter), '--output', str(out)])
    runner.run(args)
    saved = json.loads((out/'experiment.json').read_text())
    assert saved['heatmap_adapter'] == 'residual'
    assert saved['heatmap_objective'] == 'roi_iou' and saved['heatmap_aggregation'] == 'origin'


@pytest.mark.parametrize('raises', [False, True])
def test_recorded_guard_covers_work_and_stops_after_success_or_error(tmp_path, monkeypatch, raises):
    import baf_roi_heatmap.recorded as recorded
    checked = threading.Event()
    calls, monitors = [], []
    original_thread = threading.Thread
    def make_thread(*args, **kwargs):
        thread = original_thread(*args, **kwargs)
        monitors.append(thread)
        return thread
    def check(path, initial=False):
        calls.append((str(path), initial))
        if not initial:
            checked.set()
    monkeypatch.setattr(recorded, 'check_resources', check)
    monkeypatch.setattr(runner.threading, 'Thread', make_thread)
    monkeypatch.setattr(runner, 'RESOURCE_CHECK_SECONDS', .001)
    class TrainingFailed(Exception):
        pass
    try:
        with runner.resource_guard({'recorded_heatmaps': {}}, tmp_path):
            assert checked.wait(1.), 'Recorded resources must continue being checked during training.'
            assert monitors[0].is_alive()
            if raises:
                raise TrainingFailed()
    except TrainingFailed:
        assert raises
    assert calls[0] == (str(tmp_path), True)
    assert any(not initial for _, initial in calls[1:])
    assert monitors[0].daemon and not monitors[0].is_alive()


def test_initial_recorded_resource_failure_prevents_work(tmp_path, monkeypatch):
    import baf_roi_heatmap.recorded as recorded
    def check(path, initial=False):
        assert initial
        raise RuntimeError('Resource reserve reached: disk')
    monkeypatch.setattr(recorded, 'check_resources', check)
    with pytest.raises(RuntimeError, match='reserve reached'):
        with runner.resource_guard({'recorded_heatmaps': {}}, tmp_path):
            pytest.fail('Training must not start after the initial resource check failed.')


def test_recorded_monitor_retains_five_second_check_and_exit_code(tmp_path, monkeypatch):
    import baf_roi_heatmap.recorded as recorded
    intervals, exits = [], []
    class Stop:
        def wait(self, interval):
            intervals.append(interval)
            return False
    def check(path, initial=False):
        assert not initial
        raise RuntimeError('Resource reserve reached: RAM/RSS/disk')
    monkeypatch.setattr(recorded, 'check_resources', check)
    monkeypatch.setattr(runner.os, '_exit', lambda code: exits.append(code))
    runner._monitor_resources(Stop(), tmp_path)
    assert intervals == [5] and exits == [75]


def test_generic_dataset_and_recorded_dry_run_do_not_start_resource_guard(tmp_path, monkeypatch):
    import baf_roi_heatmap.recorded as recorded
    monkeypatch.setattr(recorded, 'check_resources', lambda *a, **k: pytest.fail('Unexpected resource check'))
    with runner.resource_guard(config(), tmp_path):
        pass
    cfg = config(); cfg['recorded_heatmaps'] = {'epochs': 2}
    path = tmp_path / 'config.json'; path.write_text(json.dumps(cfg))
    args = runner.parser().parse_args(['--config', str(path), '--output', str(tmp_path / 'out'), '--dry-run'])
    assert runner.run(args)['recorded_heatmap_import']
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('source', ['cache', 'prepared'])
def test_recorded_profile_reused_inputs_keep_guard_through_training_and_evaluation(tmp_path, monkeypatch, source):
    import baf_roi_heatmap.recorded as recorded
    cfg = config(); cfg['recorded_heatmaps'] = {'epochs': 2}
    cache = tmp_path / 'cache'; cache.mkdir()
    (cache / 'manifest.json').write_text(json.dumps({'config': cfg}))
    path = cache
    if source == 'prepared':
        path = tmp_path / 'prepared'; path.mkdir()
        (path / 'manifest.json').write_text(json.dumps({'pose_cache': str(cache), 'horizons_ms': [50]}))
    checks, stages = [], []
    monkeypatch.setattr(recorded, 'check_resources', lambda output, initial=False: checks.append(initial))
    monkeypatch.setattr(runner, 'resolve_device', lambda device: 'cpu')
    def is_monitoring():
        return any(t.name == 'roi-recorded-resource-monitor' and t.is_alive() for t in threading.enumerate())
    def train(*args):
        assert checks[0] is True and is_monitoring()
        stages.append('train')
        return {'smoke_test': False, 'complete': False}
    def evaluate(*args):
        assert is_monitoring()
        stages.append('evaluate')
    monkeypatch.setattr(runner, '_train_selected', train)
    monkeypatch.setattr(runner, '_evaluate_selected', evaluate)
    args = runner.parser().parse_args([f'--{source}', str(path), '--method', 'motion_transformer_roi',
                                      '--output', str(tmp_path / 'out')])
    runner.run(args)
    assert stages == ['train', 'evaluate'] and not is_monitoring()
