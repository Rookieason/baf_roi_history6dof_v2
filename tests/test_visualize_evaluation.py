"""Reporting must preserve matched cohorts and the evaluator's aggregation semantics."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from roi_experiments.evaluation import balanced_summary
from roi_experiments.visualize import load_evaluations, primary_margin


def make_evaluation(root, method='baf_roi'):
    directory = root / 'evaluations' / 'test' / method
    directory.mkdir(parents=True)
    rows = []
    for horizon in (50, 1000):
        for margin in (0., 5.):
            for i in range(3):
                # Recording a has twice the origins of b; micro F1 is not mean F1.
                tp, fn = ((1, 1) if i < 2 else (2, 0))
                rows.append(dict(origin_id=f'o{i}', record=f'segment{i}',
                                 recording_id='a' if i < 2 else 'b', group_id=f'g{i // 2}',
                                 scenario='walk' if i < 2 else 'sit', horizon_ms=horizon,
                                 margin_deg=margin, rf_available=1, tp=tp, fp=0, fn=fn, tn=3,
                                 iou=tp / (tp + fn), translation_error_m=.01 * i,
                                 rotation_error_deg=float(i), correction_applied=0))
    path = directory / 'per_origin.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(experiment_id='experiment', scene_id='scene', split='test',
                  horizons_ms=[50, 1000], margins_deg=[0., 5.], delivery_delay_s=0.,
                  protocol_id='protocol', origin_keys_sha256='origins', targets_sha256='targets',
                  rf_cohort_sha256='rf', rows_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  summaries={f'{h}ms_margin{m:g}': {'all': balanced_summary(
                      [r for r in rows if r['horizon_ms'] == h and r['margin_deg'] == m])}
                      for h in (50, 1000) for m in (0., 5.)})
    (directory / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    return directory


def test_cli_only_path_from_another_cwd_exports_canonical_metrics_and_can_rerun(tmp_path):
    run = tmp_path / 'run with spaces'
    directory = make_evaluation(run)
    make_evaluation(run, 'baf_roi_heatmap_alpha')
    (run / 'experiment.json').write_text('{}', encoding='utf-8')
    training = run / 'models' / 'baf_roi_heatmap_alpha'
    training.mkdir(parents=True)
    (training / 'training.json').write_text(json.dumps({'history': [
        {'epoch': 0, 'train_loss': None, 'validation_recording_macro_iou': .75},
        {'epoch': 1, 'train_loss': .2, 'validation_recording_macro_iou': .7},
    ]}), encoding='utf-8')
    original = (directory / 'per_origin.csv').read_bytes()
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'plot_evaluation.py'
    for _ in range(2):
        result = subprocess.run([sys.executable, str(script), '--path', str(run)],
                                cwd=tmp_path, text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stderr
    output = run / 'plots' / 'test'
    manifest = json.loads((output / 'plot_manifest.json').read_text())
    assert manifest['origins'] == 3
    assert manifest['recordings'] == 2
    assert manifest['primary_margin_deg'] == 0
    assert manifest['training'][0]['best_history_epoch'] == 0
    with (output / 'metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    pooled = next(r for r in rows if r['aggregation'] == 'pooled')
    macro = next(r for r in rows if r['aggregation'] == 'recording_macro')
    assert float(pooled['iou_mean']) == pytest.approx(2 / 3)
    assert float(macro['iou_mean']) == pytest.approx(.75)
    assert float(pooled['f1_micro']) == pytest.approx(.8)
    assert (output / 'index.html').is_file()
    for plot in manifest['plots']:
        for extension in ('png', 'pdf'):
            assert (output / f"{plot['name']}.{extension}").stat().st_size > 1000
    assert (directory / 'per_origin.csv').read_bytes() == original


def test_altered_evaluation_rows_are_rejected(tmp_path):
    directory = make_evaluation(tmp_path)
    with (directory / 'per_origin.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='SHA-256'):
        load_evaluations([directory])


@pytest.mark.parametrize('field', ['protocol_id', 'origin_keys_sha256', 'targets_sha256', 'rf_cohort_sha256'])
def test_different_protocol_or_cohort_is_rejected_even_with_equal_counts(tmp_path, field):
    left = make_evaluation(tmp_path)
    right = make_evaluation(tmp_path, 'baf_roi_heatmap')
    path = right / 'report.json'
    report = json.loads(path.read_text())
    report[field] = 'different'
    path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match=field):
        load_evaluations([left, right])


def test_actual_origins_are_checked_as_well_as_report_hashes(tmp_path):
    left = make_evaluation(tmp_path)
    right = make_evaluation(tmp_path, 'baf_roi_heatmap')
    path = right / 'per_origin.csv'
    path.write_text(path.read_text().replace('o0,', 'different_origin,'), encoding='utf-8')
    report_path = right / 'report.json'
    report = json.loads(report_path.read_text())
    report['rows_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='actual origin cohorts'):
        load_evaluations([left, right])


def test_missing_zero_margin_is_disclosed_and_uses_smallest_available(tmp_path):
    directory = make_evaluation(tmp_path)
    evaluations = load_evaluations([directory])
    evaluations[0].metrics = evaluations[0].metrics.query('margin_deg == 5')
    assert primary_margin(evaluations) == 5
