"""Checkpoint-free plots and an HTML report for unified ROI evaluations."""
import argparse
from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = {
    'baf_roi': 'BAF', 'motion_transformer_roi': 'Motion Transformer',
    'baf_roi_heatmap': 'BAF + heatmap residual',
    'baf_roi_heatmap_alpha': 'BAF + heatmap alpha',
}
MATCH_FIELDS = (
    'experiment_id', 'scene_id', 'split', 'horizons_ms', 'margins_deg',
    'delivery_delay_s', 'protocol_id', 'origin_keys_sha256', 'targets_sha256',
    'rf_cohort_sha256',
)
ROW_KEYS = ['origin_id', 'horizon_ms', 'margin_deg']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Evaluation:
    method: str
    report: dict
    rows: pd.DataFrame
    metrics: pd.DataFrame


def load_evaluations(directories):
    """Refuse altered rows or different cohorts, even when their sizes match."""
    evaluations = []
    for directory in directories:
        directory = Path(directory)
        report = read_json(directory / 'report.json')
        path = directory / 'per_origin.csv'
        if sha256(path) != report['rows_sha256']:
            raise ValueError(f'{path}: rows changed after evaluation (SHA-256 mismatch).')
        if evaluations:
            for field in MATCH_FIELDS:
                if field not in report or report[field] != evaluations[0].report.get(field):
                    raise ValueError(f'Unmatched evaluations: {field} differs or is missing.')
        rows = pd.read_csv(path)
        required = set(ROW_KEYS + ['recording_id', 'group_id', 'scenario', 'tp', 'fp', 'fn',
                                  'tn', 'iou', 'translation_error_m', 'rotation_error_deg',
                                  'rf_available'])
        if required - set(rows.columns):
            raise ValueError(f'{path}: missing columns {sorted(required - set(rows.columns))}.')
        if rows.empty or rows.duplicated(ROW_KEYS).any():
            raise ValueError(f'{path}: empty data or duplicate origin/horizon/margin rows.')
        numeric = ['horizon_ms', 'margin_deg', 'tp', 'fp', 'fn', 'tn', 'iou',
                   'translation_error_m', 'rotation_error_deg', 'rf_available']
        if not np.isfinite(rows[numeric].to_numpy(dtype=float)).all():
            raise ValueError(f'{path}: required metrics must be finite.')
        rows = rows.sort_values(ROW_KEYS).reset_index(drop=True)
        if evaluations:
            reference = evaluations[0].rows
            cohort = ROW_KEYS + ['recording_id', 'group_id', 'scenario', 'rf_available']
            if not rows[cohort].equals(reference[cohort]):
                raise ValueError('Unmatched evaluations: actual origin cohorts differ.')
            if not np.array_equal(rows.tp + rows.fn, reference.tp + reference.fn):
                raise ValueError('Unmatched evaluations: ground-truth partition counts differ.')
        metrics = []
        for key, summary in report['summaries'].items():
            horizon, margin = key.split('ms_margin')
            for aggregation in ('pooled', 'recording_macro', 'scenario_macro'):
                values = summary['all'].get(aggregation)
                if values is not None:
                    metrics.append(dict(values, method=directory.name, aggregation=aggregation,
                                        horizon_ms=float(horizon), margin_deg=float(margin),
                                        n_origins=summary['all']['pooled']['n_origins']))
        metrics = pd.DataFrame(metrics)
        actual = set(map(tuple, rows[['horizon_ms', 'margin_deg']].drop_duplicates().to_numpy()))
        reported = set(map(tuple, metrics.loc[metrics.aggregation == 'pooled',
                                              ['horizon_ms', 'margin_deg']].to_numpy()))
        if actual != reported:
            raise ValueError(f'{path}: reported horizon/margin groups do not match rows.')
        evaluations.append(Evaluation(directory.name, report, rows, metrics))
    if not evaluations or len({e.method for e in evaluations}) != len(evaluations):
        raise ValueError('Supply nonempty evaluations with distinct method directory names.')
    return sorted(evaluations, key=lambda e: list(LABELS).index(e.method)
                  if e.method in LABELS else len(LABELS))


def discover(path):
    """Accept a run, evaluations directory, split directory, or one method directory."""
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f'Not a directory: {path}')
    if (path / 'report.json').is_file():
        reports = [path / 'report.json']
    else:
        reports = []
        for pattern in ('evaluations/*/*/report.json', '*/*/report.json', '*/report.json'):
            reports = sorted(path.glob(pattern))
            if reports:
                break
    if not reports:
        raise ValueError(f'No report.json / per_origin.csv evaluations found under {path}.')
    groups = {}
    for report in reports:
        split = read_json(report)['split']
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f'Unsupported evaluation split: {split!r}')
        groups.setdefault(split, []).append(report.parent)
    run = next((p for p in (path, *path.parents) if (p / 'experiment.json').is_file()), path)
    return path, run, groups


def primary_margin(evaluations):
    """Use zero margin if available everywhere, otherwise the smallest common margin."""
    candidates = None
    for evaluation in evaluations:
        for _, group in evaluation.metrics.query("aggregation == 'pooled'").groupby('horizon_ms'):
            available = set(group.margin_deg)
            candidates = available if candidates is None else candidates & available
    if not candidates:
        raise ValueError('No common prediction margin across all methods and horizons.')
    return min(candidates)


def training_histories(run, methods):
    """Read both nested BAF/MT losses and flat adapter validation IoU histories."""
    result = []
    for method in methods:
        directory = run / 'models' / method
        path = next((p for p in (directory / 'training_history.json', directory / 'training.json')
                     if p.is_file()), None)
        if path is None:
            continue
        document = read_json(path)
        for stage, history in document.items():
            if not isinstance(history, list) or not history or not isinstance(history[0], dict):
                continue
            if 'epoch' not in history[0]:
                continue
            values = pd.json_normalize(history).set_index('epoch').sort_index()
            validation = ('validation_recording_macro_iou' if 'validation_recording_macro_iou' in values
                          else 'valid.loss_total')
            if validation not in values or values[validation].dropna().empty:
                continue
            best = (values[validation].idxmax() if validation.endswith('iou')
                    else values[validation].idxmin())
            result.append(dict(method=method, stage=stage, values=values, validation=validation,
                               best_history_epoch=int(best),
                               first_validation=float(values[validation].dropna().iloc[0]),
                               best_validation=float(values.loc[best, validation]),
                               last_validation=float(values[validation].dropna().iloc[-1])))
    return result


def render(evaluations, run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    margin = primary_margin(evaluations)
    split = evaluations[0].report['split']
    all_metrics = pd.concat([e.metrics for e in evaluations], ignore_index=True)
    primary = all_metrics.loc[all_metrics.margin_deg == margin]
    all_metrics.to_csv(output / 'metrics.csv', index=False)
    horizons = sorted(primary.horizon_ms.unique())
    baseline = next((e.method for e in evaluations if e.method == 'baf_roi'), None)
    histories = training_histories(run, [e.method for e in evaluations])
    plots = []
    styles = {e.method: dict(color=f'C{i % 10}', marker=['o', 's', '^', 'x'][i % 4],
                             linestyle=['-', '--', '-.', ':'][i % 4], markersize=4,
                             markevery=max(1, len(horizons) // 10), linewidth=1.7)
              for i, e in enumerate(evaluations)}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def label(method):
        return LABELS.get(method, method)

    def save(fig, name, caption):
        fig.suptitle(caption, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .96))
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'{name}.{ext}', dpi=170, bbox_inches='tight')
        plt.close(fig)
        plots.append(dict(name=name, caption=caption))

    def curves(axes, specs, aggregation='pooled'):
        for ax, (metric, title, bounded) in zip(np.ravel(axes), specs):
            for evaluation in evaluations:
                data = primary.loc[(primary.method == evaluation.method) &
                                   (primary.aggregation == aggregation)].sort_values('horizon_ms')
                if metric in data and data[metric].notna().any():
                    ax.plot(data.horizon_ms, data[metric], label=label(evaluation.method),
                            **styles[evaluation.method])
            ax.set(xlabel='Prediction horizon (ms)', ylabel=title)
            if bounded:
                ax.set_ylim(0, 1.02)
            else:
                ax.set_ylim(bottom=0)
            ax.grid(alpha=.22)
            ax.legend(fontsize=8)

    suffix = f'{split} | prediction margin {margin:g} deg'
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    curves(axes, [('iou_mean', 'Mean IoU (per-origin mean)', True),
                  ('f1_micro', 'Micro F1 (summed TP/FP/FN)', True),
                  ('coverage_micro', 'Coverage / recall', True),
                  ('precision_micro', 'Precision', True)])
    save(fig, '01_roi_metrics', f'ROI accuracy | pooled | {suffix}')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    for ax, aggregation in zip(axes, ('pooled', 'recording_macro', 'scenario_macro')):
        curves([ax], [('iou_mean', 'Mean IoU', True)], aggregation)
        ax.set_title(aggregation.replace('_', ' '))
    save(fig, '02_aggregation', f'Origin / recording / scenario weighting | {suffix}')

    if baseline and len(evaluations) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
        for ax, aggregation in zip(axes, ('pooled', 'recording_macro')):
            base = primary.loc[(primary.method == baseline) & (primary.aggregation == aggregation)]
            base = base.set_index('horizon_ms').iou_mean
            for evaluation in evaluations:
                if evaluation.method == baseline:
                    continue
                data = primary.loc[(primary.method == evaluation.method) &
                                   (primary.aggregation == aggregation)].set_index('horizon_ms').sort_index()
                ax.plot(data.index, (data.iou_mean - base) * 100, label=label(evaluation.method),
                        **styles[evaluation.method])
            ax.axhline(0, color='black', linewidth=.8)
            ax.set(xlabel='Prediction horizon (ms)', ylabel='Delta Mean IoU (percentage points)',
                   title=aggregation.replace('_', ' '))
            ax.grid(alpha=.22)
            ax.legend(fontsize=8)
        save(fig, '03_delta_vs_baf', f'Improvement over BAF (higher is better) | {suffix}')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    curves(axes, [('translation_mean_m', 'Mean position error (m)', False),
                  ('rotation_mean_deg', 'Mean rotation error (deg)', False),
                  ('translation_p95_m', '95th-percentile position error (m)', False),
                  ('rotation_p95_deg', '95th-percentile rotation error (deg)', False)])
    save(fig, '04_pose_errors', f'Pose errors (lower is better) | {suffix}')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    curves(axes, [('fn_mean', 'Missed partitions / origin (FN)', False),
                  ('fp_mean', 'Redundant partitions / origin (FP)', False),
                  ('selected_mean', 'Selected partitions / origin', False)])
    required = primary.loc[(primary.method == evaluations[0].method) & (primary.aggregation == 'pooled')]
    axes[-1].plot(required.horizon_ms, required.required_mean, color='black', linestyle='--', label='Required (GT)')
    axes[-1].legend(fontsize=8)
    save(fig, '05_partition_cost', f'Misses and selection cost | {suffix}')

    selected_horizons = sorted({min(horizons, key=lambda h: abs(h - target)) for target in (200, 500, 1000)})
    fig, axes = plt.subplots(1, len(selected_horizons), figsize=(5.4 * len(selected_horizons), 4.9), squeeze=False)
    for ax, horizon in zip(axes.ravel(), selected_horizons):
        for evaluation in evaluations:
            data = all_metrics.loc[(all_metrics.method == evaluation.method) &
                                   (all_metrics.aggregation == 'pooled') &
                                   (all_metrics.horizon_ms == horizon)].sort_values('margin_deg')
            ax.plot(data.selected_mean, data.coverage_micro, label=label(evaluation.method),
                    **dict(styles[evaluation.method], markevery=1))
            # All methods share these margins. Label one curve to keep coincident
            # BAF / alpha points and nearby residual points readable.
            if evaluation is evaluations[0]:
                for row in data.itertuples():
                    ax.annotate(f'{row.margin_deg:g}', (row.selected_mean, row.coverage_micro),
                                xytext=(3, 6), textcoords='offset points', fontsize=8)
        ax.set(title=f'{horizon:g} ms', xlabel='Mean selected partitions', ylabel='Coverage / recall', ylim=(0, 1.03))
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)
    save(fig, '06_coverage_tradeoff', f'Coverage versus selection cost | point labels: margin (deg) | {split}')

    recording_tables = []
    origin_tables = []
    for evaluation in evaluations:
        rows = evaluation.rows.loc[evaluation.rows.margin_deg == margin]
        table = rows.groupby(['scenario', 'recording_id', 'horizon_ms'], sort=True).agg(
            iou_mean=('iou', 'mean'), n_origins=('iou', 'size')).reset_index()
        recording_tables.append(table.assign(method=evaluation.method))
        # One descriptive point per origin, averaging its horizons; no IID error bars.
        origin_tables.append(rows.groupby('origin_id').iou.mean().rename(evaluation.method))
    recordings = pd.concat(recording_tables, ignore_index=True)
    recordings.to_csv(output / 'recording_metrics.csv', index=False)
    scenarios = recordings.groupby(['method', 'scenario', 'horizon_ms']).iou_mean.mean().reset_index()
    scenarios.to_csv(output / 'scenario_metrics.csv', index=False)
    scenario_average = scenarios.groupby(['scenario', 'method']).iou_mean.mean().unstack('method')
    scenario_average = scenario_average.reindex(columns=[e.method for e in evaluations])
    fig, ax = plt.subplots(figsize=(max(8, len(evaluations) * 2.7), max(3.5, len(scenario_average) * .8)))
    heat = ax.imshow(scenario_average.to_numpy(), vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(evaluations)), [label(e.method) for e in evaluations], rotation=12, ha='right')
    ax.set_yticks(range(len(scenario_average)), scenario_average.index)
    for i in range(len(scenario_average)):
        for j in range(len(evaluations)):
            value = scenario_average.iloc[i, j]
            ax.text(j, i, f'{value:.4f}', ha='center', va='center', color='white' if value > .65 else 'black')
    fig.colorbar(heat, ax=ax, label='Recording-macro IoU\n(equal mean over horizons)')
    save(fig, '07_scenario_iou', f'Scenario breakdown | {suffix}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for evaluation, series in zip(evaluations, origin_tables):
        values = np.sort(series.to_numpy())
        axes[0].plot(values, np.arange(1, len(values) + 1) / len(values),
                     label=label(evaluation.method), **dict(styles[evaluation.method], marker=None))
    axes[0].set(xlabel='Per-origin IoU averaged over horizons', ylabel='Empirical cumulative fraction', xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8)
    cohort = evaluations[0].rows.loc[(evaluations[0].rows.margin_deg == margin) &
                                    (evaluations[0].rows.horizon_ms == horizons[0])]
    counts = cohort.groupby('scenario').size().sort_index()
    bars = axes[1].bar(counts.index, counts.values, color='C0')
    axes[1].bar_label(bars, labels=[f'{n:,} ({n / counts.sum():.1%})' for n in counts], padding=4)
    axes[1].set(ylabel='Distinct prediction origins', ylim=(0, counts.max() * 1.2))
    axes[1].tick_params(axis='x', rotation=12)
    for ax in axes:
        ax.grid(axis='y', alpha=.22)
    save(fig, '08_distribution_and_cohort', f'Error distribution and test composition | {suffix}')

    if baseline:
        means = recordings.groupby(['recording_id', 'method']).iou_mean.mean().unstack('method')
        alternatives = [e.method for e in evaluations if e.method != baseline]
        if alternatives:
            delta = means[alternatives].subtract(means[baseline], axis=0) * 100
            delta.columns = [label(m) for m in alternatives]
            delta.index = [str(i).split(':')[0] + ' / ' + str(i).split(':')[-1][:8]
                           for i in delta.index]
            fig, ax = plt.subplots(figsize=(13, max(4.5, len(delta) * .65)))
            delta.plot.barh(ax=ax, color=[styles[m]['color'] for m in alternatives])
            ax.axvline(0, color='black', linewidth=.8)
            ax.set(xlabel='Delta Mean IoU vs BAF (percentage points), equal mean over horizons', ylabel='Source recording')
            ax.grid(axis='x', alpha=.22)
            save(fig, '09_recording_delta', f'Gain or regression on individual recordings | {suffix}')

    if histories:
        fig, axes = plt.subplots(len(histories), 2, figsize=(13, 3.2 * len(histories)), squeeze=False)
        for (loss_ax, val_ax), history in zip(axes, histories):
            values = history['values']
            train = 'train_loss' if 'train_loss' in values else 'train.loss_total'
            if train in values:
                loss_ax.plot(values.index, values[train], color='C0', label='Train loss')
            if 'valid.loss_total' in values:
                loss_ax.plot(values.index, values['valid.loss_total'], color='C1', label='Validation loss')
            loss_ax.set(title=f"{label(history['method'])} / {history['stage']}", ylabel='Loss (within this stage only)')
            val_ax.plot(values.index, values[history['validation']], color='C1', label=history['validation'])
            val_ax.axvline(history['best_history_epoch'], color='black', linestyle='--',
                           label=f"Best in history: epoch {history['best_history_epoch']}")
            val_ax.set(ylabel='Recording-macro IoU' if history['validation'].endswith('iou') else 'Validation loss')
            for ax in (loss_ax, val_ax):
                ax.set_xlabel('Epoch (stage-local)')
                ax.grid(alpha=.22)
                ax.legend(fontsize=8)
        save(fig, '10_training', 'Training and validation | best epoch from history; losses have different scales')

    average = primary.loc[primary.aggregation == 'pooled'].groupby('method')[
        ['iou_mean', 'f1_micro', 'coverage_micro', 'precision_micro', 'translation_mean_m', 'rotation_mean_deg']].mean()
    average = average.reindex([e.method for e in evaluations])
    if baseline:
        average['delta_iou_pp_vs_baf'] = (average.iou_mean - average.loc[baseline, 'iou_mean']) * 100
    average.to_csv(output / 'horizon_average.csv')
    training_summary = [{k: v for k, v in history.items() if k not in ('values', 'validation')} for history in histories]
    manifest = dict(split=split, primary_margin_deg=margin, horizons_ms=horizons,
                    methods=[e.method for e in evaluations], plots=plots,
                    origins=int(len(cohort)), recordings=int(cohort.recording_id.nunique()),
                    groups=int(cohort.group_id.nunique()), scenario_origins={str(k): int(v) for k, v in counts.items()},
                    training=training_summary,
                    input_sha256={e.method: e.report['rows_sha256'] for e in evaluations},
                    notes=['Horizon averages give every evaluated horizon equal weight.',
                           'History best epoch does not independently verify which checkpoint was evaluated.',
                           'Coverage and partition counts are geometric proxies, not SSIM or measured bitrate.',
                           'Origins overlap in time; plots are descriptive, without IID significance claims.',
                           'Within-run comparisons only; improvement over an earlier run requires matched earlier results.'])
    (output / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    title = f'ROI evaluation | {split} | margin {margin:g} deg'
    gallery = '\n'.join(f'<section><h2>{html.escape(p["caption"])}</h2>'
                        f'<a href="{p["name"]}.pdf">PDF</a> · <a href="{p["name"]}.png">PNG</a>'
                        f'<img loading="lazy" src="{p["name"]}.png" alt="{html.escape(p["caption"])}"></section>'
                        for p in plots)
    table = average.rename(index=LABELS).to_html(float_format=lambda v: f'{v:.5f}', border=0)
    training_table = pd.DataFrame(training_summary).to_html(index=False, float_format=lambda v: f'{v:.6f}', border=0) if histories else '<p>No training histories supplied.</p>'
    notes = ''.join(f'<li>{html.escape(note)}</li>' for note in manifest['notes'])
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>body{{font:15px system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 20px;color:#202c3c;background:#f5f7fa}}section{{background:white;padding:20px;margin:24px 0;border-radius:12px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;display:block;overflow-x:auto}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}a{{color:#1365b1}}li{{margin:6px 0}}</style>
<h1>{html.escape(title)}</h1><p>{len(cohort):,} origins · {manifest['recordings']} source recordings · {manifest['groups']} groups · {len(horizons)} horizons · matched evaluation cohorts verified.</p>
<section><h2>Equal mean over evaluated horizons (pooled at each horizon)</h2>{table}
<p>Delta is in percentage points; pose errors are in metres / degrees. Alpha and BAF may overlap exactly; use the delta plot to check.</p>
<p><a href="metrics.csv">All metrics / margins / aggregations CSV</a> · <a href="recording_metrics.csv">Recording CSV</a> · <a href="scenario_metrics.csv">Scenario CSV</a> · <a href="horizon_average.csv">Horizon averages CSV</a> · <a href="plot_manifest.json">Manifest</a></p></section>
<section><h2>Training diagnostics</h2>{training_table}</section><ul>{notes}</ul>{gallery}</html>'''
    (output / 'index.html').write_text(content, encoding='utf-8')
    return manifest


def plot_evaluation(path):
    selected, run, groups = discover(path)
    results = []
    # Validate every split before writing any plots.
    loaded = {split: load_evaluations(directories) for split, directories in groups.items()}
    for split, evaluations in loaded.items():
        output = selected / 'plots' / split
        manifest = render(evaluations, run, output)
        results.append(dict(output=str(output), plots=len(manifest['plots']), split=split))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True, help='Experiment run, evaluations, split, or method directory.')
    args = parser.parse_args(argv)
    try:
        results = plot_evaluation(args.path)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    for result in results:
        print(f"{result['split']}: {result['plots']} plots (PNG + PDF)")
        print(f"Report: {Path(result['output']) / 'index.html'}")


if __name__ == '__main__':
    main()
