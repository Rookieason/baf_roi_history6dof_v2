"""Matched Mean IoU/F1 and geometric coverage/selection curves for all methods."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from baf_roi_heatmap.contracts import new_directory, read_json, write_json, file_hash
from .evaluation import balanced_summary, read_rows


METRICS = {
    'iou_mean': 'Mean IoU', 'f1_micro': 'Micro F1', 'coverage_micro': 'Coverage / recall',
    'fp_mean': 'Redundant partitions (FP)', 'fn_mean': 'Missed partitions (FN)',
    'selected_mean': 'Selected partitions', 'translation_mean_m': 'Position error (m)',
    'rotation_mean_deg': 'Rotation error (degrees)',
}


def method_label(method):
    if isinstance(method, str):
        return {'baf': 'BAF-Pose', 'motion_transformer': 'Motion Transformer'}.get(method, method)
    name = method.get('method', 'adapter')
    if name == 'baf_roi_heatmap_alpha':
        return 'BAF + heatmap alpha'
    if name == 'baf_roi_heatmap_residual':
        return 'BAF + internal residual / ' + method.get('modality', 'rf') + ' / ' + method.get('ablation', 'full')
    return ' / '.join(str(method[k]) for k in ('method', 'modality', 'ablation') if k in method)


def plot(directories, output, aggregation='pooled', margin=0.):
    """Horizon plots use one named margin; each horizon also gets a full margin sweep.

    Coverage curves vary prediction FoV only. These are not neural-confidence
    threshold curves and they are not a measurement of rendered SSIM.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if aggregation not in ('pooled', 'recording_macro', 'scenario_macro'):
        raise ValueError('Unknown aggregation.')
    if not directories or not np.isfinite(margin) or margin < 0:
        raise ValueError('Supply evaluations and a finite nonnegative prediction margin.')
    reports = [read_json(Path(d)/'report.json') for d in directories]
    for key in ('experiment_id', 'scene_id', 'split', 'horizons_ms', 'delivery_delay_s',
                'protocol_id', 'origin_keys_sha256', 'targets_sha256', 'rf_cohort_sha256'):
        values = [r.get(key) for r in reports]
        if any(v != values[0] for v in values[1:]):
            raise ValueError(f'Plot only matched evaluations: {key} differs.')
    grouped = []
    for directory, report in zip(directories, reports):
        if file_hash(Path(directory)/'per_origin.csv') != report['rows_sha256']:
            raise ValueError('Evaluation rows changed after reporting.')
        groups = defaultdict(list)
        for row in read_rows(directory):
            groups[(row['horizon_ms'], row['margin_deg'])].append(row)
        metrics = {key: balanced_summary(rows)[aggregation] for key, rows in groups.items()}
        if any(value is None for value in metrics.values()):
            raise ValueError(f'{aggregation} is unavailable; supply scenario metadata or select pooled/recording_macro.')
        grouped.append(metrics)
    horizons = reports[0]['horizons_ms']
    for metrics in grouped:
        if any((h, float(margin)) not in metrics for h in horizons):
            raise ValueError(f'Margin {margin:g} is not evaluated at every horizon; choose an available margin.')
    out = new_directory(output)
    names = []
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(out/f'{name}.{extension}', dpi=180)
        plt.close(fig)
        names.append(name)

    for metric, title in METRICS.items():
        fig, ax = plt.subplots(figsize=(7.5, 4.7))
        for report, values in zip(reports, grouped):
            ys = [values[(h, float(margin))][metric] for h in horizons]
            ax.plot(horizons, ys, marker='o', markersize=4, linewidth=1.8, label=method_label(report['method']))
        if aggregation != 'pooled':
            title = f'{title} (mean over {"recordings" if aggregation == "recording_macro" else "scenarios"})'
        ax.set_xlabel('Prediction horizon (ms)'); ax.set_ylabel(title)
        ax.set_title(f'Prediction margin {margin:g}° · {aggregation.replace("_", " ")}')
        ax.grid(True, alpha=.22); ax.legend(fontsize=8)
        save(fig, metric+'_vs_horizon')

    for horizon in horizons:
        fig, ax = plt.subplots(figsize=(7.5, 4.7))
        for report, values in zip(reports, grouped):
            points = sorted((m, v) for (h, m), v in values.items() if h == horizon)
            ax.plot([v['selected_mean'] for _, v in points], [v['coverage_micro'] for _, v in points],
                    marker='o', linewidth=1.8, label=method_label(report['method']))
            for m, v in points:
                ax.annotate(f'{m:g}°', (v['selected_mean'], v['coverage_micro']),
                            textcoords='offset points', xytext=(3, 4), fontsize=7)
        ax.set_xlabel('Mean selected partitions'); ax.set_ylabel('Coverage / recall')
        ax.set_title(f'{horizon:g} ms · prediction margin sweep · {aggregation.replace("_", " ")}')
        ax.grid(True, alpha=.22); ax.legend(fontsize=8)
        save(fig, f'coverage_vs_selected_{horizon:g}ms')
    result = {'output': str(out), 'plots': len(names), 'files': names, 'aggregation': aggregation,
              'horizon_plot_margin_deg': float(margin),
              'primary_definition': 'Pooled Mean IoU averages origins; pooled Micro F1 sums TP/FP/FN first.',
              'quality_status': 'Geometric coverage only; rendered SSIM is not evaluated.'}
    write_json(out/'plot_manifest.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--aggregation', choices=['pooled', 'recording_macro', 'scenario_macro'], default='pooled')
    parser.add_argument('--margin', type=float, default=0.)
    args = parser.parse_args(argv)
    print(plot(args.evaluations, args.output, args.aggregation, args.margin))


if __name__ == '__main__':
    main()
