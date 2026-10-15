"""Plot one or more evaluator metrics.csv files without loading a checkpoint."""
import argparse
import csv
import math
from pathlib import Path


def read_metrics(path):
    """Keep macro IoU and normalize accumulated counts per prediction origin."""
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        required = {'horizon_ms', 'iou_mean', 'tp', 'fp', 'fn', 'n_origins'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {sorted(required - set(reader.fieldnames or []))}')
        rows = []
        for raw in reader:
            values = {key: float(raw[key]) for key in required}
            if not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f'{path}: metrics must be finite.')
            h, n = values['horizon_ms'], values['n_origins']
            tp, fp, fn = (values[k] for k in ('tp', 'fp', 'fn'))
            if not 50 <= h <= 1000 or n <= 0 or n != int(n) or min(tp, fp, fn) < 0 or not 0 <= values['iou_mean'] <= 1:
                raise ValueError(f'{path}: invalid horizon, count, or IoU.')
            rows.append(dict(horizon_ms=h, n_origins=n, iou_mean=values['iou_mean'],
                             missed=fn / n, redundant=fp / n, selected=(tp + fp) / n,
                             coverage=tp / (tp + fn) if tp + fn else 0.))
    if not rows or len({r['horizon_ms'] for r in rows}) != len(rows):
        raise ValueError(f'{path}: expected nonempty metrics with distinct horizons.')
    if len({r['n_origins'] for r in rows}) != 1:
        raise ValueError(f'{path}: horizons must use the same number of origins.')
    return sorted(rows, key=lambda r: r['horizon_ms'])


def plot_metrics(metrics_paths, output, labels=None, include_coverage=False):
    paths = [Path(p) for p in metrics_paths]
    if not paths:
        raise ValueError('Supply at least one metrics.csv file.')
    labels = list(labels) if labels is not None else [p.parent.name for p in paths]
    if len(labels) != len(paths) or len(set(labels)) != len(labels):
        raise ValueError('Supply one distinct label per metrics.csv file.')
    data = [read_metrics(p) for p in paths]
    horizons = [r['horizon_ms'] for r in data[0]]
    if any([r['horizon_ms'] for r in rows] != horizons for rows in data[1:]):
        raise ValueError('Model comparisons require the same horizons.')
    if len({rows[0]['n_origins'] for rows in data}) != 1:
        raise ValueError('Model comparisons require the same number of origins.')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    names = ['mean_iou_vs_horizon', 'missed_redundant_vs_horizon']
    if include_coverage:
        names.append('coverage_vs_selected')
    files = [output / f'{name}.{ext}' for name in names for ext in ('png', 'pdf')]
    if any(p.exists() for p in files):
        raise ValueError('Plot files already exist; use a new output directory.')
    output.mkdir(parents=True, exist_ok=True)

    for name in names:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), layout='constrained')
        try:
            for i, (label, rows) in enumerate(zip(labels, data)):
                color = f'C{i % 10}'
                if name == 'mean_iou_vs_horizon':
                    ax.plot(horizons, [r['iou_mean'] for r in rows], marker='o', color=color, label=label)
                elif name == 'missed_redundant_vs_horizon':
                    for key, title, style, marker in [('missed', 'Missed (FN)', '-', 'o'),
                                                       ('redundant', 'Redundant (FP)', '--', 's')]:
                        ax.plot(horizons, [r[key] for r in rows], linestyle=style, marker=marker,
                                color=color, label=f'{label}: {title}')
                else:
                    ax.scatter([r['selected'] for r in rows], [r['coverage'] for r in rows], color=color, label=label)
                    for r in rows:
                        ax.annotate(f"{r['horizon_ms']:g} ms", (r['selected'], r['coverage']),
                                    xytext=(5, 5), textcoords='offset points', fontsize=8)
            if name == 'coverage_vs_selected':
                ax.set(xlabel='Average Selected Partitions per Origin', ylabel='ROI Coverage (Recall)',
                       title='ROI Coverage vs. Selected Partitions (Operating Points)', ylim=(0, 1.05))
            else:
                ax.set_xlabel('Prediction Horizon (ms)')
                ax.set_xticks(horizons)
                if name == 'mean_iou_vs_horizon':
                    ax.set(ylabel='Mean IoU', title='Mean IoU vs. Prediction Horizon', ylim=(0, 1.05))
                else:
                    ax.set(ylabel='Average Number of Partitions per Prediction Origin',
                           title='Missed / Redundant Partitions vs. Prediction Horizon', ylim=(0, None))
            ax.grid(alpha=.25)
            ax.legend()
            for ext in ('png', 'pdf'):
                fig.savefig(output / f'{name}.{ext}', dpi=200)
        finally:
            plt.close(fig)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', nargs='+', required=True, help='One or more evaluation metrics.csv files.')
    parser.add_argument('--labels', nargs='+', help='One distinct model label per input file.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--include-coverage', action='store_true', help='Add coverage operating points, annotated by horizon.')
    args = parser.parse_args()
    try:
        files = plot_metrics(args.metrics, args.output, args.labels, args.include_coverage)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    for path in files:
        print(path)


if __name__ == '__main__':
    main()
