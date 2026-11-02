"""Hindsight candidate diagnostics and plots, deliberately separate from deployed inference."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from baf_roi.upstream_ops import slerp_quaternion_wxyz
from .contracts import read_json, write_json, new_directory
from .data import ForecastDataset
from .evaluation import roi_counts, read_rows


def oracle(prepared, output_path, split='valid', grid_points=21):
    if grid_points < 2 or Path(output_path).exists():
        raise ValueError('Use at least two candidate weights and a new output path.')
    dataset = ForecastDataset(prepared, split)
    records = []
    for ri, record in enumerate(dataset.records):
        a = dataset.arrays(ri)
        def iou(p, q):
            tp, fp, fn, _ = roi_counts(dataset.scene.mask(p, q), a['target_mask'])
            union = tp+fp+fn
            return np.divide(tp, union, out=np.ones_like(tp, dtype=float), where=union != 0)
        original = iou(a['base_p'], a['base_q'])
        best = original.copy()
        for weight in np.linspace(0, 1, grid_points):
            p = (1-weight)*a['focused_p'] + weight*a['explorer_p']
            shape = a['focused_q'].shape
            q = slerp_quaternion_wxyz(a['focused_q'].reshape(-1, 4), a['explorer_q'].reshape(-1, 4),
                                      np.full(np.prod(shape[:-1]), weight)).reshape(shape)
            best = np.maximum(best, iou(p, q))
        records.append({'record': record['key'], 'baseline_iou': original.mean(0).tolist(),
                        'best_tested_candidate_iou': best.mean(0).tolist(), 'headroom': (best-original).mean(0).tolist()})
    result = {'experiment_id': dataset.manifest['experiment_id'], 'split': split, 'horizons_ms': dataset.horizons,
              'grid_points': grid_points, 'records': records,
              'recording_macro_headroom': np.mean([r['headroom'] for r in records], axis=0).tolist(),
              'warning': 'Hindsight uses ground-truth ROI. Not deployable and NOT an upper bound over all continuous weights or retrained experts. Use validation for design choices.'}
    write_json(output_path, result)
    return result


def plot(directories, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = new_directory(output)
    reports = [read_json(Path(d)/'report.json') for d in directories]
    if len({r['experiment_id'] for r in reports}) != 1:
        raise ValueError('Plot only matched prepared experiments.')
    for metric in ('iou_mean', 'f1_micro', 'coverage_micro', 'fp_mean', 'fn_mean', 'selected_mean', 'translation_mean_m', 'rotation_mean_deg'):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for directory, report in zip(directories, reports):
            groups = defaultdict(list)
            for row in read_rows(directory): groups[row['horizon_ms']].append(row)
            for values in groups.values():
                if len({v['margin_deg'] for v in values}) != 1:
                    raise ValueError('Plot one policy/margin per horizon, not a margin sweep.')
            from .evaluation import balanced_summary
            horizons = sorted(groups)
            values = [balanced_summary(groups[h])['recording_macro'][metric] for h in horizons]
            method = report['method']
            label = method if isinstance(method, str) else '/'.join(method[k] for k in ('method', 'modality', 'ablation'))
            ax.plot(horizons, values, marker='o', label=label)
        ax.set_xlabel('Prediction horizon (ms)'); ax.set_ylabel(metric + ' (recording macro)')
        ax.grid(True, alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out/(metric+'.png'), dpi=160); plt.close(fig)
    return {'output': str(out), 'plots': 8}
