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


def plot(directories, output, aggregation='pooled', margin=0.):
    """Compatibility wrapper for the shared, explicitly aggregated plots."""
    from roi_experiments.plot import plot as shared_plot
    return shared_plot(directories, output, aggregation, margin)
