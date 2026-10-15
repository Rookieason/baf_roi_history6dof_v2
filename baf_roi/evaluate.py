"""Common-origin, all-regime, exact-horizon evaluation. No test-time fitting."""
from pathlib import Path
import csv
import json
import time
import numpy as np
import torch
from .config import dump_json, validate_horizons
from .data import Cache
from .runtime import Predictor
from .pose import ypr_to_quat, quaternion_error_deg
from .roi import load_scene
from .upstream_ops import compute_temporal_optimized_alpha


def binary_counts(pred, truth):
    p, y = np.asarray(pred, bool), np.asarray(truth, bool)
    return np.stack([(p & y).sum(-1), (p & ~y).sum(-1), (~p & y).sum(-1), (~p & ~y).sum(-1)], axis=-1)


class Metrics:
    def __init__(self):
        self.n = self.exact = self.gt_empty = self.empty_both = 0
        self.c = np.zeros(4, np.int64)
        self.iou_sum = self.iou_nonempty_sum = 0.
        self.nonempty = 0
        self.translation, self.rotation = [], []

    def add(self, p, y, tr, rot):
        c = binary_counts(p, y)
        tp, fp, fn, tn = c.T
        union = tp + fp + fn
        iou = np.divide(tp, union, out=np.ones(len(c), np.float64), where=union != 0)
        self.c += c.sum(0); self.n += len(c)
        self.exact += int(np.sum(fp + fn == 0))
        self.gt_empty += int(np.sum(tp + fn == 0)); self.empty_both += int(np.sum(union == 0))
        self.iou_sum += float(iou.sum()); self.iou_nonempty_sum += float(iou[union > 0].sum())
        self.nonempty += int(np.sum(union > 0))
        self.translation.extend(np.asarray(tr, float).tolist()); self.rotation.extend(np.asarray(rot, float).tolist())
        return c, iou

    def result(self):
        if not self.n:
            raise ValueError('No evaluation origins; insufficient history/future coverage.')
        tp, fp, fn, tn = map(int, self.c)
        tr, rot = np.asarray(self.translation), np.asarray(self.rotation)
        return {'n_origins': self.n, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'exact_match': self.exact / self.n, 'per_cell_accuracy': (tp + tn) / (tp + fp + fn + tn),
                'precision_micro': tp / (tp + fp) if tp + fp else 0.,
                'recall_micro': tp / (tp + fn) if tp + fn else 0.,
                'f1_micro': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                'iou_micro': tp / (tp + fp + fn) if tp + fp + fn else 1.,
                'iou_mean': self.iou_sum / self.n,
                'iou_mean_nonempty_union': self.iou_nonempty_sum / self.nonempty if self.nonempty else None,
                'ground_truth_empty_count': self.gt_empty, 'both_empty_count': self.empty_both,
                'translation_mean_m': float(tr.mean()), 'translation_p95_m': float(np.quantile(tr, .95)),
                'rotation_mean_deg': float(rot.mean()), 'rotation_p95_deg': float(np.quantile(rot, .95)),
                'translation_accuracy_at_0_02m': float(np.mean(tr <= .02)),
                'rotation_accuracy_at_5deg': float(np.mean(rot <= 5))}


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def evaluate(cache_path, checkpoint, output, device='cpu', split='test', batch_size=128,
             scene_cfg=None, save_masks=True, *, horizons_ms=None, predictor_class=None):
    if split not in ('valid', 'test') or batch_size < 1:
        raise ValueError('Evaluation split must be valid or test and batch size positive.')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a NEW evaluation directory, not an existing report.')
    cache = Cache(cache_path)
    predictor = (predictor_class or Predictor).load(checkpoint, device)
    uses_confidence = callable(getattr(predictor, 'confidence', None))
    if predictor.bundle['cache_id'] != cache.manifest['cache_id']:
        raise ValueError('Checkpoint/cache mismatch. Use the exact cache used for this model.')
    cfg = cache.manifest['config']
    hs = validate_horizons(cfg['evaluation']['horizons_ms'] if horizons_ms is None else horizons_ms)
    scene = load_scene(scene_cfg or cfg['scene'])
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / 'scene_snapshot.json', scene.snapshot())
    dump_json(output / 'split_manifest.json', cache.manifest['split'])
    metrics = [Metrics() for _ in hs]
    alpha_state, per_trajectory, timings = {}, [], []
    model_modes = {name: module.training for name, module in predictor.models.items()}
    assert not any(model_modes.values())
    csv_path = output / 'per_origin_metrics.csv'
    fields = ['trajectory', 'session', 'origin_relative_s', 'origin_epoch_s', 'horizon_ms', 'target_epoch_s',
              'alpha_t', 'tp', 'fp', 'fn', 'tn', 'iou', 'translation_error_m', 'rotation_error_deg']
    with csv_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for ti, record in enumerate(cache.records):
            if record['split'] != split:
                continue
            arrays = cache.arrays(ti); series = arrays['series']
            local_metrics = [Metrics() for _ in hs]
            evaluated = rejected = 0
            for start in range(0, len(arrays['anchors']), batch_size):
                anchors = arrays['anchors'][start:start + batch_size]
                histories = np.stack([series.history(int(i)) for i in anchors])
                targets, eligible = [], []
                for index, i in enumerate(anchors):
                    try:
                        target = series.at(series.t[i] + np.asarray(hs) / 1000)
                    except ValueError:
                        rejected += 1; continue
                    targets.append(target); eligible.append(index)
                # Update real-origin EMA for ALL causal origins, including warm-up/tail.
                synchronize(device); started = time.perf_counter()
                if uses_confidence:
                    raw = predictor.confidence(histories)
                    alpha = compute_temporal_optimized_alpha(raw, [{'file_name': record['key']}] * len(histories),
                                                             alpha_state, predictor.mu, init_mode='alpha')
                if not eligible:
                    continue
                targets = np.stack(targets)
                chosen = np.array(eligible)
                if uses_confidence:
                    result = predictor.forecast_batch(histories[chosen], hs, initial_alpha=alpha[chosen])
                else:
                    result = predictor.forecast_batch(histories[chosen], hs)
                synchronize(device)
                timings.append({'batch_origins': len(chosen), 'elapsed_ms': (time.perf_counter() - started) * 1000})
                pred_mask = scene.mask(result['positions_m'], result['quaternions_wxyz'])
                gt_q = ypr_to_quat(targets[:, :, 3:])
                truth_mask = scene.mask(targets[:, :, :3], gt_q)
                tr = np.linalg.norm(result['positions_m'] - targets[:, :, :3], axis=-1)
                rot = quaternion_error_deg(result['quaternions_wxyz'], gt_q)
                chosen_anchors = anchors[chosen]
                epoch = series.t[chosen_anchors] + record['epoch_start_s']
                for j, horizon in enumerate(hs):
                    counts, ious = metrics[j].add(pred_mask[:, j], truth_mask[:, j], tr[:, j], rot[:, j])
                    local_metrics[j].add(pred_mask[:, j], truth_mask[:, j], tr[:, j], rot[:, j])
                    for i, anchor in enumerate(chosen_anchors):
                        tp, fp, fn, tn = counts[i].tolist()
                        writer.writerow(dict(trajectory=record['key'], session=record['session'],
                                             origin_relative_s=float(series.t[anchor]), origin_epoch_s=epoch[i],
                                             horizon_ms=horizon, target_epoch_s=epoch[i] + horizon / 1000,
                                             alpha_t=float(result['alpha_t'][i]) if uses_confidence else '',
                                             tp=tp, fp=fp, fn=fn, tn=tn,
                                             iou=ious[i], translation_error_m=tr[i, j], rotation_error_deg=rot[i, j]))
                if save_masks:
                    folder = output / 'predictions'; folder.mkdir(exist_ok=True)
                    extras = {k: result[k] for k in ('alpha_t', 'rollout_alpha') if k in result}
                    np.savez_compressed(folder / f"{record['key']}_{start:07d}.npz",
                        origin_epoch_s=epoch, horizons_ms=np.array(hs), positions_m=result['positions_m'],
                        quaternions_wxyz=result['quaternions_wxyz'], target_positions_m=targets[:, :, :3],
                        target_quaternions_wxyz=gt_q, **extras,
                        roi_pred_packed=np.packbits(pred_mask, axis=-1, bitorder='little'),
                        roi_true_packed=np.packbits(truth_mask, axis=-1, bitorder='little'),
                        num_partitions=np.array(len(scene.centers_m)))
                evaluated += len(chosen)
            per_trajectory.append({'trajectory': record['key'], 'session': record['session'], 'evaluated': evaluated,
                                   'rejected_missing_future': rejected,
                                   'metrics': {str(h): m.result() for h, m in zip(hs, local_metrics)} if evaluated else {}})
    method = 'BAF-Pose' if uses_confidence else 'Motion Transformer'
    report = {'method': method + ' recursive rollout + deterministic partition-center ROI',
              'split': split, 'cache_id': cache.manifest['cache_id'], 'scene_id': scene.snapshot()['scene_id'],
              'input': 'history_6dof_only', 'supervision_protocol': cache.manifest['supervision']['id'], 'horizons_ms': hs, 'common_origins_for_all_horizons': True,
              'interpolation': 'predicted position LERP + predicted quaternion SLERP; exact requested milliseconds',
              'metrics': {str(h): m.result() for h, m in zip(hs, metrics)},
              'aggregation': 'micro over all eligible windows/cells; iou_mean is window mean',
              'empty_policy': 'both-empty IoU=1; precision/recall/F1 zero when their denominators are zero',
              'latency': {'kind': ('amortized batched wall time incl. confidence/EMA and full rollout, excludes I/O and ROI mapping'
                                   if uses_confidence else
                                   'amortized batched wall time incl. full rollout, excludes I/O and ROI mapping'),
                          'device': device, 'batch_size_limit': batch_size,
                          'ms_per_origin': sum(t['elapsed_ms'] for t in timings) / sum(t['batch_origins'] for t in timings)},
              'trajectory_results': per_trajectory}
    dump_json(output / 'metrics.json', report)
    rows = [{'horizon_ms': h, **metrics[j].result()} for j, h in enumerate(hs)]
    with (output / 'metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return report
