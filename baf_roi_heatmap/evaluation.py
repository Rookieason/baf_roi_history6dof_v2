"""Matched ROI metrics, validation-only policies, and paired cluster uncertainty."""
from collections import defaultdict
import csv
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from baf_roi.pose import quaternion_error_deg
from baf_roi.roi import Scene
from .contracts import ModelConfig, file_hash, new_directory, read_json, write_json
from .data import ForecastDataset
from .model import ForecastAdapter
from .losses import numpy_prediction


def to_device(values, device):
    return {key: value.to(device) for key, value in values.items()}


def load_adapter(path, dataset, device='cpu'):
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle.get('format_version') != 1 or bundle.get('experiment_id') != dataset.manifest['experiment_id']:
        raise ValueError('Adapter checkpoint/prepared experiment mismatch.')
    model = ForecastAdapter(ModelConfig(**bundle['model_config'])).to(device)
    model.load_state_dict(bundle['model'], strict=True)
    model.eval()
    return model, bundle


def margin_scene(scene, margin_deg):
    if not np.isfinite(margin_deg) or margin_deg < 0:
        raise ValueError('Prediction-side angular margin must be finite and nonnegative.')
    cfg = scene.cfg.copy()
    keys = ['fov_deg'] if cfg['visibility_mode'] == 'legacy_center_cone' else ['horizontal_fov_deg', 'vertical_fov_deg']
    for key in keys:
        cfg[key] += 2 * margin_deg
    return Scene(scene.centers_m, scene.partition_ids, cfg)


def roi_counts(prediction, truth):
    p, y = np.asarray(prediction, bool), np.asarray(truth, bool)
    if p.shape != y.shape:
        raise ValueError('ROI shapes must match.')
    return (p & y).sum(-1), (p & ~y).sum(-1), (~p & y).sum(-1), (~p & ~y).sum(-1)


def summarize(rows):
    if not rows:
        return None
    tp, fp, fn, tn = (sum(int(r[k]) for r in rows) for k in ('tp', 'fp', 'fn', 'tn'))
    n = len(rows)
    union, required, selected = tp+fp+fn, tp+fn, tp+fp
    nonempty = [r['iou'] for r in rows if r['tp']+r['fn'] > 0]
    return {'n_origins': n, 'n_nonempty_ground_truth': len(nonempty),
            'iou_mean_nonempty_gt': float(np.mean(nonempty)) if nonempty else None, 'iou_mean': float(np.mean([r['iou'] for r in rows])),
            'iou_micro': tp/union if union else 1., 'f1_micro': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'precision_micro': tp/selected if selected else 0., 'coverage_micro': tp/required if required else None,
            'fn_mean': fn/n, 'fp_mean': fp/n, 'selected_mean': selected/n, 'required_mean': required/n,
            'exact_match': float(np.mean([r['fp']+r['fn'] == 0 for r in rows])),
            'empty_ground_truth_fraction': float(np.mean([r['tp']+r['fn'] == 0 for r in rows])),
            'translation_mean_m': float(np.mean([r['translation_error_m'] for r in rows])),
            'rotation_mean_deg': float(np.mean([r['rotation_error_deg'] for r in rows])),
            'translation_p95_m': float(np.quantile([r['translation_error_m'] for r in rows], .95)),
            'rotation_p95_deg': float(np.quantile([r['rotation_error_deg'] for r in rows], .95))}


def balanced_summary(rows):
    if not rows:
        return None
    by_record = defaultdict(list)
    for row in rows:
        by_record[row['record']].append(row)
    record_metrics = {key: summarize(value) for key, value in by_record.items()}
    fields = ('iou_mean_nonempty_gt', 'iou_mean', 'f1_micro', 'coverage_micro', 'fp_mean', 'fn_mean', 'selected_mean', 'translation_mean_m', 'rotation_mean_deg')
    def mean_metrics(values):
        return {key: float(np.mean([v[key] for v in values if v[key] is not None]))
                if any(v[key] is not None for v in values) else None for key in fields}
    result = {'pooled': summarize(rows), 'recording_macro': mean_metrics(list(record_metrics.values())),
              'recordings': len(record_metrics), 'scenario_macro': None}
    if all(r['scenario'] != 'unspecified' for r in rows):
        scenarios = defaultdict(list)
        for key, values in by_record.items():
            scenarios[values[0]['scenario']].append(record_metrics[key])
        result['scenario_macro'] = mean_metrics([mean_metrics(values) for values in scenarios.values()])
    return result


@torch.inference_mode()
def evaluate(prepared, output, checkpoint=None, baseline='baf', split='test', device='cpu', batch_size=64,
             margins=(0.,), delivery_delay_s=0., policy_path=None, motion_transformer_checkpoint=None):
    if split not in ('valid', 'test') or batch_size < 1:
        raise ValueError('Evaluation uses valid/test and a positive batch size.')
    dataset = ForecastDataset(prepared, split, delivery_delay_s)
    if checkpoint and motion_transformer_checkpoint:
        raise ValueError('Choose an adapter or Motion Transformer, not both.')
    transformer = None
    if motion_transformer_checkpoint:
        from motion_transformer_roi.runtime import Predictor as MotionPredictor
        transformer = MotionPredictor.load(motion_transformer_checkpoint, device)
        if transformer.bundle['cache_id'] != dataset.manifest['hmd_cache_id'] or transformer.stats != dataset.cache.stats:
            raise ValueError('Motion Transformer must use exactly the shared HMD cache and normalization.')
    model, bundle = load_adapter(checkpoint, dataset, device) if checkpoint else (None, None)
    if not checkpoint and baseline not in ('baf', 'focused', 'explorer', 'fixed'):
        raise ValueError('Unknown baseline.')
    method_id = file_hash(checkpoint or motion_transformer_checkpoint) if checkpoint or motion_transformer_checkpoint else 'frozen:' + baseline
    policy = read_json(policy_path) if policy_path else None
    if policy and (policy['experiment_id'] != dataset.manifest['experiment_id'] or policy['method_id'] != method_id
                   or policy['selected_on_split'] != 'valid'):
        raise ValueError('Policy must be selected on validation for this method and experiment.')
    if policy and any(v is None for v in policy['margins_by_horizon'].values()):
        raise ValueError('Validation coverage was infeasible for at least one horizon; do not silently relax it on test.')
    if not margins or len(set(margins)) != len(margins):
        raise ValueError('Supply distinct prediction margins.')
    margin_values = sorted(set(policy['margins_by_horizon'].values())) if policy else list(margins)
    scenes = {float(m): margin_scene(dataset.scene, float(m)) for m in margin_values}
    out = new_directory(output)
    rows = []
    previous = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        ris, ois = batch['record_index'].numpy(), batch['origin_index'].numpy()
        original_p = np.stack([dataset.arrays(int(ri))['base_p'][int(oi)] for ri, oi in zip(ris, ois)])
        original_q = np.stack([dataset.arrays(int(ri))['base_q'][int(oi)] for ri, oi in zip(ris, ois)])
        if transformer:
            histories = []
            for ri, oi in zip(ris, ois):
                record = dataset.records[int(ri)]
                raw = dataset.cache.arrays(record['cache_index'])
                anchor = int(dataset.arrays(int(ri))['anchors'][int(oi)])
                histories.append(raw['series'].history(anchor))
            prediction = transformer.forecast_batch(np.stack(histories), dataset.horizons)
            p, q = prediction['positions_m'], prediction['quaternions_wxyz']
            gates = None  # A transformer has no expert mixing weight.
        elif model:
            prediction = model(to_device(batch['inputs'], device))
            p, q = numpy_prediction(prediction, original_p, original_q)
            gates = prediction['gate'].cpu().numpy()
        else:
            prefix = 'base' if baseline == 'baf' else baseline
            p = np.stack([dataset.arrays(int(ri))[prefix+'_p'][int(oi)] for ri, oi in zip(ris, ois)])
            q = np.stack([dataset.arrays(int(ri))[prefix+'_q'][int(oi)] for ri, oi in zip(ris, ois)])
            gates = np.broadcast_to(batch['inputs']['alpha'].numpy()[:, None], p.shape[:2]) if baseline == 'baf' else np.full(p.shape[:2], {'focused': 0., 'explorer': 1., 'fixed': .5}[baseline])
        target_p = np.stack([dataset.arrays(int(ri))['target_p'][int(oi)] for ri, oi in zip(ris, ois)])
        target_q = np.stack([dataset.arrays(int(ri))['target_q'][int(oi)] for ri, oi in zip(ris, ois)])
        truth = np.stack([dataset.arrays(int(ri))['target_mask'][int(oi)] for ri, oi in zip(ris, ois)])
        tr = np.linalg.norm(p-target_p, axis=-1)
        rot = quaternion_error_deg(q, target_q)
        for margin, scene in scenes.items():
            pred = scene.mask(p, q)
            tp, fp, fn, tn = roi_counts(pred, truth)
            union = tp+fp+fn
            iou = np.divide(tp, union, out=np.ones_like(tp, dtype=float), where=union != 0)
            for b, (ri, oi) in enumerate(zip(ris, ois)):
                record, a = dataset.records[int(ri)], dataset.arrays(int(ri))
                origin = float(a['origin_s'][int(oi)])
                for j, horizon in enumerate(dataset.horizons):
                    if policy and margin != policy['margins_by_horizon'][str(horizon)]:
                        continue
                    key = (record['key'], horizon, margin)
                    state = previous.get(key)
                    transition = None
                    if state and 0 < origin-state[0] <= dataset.cache.manifest['config']['data']['max_gap_ms']/1000 + 1e-6:
                        transition = float(np.mean((pred[b, j] ^ state[1]) != (truth[b, j] ^ state[2])))
                    previous[key] = (origin, pred[b, j].copy(), truth[b, j].copy())
                    rows.append({'record': record['key'], 'group_id': record['group_id'], 'scenario': record['scenario'],
                                 'origin_id': record['key']+':'+str(int(a['anchors'][int(oi)])), 'origin_s': origin,
                                 'horizon_ms': horizon, 'margin_deg': margin,
                                 'rf_available': int(batch['inputs']['rf_available'][b]), 'gate': float(gates[b, j]) if gates is not None else None,
                                 'tp': int(tp[b, j]), 'fp': int(fp[b, j]), 'fn': int(fn[b, j]), 'tn': int(tn[b, j]),
                                 'iou': float(iou[b, j]), 'translation_error_m': float(tr[b, j]),
                                 'rotation_error_deg': float(rot[b, j]), 'transition_error': transition})
    with (out/'per_origin.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['horizon_ms']}ms_margin{row['margin_deg']:g}"].append(row)
    summaries = {key: {'all': balanced_summary(values),
                       'rf_available': balanced_summary([r for r in values if r['rf_available']]),
                       'transition_error_mean': float(np.mean([r['transition_error'] for r in values if r['transition_error'] is not None]))
                       if any(r['transition_error'] is not None for r in values) else None} for key, values in groups.items()}
    report = {'experiment_id': dataset.manifest['experiment_id'], 'scene_id': dataset.manifest['scene']['scene_id'],
              'method_id': method_id, 'method': bundle['model_config'] if bundle else ('motion_transformer' if transformer else baseline), 'split': split,
              'horizons_ms': dataset.horizons, 'margins_deg': margin_values, 'delivery_delay_s': delivery_delay_s,
              'policy': policy, 'summaries': summaries, 'rows_sha256': file_hash(out/'per_origin.csv'),
              'empty_policy': 'Both-empty IoU=1; zero-denominator F1/precision=0; coverage undefined when no required cells.',
              'evidence': 'Geometric ROI proxy only. No rendered-quality, bitrate or realtime claim.',
              'rf_physical_axes_verified': bool(dataset.rf['grid_id']) and all(r['physical_axes_verified'] for r in dataset.rf['records'] if r['file'])}
    write_json(out/'report.json', report)
    return report


def read_rows(directory):
    with (Path(directory)/'per_origin.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ints = ('tp', 'fp', 'fn', 'tn', 'rf_available')
    floats = ('origin_s', 'horizon_ms', 'margin_deg', 'iou', 'translation_error_m', 'rotation_error_deg')
    for r in rows:
        for k in ints: r[k] = int(r[k])
        for k in floats: r[k] = float(r[k])
        r['gate'] = float(r['gate']) if r['gate'] else None
        r['transition_error'] = float(r['transition_error']) if r['transition_error'] else None
    return rows


def select_policy(validation_directory, output_path, coverage=.95):
    report = read_json(Path(validation_directory)/'report.json')
    if report['split'] != 'valid' or not 0 < coverage <= 1:
        raise ValueError('Policy selection requires validation data and coverage in (0,1].')
    if file_hash(Path(validation_directory)/'per_origin.csv') != report['rows_sha256']:
        raise ValueError('Validation rows were modified.')
    selected = {}
    for h in report['horizons_ms']:
        candidates = []
        for margin in report['margins_deg']:
            summary = report['summaries'][f'{h}ms_margin{margin:g}']['all']['pooled']
            value = summary['coverage_micro']
            if value is not None and value >= coverage:
                candidates.append((summary['selected_mean'], margin))
        selected[str(h)] = min(candidates)[1] if candidates else None
    policy = {'experiment_id': report['experiment_id'], 'method_id': report['method_id'], 'selected_on_split': 'valid',
              'coverage_constraint': coverage, 'coverage_aggregation': 'micro on validation',
              'margins_by_horizon': selected, 'validation_rows_sha256': report['rows_sha256'],
              'note': 'No guarantee test coverage reaches this level; report achieved test coverage.'}
    if Path(output_path).exists(): raise ValueError('Policy output exists.')
    write_json(output_path, policy)
    return policy


def compare(left_directory, right_directory, output_path, bootstrap_samples=2000, seed=42):
    """right-minus-left; cluster resampling, never resample overlapping windows independently."""
    if bootstrap_samples < 100:
        raise ValueError('Use at least 100 bootstrap resamples.')
    reports = [read_json(Path(d)/'report.json') for d in (left_directory, right_directory)]
    for key in ('experiment_id', 'scene_id', 'split', 'horizons_ms'):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f'Unmatched comparison: {key}.')
    rows = [read_rows(d) for d in (left_directory, right_directory)]
    for d, report in zip((left_directory, right_directory), reports):
        if file_hash(Path(d)/'per_origin.csv') != report['rows_sha256']:
            raise ValueError('Evaluation rows were modified.')
    def indexed(values):
        result = {}
        for r in values:
            key = (r['origin_id'], r['horizon_ms'])
            if key in result: raise ValueError('Compare one policy/margin per horizon, not an entire margin sweep.')
            result[key] = r
        return result
    a, b = map(indexed, rows)
    if a.keys() != b.keys(): raise ValueError('Origin sets differ; refusing silent intersection.')
    for key in a:
        if a[key]['tp']+a[key]['fn'] != b[key]['tp']+b[key]['fn'] or a[key]['group_id'] != b[key]['group_id']:
            raise ValueError('Targets or recording groups differ.')
    rng = np.random.default_rng(seed)
    result = []
    for h in reports[0]['horizons_ms']:
        for subset in ('all', 'common_rf_available'):
            keys = [k for k in a if k[1] == h and (subset == 'all' or a[k]['rf_available'] and b[k]['rf_available'])]
            by_record = defaultdict(list)
            for k in keys: by_record[a[k]['record']].append(k)
            records = [(a[ks[0]]['group_id'], float(np.mean([b[k]['iou']-a[k]['iou'] for k in ks]))) for ks in by_record.values()]
            groups = sorted({r[0] for r in records})
            observed = float(np.mean([r[1] for r in records])) if records else None
            interval = None
            if len(groups) >= 2:
                group_values = {g: [v for group, v in records if group == g] for g in groups}
                draws = []
                for _ in range(bootstrap_samples):
                    picked = rng.choice(groups, len(groups), replace=True)
                    draws.append(np.mean([v for g in picked for v in group_values[g]]))
                interval = np.quantile(draws, [.025, .975]).tolist()
            result.append({'horizon_ms': h, 'subset': subset, 'recording_macro_delta_iou': observed,
                           'cluster_percentile_ci95': interval, 'clusters': len(groups), 'recordings': len(records),
                           'origins': len(keys)})
    output = {'left_method': reports[0]['method_id'], 'right_method': reports[1]['method_id'], 'results': result,
              'bootstrap_samples': bootstrap_samples, 'seed': seed,
              'interpretation': 'Descriptive unadjusted per-horizon intervals. Pre-register primary comparisons; not blanket significance across horizons.',
              'estimand': 'Recording-macro delta IoU; entire subject/session groups resampled together.'}
    if Path(output_path).exists(): raise ValueError('Comparison output exists.')
    write_json(output_path, output)
    return output
