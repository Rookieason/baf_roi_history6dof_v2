"""Matched ROI metrics, validation-only policies, and paired cluster uncertainty."""
from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from baf_roi.pose import quaternion_error_deg
from baf_roi.roi import Scene
from baf_roi_heatmap.contracts import ModelConfig, digest, file_hash, new_directory, read_json, write_json
from baf_roi_heatmap.data import ForecastDataset
from baf_roi_heatmap.model import ForecastAdapter
from baf_roi_heatmap.losses import numpy_prediction


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
    nonempty_union = [r['iou'] for r in rows if r['tp']+r['fp']+r['fn'] > 0]
    return {'n_origins': n, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'n_nonempty_ground_truth': len(nonempty),
            'iou_mean_nonempty_gt': float(np.mean(nonempty)) if nonempty else None, 'iou_mean': float(np.mean([r['iou'] for r in rows])),
            'iou_mean_nonempty_union': float(np.mean(nonempty_union)) if nonempty_union else None,
            'iou_micro': tp/union if union else 1., 'f1_micro': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'precision_micro': tp/selected if selected else 0.,
            'recall_micro': tp/required if required else 0., 'coverage_micro': tp/required if required else 0.,
            'coverage_defined': bool(required),
            'per_cell_accuracy': (tp+tn)/(tp+fp+fn+tn) if tp+fp+fn+tn else 0.,
            'fn_mean': fn/n, 'fp_mean': fp/n, 'selected_mean': selected/n, 'required_mean': required/n,
            'exact_match': float(np.mean([r['fp']+r['fn'] == 0 for r in rows])),
            'ground_truth_empty_count': n-len(nonempty), 'both_empty_count': n-len(nonempty_union),
            'empty_ground_truth_fraction': float(np.mean([r['tp']+r['fn'] == 0 for r in rows])),
            'translation_mean_m': float(np.mean([r['translation_error_m'] for r in rows])),
            'rotation_mean_deg': float(np.mean([r['rotation_error_deg'] for r in rows])),
            'translation_p95_m': float(np.quantile([r['translation_error_m'] for r in rows], .95)),
            'rotation_p95_deg': float(np.quantile([r['rotation_error_deg'] for r in rows], .95)),
            'translation_accuracy_at_0_02m': float(np.mean([r['translation_error_m'] <= .02 for r in rows])),
            'rotation_accuracy_at_5deg': float(np.mean([r['rotation_error_deg'] <= 5 for r in rows]))}


def balanced_summary(rows):
    if not rows:
        return None
    by_record = defaultdict(list)
    for row in rows:
        by_record[row.get('recording_id', row['record'])].append(row)
    record_metrics = {key: summarize(value) for key, value in by_record.items()}
    fields = ('iou_mean_nonempty_gt', 'iou_mean_nonempty_union', 'iou_mean', 'f1_micro', 'precision_micro',
              'recall_micro', 'coverage_micro', 'fp_mean', 'fn_mean', 'selected_mean', 'translation_mean_m', 'rotation_mean_deg')
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


def recording_id(dataset, record):
    """All gap segments from one source recording have one aggregation weight."""
    if record.get('recording_id'):
        return record['recording_id']
    source = dataset.cache.records[record['cache_index']]
    return f"{source['session']}:{source.get('source_sha256', source.get('source', source['key']))}"


def evaluation_protocol(dataset):
    """Model-independent identity for geometry, data, timing and aggregation."""
    return {'version': 2, 'hmd_cache_id': dataset.manifest['hmd_cache_id'], 'split': dataset.split,
            'horizons_ms': [float(h) for h in dataset.horizons], 'scene_id': dataset.scene.snapshot()['scene_id'],
            'origins': 'history-valid cache anchors with targets at every requested horizon',
            'history_frames': 30, 'sample_rate_hz': 30, 'targets': 'exact-time interpolation',
            'observed_alpha': 'chronological EMA, reset on max-gap, never commit forecast state',
            'primary_metrics': ['iou_mean', 'f1_micro'], 'primary_aggregation': 'pooled',
            'empty_policy': 'both-empty IoU=1; zero-denominator precision/recall/coverage/F1=0'}


@torch.inference_mode()
def evaluate(prepared, output, checkpoint=None, baseline='baf', split='test', device='cpu', batch_size=64,
             margins=(0.,), delivery_delay_s=0., policy_path=None, motion_transformer_checkpoint=None,
             alpha_checkpoint=None, *, residual_checkpoint=None, _dataset=None, _pose_predictor=None, _pose_checkpoint=None):
    if split not in ('valid', 'test') or batch_size < 1:
        raise ValueError('Evaluation uses valid/test and a positive batch size.')
    dataset = _dataset if _dataset is not None else ForecastDataset(prepared, split, delivery_delay_s)
    if sum(bool(p) for p in (checkpoint, motion_transformer_checkpoint, alpha_checkpoint, residual_checkpoint, _pose_checkpoint)) > 1:
        raise ValueError('Choose exactly one prediction source.')
    transformer = None
    if motion_transformer_checkpoint:
        from motion_transformer_roi.runtime import Predictor as MotionPredictor
        transformer = MotionPredictor.load(motion_transformer_checkpoint, device)
        if transformer.bundle['cache_id'] != dataset.manifest['hmd_cache_id'] or transformer.stats != dataset.cache.stats:
            raise ValueError('Motion Transformer must use exactly the shared HMD cache and normalization.')
    model, bundle = load_adapter(checkpoint, dataset, device) if checkpoint else (None, None)
    alpha_model = None
    if alpha_checkpoint:
        from baf_roi_heatmap_alpha.runtime import load_alpha
        alpha_model, bundle = load_alpha(alpha_checkpoint, dataset, device)
    residual_model = None
    if residual_checkpoint:
        from baf_roi_heatmap_residual.runtime import load_residual
        residual_model, bundle = load_residual(residual_checkpoint, dataset, device)
    if not checkpoint and baseline not in ('baf', 'focused', 'explorer', 'fixed'):
        raise ValueError('Unknown baseline.')
    source_checkpoint = checkpoint or motion_transformer_checkpoint or alpha_checkpoint or residual_checkpoint or _pose_checkpoint
    method_id = file_hash(source_checkpoint) if source_checkpoint else 'frozen:' + baseline
    policy = read_json(policy_path) if policy_path else None
    if policy and (policy['experiment_id'] != dataset.manifest['experiment_id'] or policy['method_id'] != method_id
                   or policy['selected_on_split'] != 'valid'):
        raise ValueError('Policy must be selected on validation for this method and experiment.')
    if policy and policy.get('delivery_delay_s', 0.) != delivery_delay_s:
        raise ValueError('Policy delivery delay differs from evaluation; calibrate the same sensor protocol.')
    if policy and any(v is None for v in policy['margins_by_horizon'].values()):
        raise ValueError('Validation coverage was infeasible for at least one horizon; do not silently relax it on test.')
    if not margins or len(set(margins)) != len(margins):
        raise ValueError('Supply distinct prediction margins.')
    margin_values = sorted(set(policy['margins_by_horizon'].values())) if policy else list(margins)
    scenes = {float(m): margin_scene(dataset.scene, float(m)) for m in margin_values}
    out = new_directory(output)
    rows = []
    origins_hash, targets_hash, rf_hash = (hashlib.sha256() for _ in range(3))
    previous = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        ris, ois = batch['record_index'].numpy(), batch['origin_index'].numpy()
        applied = np.zeros((len(ris), len(dataset.horizons)), dtype=bool)
        if transformer is not None or _pose_predictor is not None:
            histories = []
            for ri, oi in zip(ris, ois):
                record = dataset.records[int(ri)]
                raw = dataset.cache.arrays(record['cache_index'])
                anchor = int(dataset.arrays(int(ri))['anchors'][int(oi)])
                histories.append(raw['series'].history(anchor))
            predictor = transformer if transformer is not None else _pose_predictor
            uses_alpha = callable(getattr(predictor, 'confidence', None))
            extras = {'initial_alpha': batch['inputs']['alpha'].numpy()} if uses_alpha else {}
            prediction = predictor.forecast_batch(np.stack(histories), dataset.horizons, **extras)
            p, q = prediction['positions_m'], prediction['quaternions_wxyz']
            gates = np.broadcast_to(prediction['alpha_t'][:, None], p.shape[:2]) if uses_alpha else None
        elif model is not None or alpha_model is not None or residual_model is not None:
            original_p = np.stack([dataset.arrays(int(ri))['base_p'][int(oi)] for ri, oi in zip(ris, ois)])
            original_q = np.stack([dataset.arrays(int(ri))['base_q'][int(oi)] for ri, oi in zip(ris, ois)])
            if residual_model is not None:
                from baf_roi_heatmap_residual.runtime import predict_dataset_batch
                prediction = predict_dataset_batch(residual_model, dataset, batch, device)
            elif alpha_model is not None:
                from baf_roi_heatmap_alpha.runtime import predict_dataset_batch
                prediction = predict_dataset_batch(alpha_model, dataset, batch, device)
            else:
                prediction = model(to_device(batch['inputs'], device))
            if residual_model is not None:
                from baf_roi_heatmap_residual.runtime import reference_prediction
                p, q = reference_prediction(residual_model, prediction,
                                            batch['inputs']['rf_available'], original_p, original_q)
            else:
                p, q = numpy_prediction(prediction, original_p, original_q)
            gates = prediction['gate'].cpu().numpy()
            applied = prediction['applied'].cpu().numpy().astype(bool)
        else:
            prefix = 'base' if baseline == 'baf' else baseline
            p = np.stack([dataset.arrays(int(ri))[prefix+'_p'][int(oi)] for ri, oi in zip(ris, ois)])
            q = np.stack([dataset.arrays(int(ri))[prefix+'_q'][int(oi)] for ri, oi in zip(ris, ois)])
            gates = np.broadcast_to(batch['inputs']['alpha'].numpy()[:, None], p.shape[:2]) if baseline == 'baf' else np.full(p.shape[:2], {'focused': 0., 'explorer': 1., 'fixed': .5}[baseline])
        target_p = np.stack([dataset.arrays(int(ri))['target_p'][int(oi)] for ri, oi in zip(ris, ois)])
        target_q = np.stack([dataset.arrays(int(ri))['target_q'][int(oi)] for ri, oi in zip(ris, ois)])
        truth = np.stack([dataset.arrays(int(ri))['target_mask'][int(oi)] for ri, oi in zip(ris, ois)])
        for b, (ri, oi) in enumerate(zip(ris, ois)):
            record, a = dataset.records[int(ri)], dataset.arrays(int(ri))
            identity = [record['key'], int(a['anchors'][int(oi)]), float(a['origin_s'][int(oi)]),
                        [float(h) for h in dataset.horizons]]
            origins_hash.update((digest(identity)+'\n').encode())
            for values in (target_p[b], target_q[b]):
                targets_hash.update(np.ascontiguousarray(values, dtype='<f8').tobytes())
            targets_hash.update(np.ascontiguousarray(truth[b], dtype=np.uint8).tobytes())
            rf_hash.update(bytes([int(batch['inputs']['rf_available'][b])]))
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
                    rows.append({'record': record['key'], 'recording_id': recording_id(dataset, record),
                                 'group_id': record['group_id'], 'scenario': record['scenario'],
                                 'origin_id': record['key']+':'+str(int(a['anchors'][int(oi)])), 'origin_s': origin,
                                 'horizon_ms': horizon, 'margin_deg': margin,
                                 'rf_available': int(batch['inputs']['rf_available'][b]), 'gate': float(gates[b, j]) if gates is not None else None,
                                 'correction_applied': int(applied[b, j]),
                                 'gate_kind': ('none' if gates is None else
                                               'rollout_upper_step' if (alpha_model is not None or residual_model is not None) and applied[b, j] else
                                               'terminal_expert_mix' if model is not None and bundle['model_config']['method'] == 'expert_gate' and applied[b, j] else
                                               'fixed_rollout_weight' if model is None and alpha_model is None and residual_model is None and _pose_predictor is None and baseline in ('focused', 'explorer', 'fixed') else
                                               'observed_origin_alpha'),
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
    protocol = evaluation_protocol(dataset)
    method = ({'method': 'baf_roi_heatmap_residual', **bundle['model_config']} if residual_model is not None else
              {'method': 'baf_roi_heatmap_alpha', **bundle['model_config']} if alpha_model is not None else
              bundle.get('model_config', {}) if bundle else
              ('motion_transformer' if transformer is not None or (_pose_predictor is not None and not callable(getattr(_pose_predictor, 'confidence', None))) else baseline))
    report = {'experiment_id': dataset.manifest['experiment_id'], 'scene_id': dataset.manifest['scene']['scene_id'],
              'protocol': protocol, 'protocol_id': digest(protocol),
              'origin_keys_sha256': origins_hash.hexdigest(), 'targets_sha256': targets_hash.hexdigest(),
              'rf_cohort_sha256': rf_hash.hexdigest(),
              'method_id': method_id, 'method': method, 'split': split,
              'horizons_ms': dataset.horizons, 'margins_deg': margin_values, 'delivery_delay_s': delivery_delay_s,
              'policy': policy, 'summaries': summaries, 'rows_sha256': file_hash(out/'per_origin.csv'),
              'empty_policy': protocol['empty_policy'],
              'primary_aggregation': 'pooled; iou_mean is per-origin mean, f1_micro uses summed counts',
              'macro_aggregation': 'recording_macro averages original source recordings; scenario_macro then averages scenarios',
              'gate_diagnostic': 'gate_kind distinguishes observed-origin alpha, applied forecast alpha, terminal mixture, and fixed weights; correction_applied identifies fallback rows.',
              'rendered_quality': {'status': 'not_evaluated', 'ssim': None, 'q_min': None,
                                   'reason': 'Requires matched rendered frames at the actual future pose; geometric coverage is not SSIM.'},
              'evidence': 'Geometric ROI proxy only. No rendered-quality, bitrate or realtime claim.',
              'rf_physical_axes_verified': bool(dataset.rf['grid_id']) and all(r['physical_axes_verified'] for r in dataset.rf['records'] if r['file'])}
    report['checkpoint_selection'] = ({
        'selected_epoch': bundle.get('selected_epoch'),
        'status': 'baseline_retained' if bundle.get('selected_epoch') == 0 else 'trained_correction_selected',
        'validation_recording_macro_iou': bundle.get('validation_recording_macro_iou'),
        'objective': bundle.get('objective', 'pose_roi'),
        'aggregation': bundle.get('aggregation', 'origin'),
    } if bundle else None)
    report['correction_applied_fraction'] = float(np.mean([r['correction_applied'] for r in rows]))
    if residual_model is not None:
        report['architecture'] = residual_model.specification()
    write_json(out/'report.json', report)
    return report


def evaluate_pose(cache, checkpoint, output, method='baf_roi', device='cpu', split='test', batch_size=64,
                  horizons_ms=None, scene_config=None, margins=(0.,)):
    """HMD-only methods use this same engine without requiring an RF or BAF artifact."""
    from .pose_evaluation import PoseEvaluationDataset
    if method == 'baf_roi':
        from baf_roi.runtime import Predictor
    elif method == 'motion_transformer_roi':
        from motion_transformer_roi.runtime import Predictor
    else:
        raise ValueError('Pose-only evaluation method must be baf_roi or motion_transformer_roi.')
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    predictor = Predictor.load(checkpoint, device)
    dataset = PoseEvaluationDataset(cache, predictor, split, horizons_ms, scene_config, batch_size)
    return evaluate(None, output, split=split, device=device, batch_size=batch_size, margins=margins,
                    _dataset=dataset, _pose_predictor=predictor, _pose_checkpoint=checkpoint)


def read_rows(directory):
    with (Path(directory)/'per_origin.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ints = ('tp', 'fp', 'fn', 'tn', 'rf_available')
    floats = ('origin_s', 'horizon_ms', 'margin_deg', 'iou', 'translation_error_m', 'rotation_error_deg')
    for r in rows:
        for k in ints: r[k] = int(r[k])
        if 'correction_applied' in r: r['correction_applied'] = int(r['correction_applied'])
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
            if summary.get('coverage_defined', summary['required_mean'] > 0) and value is not None and value >= coverage:
                candidates.append((summary['selected_mean'], margin))
        selected[str(h)] = min(candidates)[1] if candidates else None
    policy = {'experiment_id': report['experiment_id'], 'method_id': report['method_id'], 'selected_on_split': 'valid',
              'delivery_delay_s': report.get('delivery_delay_s', 0.), 'scene_id': report['scene_id'],
              'evaluation_protocol_version': report.get('protocol', {}).get('version', 1),
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
    if reports[0].get('delivery_delay_s', 0.) != reports[1].get('delivery_delay_s', 0.):
        raise ValueError('Unmatched comparison: delivery_delay_s. Evaluate all methods at the same sensor delay.')
    for key in ('protocol_id', 'origin_keys_sha256', 'targets_sha256', 'rf_cohort_sha256'):
        if key in reports[0] or key in reports[1]:
            if reports[0].get(key) != reports[1].get(key):
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
        if a[key]['rf_available'] != b[key]['rf_available']:
            raise ValueError('RF availability differs; evaluate every method on the same sensor protocol.')
    rng = np.random.default_rng(seed)
    result = []
    for h in reports[0]['horizons_ms']:
        for subset in ('all', 'common_rf_available'):
            keys = [k for k in a if k[1] == h and (subset == 'all' or a[k]['rf_available'] and b[k]['rf_available'])]
            by_record = defaultdict(list)
            for k in keys: by_record[a[k].get('recording_id', a[k]['record'])].append(k)
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
                           'pooled_delta_iou': float(np.mean([b[k]['iou']-a[k]['iou'] for k in keys])) if keys else None,
                           'cluster_percentile_ci95': interval, 'clusters': len(groups), 'recordings': len(records),
                           'origins': len(keys)})
    output = {'left_method': reports[0]['method_id'], 'right_method': reports[1]['method_id'], 'results': result,
              'bootstrap_samples': bootstrap_samples, 'seed': seed,
              'interpretation': 'Descriptive unadjusted per-horizon intervals. Pre-register primary comparisons; not blanket significance across horizons.',
              'estimand': 'Recording-macro delta IoU; entire subject/session groups resampled together.'}
    if Path(output_path).exists(): raise ValueError('Comparison output exists.')
    write_json(output_path, output)
    return output
