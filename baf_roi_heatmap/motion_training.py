"""Task-trained internal experts with per-horizon validation regression limits."""
from collections import defaultdict
from dataclasses import asdict
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from baf_roi.pose import quaternion_error_deg
from .contracts import file_hash, new_directory, seed_all, write_json
from .data import ForecastDataset
from .losses import numpy_prediction
from .model import build_forecaster, forward_batch
from .training_utils import CheckpointSelection, recording_weights


DEFAULT_LIMITS = {'coverage_drop': .005, 'translation_increase_m': .005, 'rotation_increase_deg': .5}


def validate_limits(limits=None):
    limits = dict(DEFAULT_LIMITS if limits is None else limits)
    if set(limits) != set(DEFAULT_LIMITS) or any(not math.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError('Validation limits need finite nonnegative coverage/translation/rotation tolerances.')
    return limits


def validation_eligibility(metrics, baseline, limits):
    """Inspect EVERY horizon, so long-horizon IoU cannot hide short-horizon harm."""
    reasons = []
    for horizon, reference in baseline.items():
        value = metrics[horizon]
        if value['coverage_micro'] < reference['coverage_micro']-limits['coverage_drop']-1e-12:
            reasons.append(f'{horizon}ms: coverage')
        if value['translation_mean_m'] > reference['translation_mean_m']+limits['translation_increase_m']+1e-12:
            reasons.append(f'{horizon}ms: translation')
        if value['rotation_mean_deg'] > reference['rotation_mean_deg']+limits['rotation_increase_deg']+1e-12:
            reasons.append(f'{horizon}ms: rotation')
    return not reasons, reasons


@torch.inference_mode()
def validation_metrics(model, dataset, loader, device):
    from roi_experiments.evaluation import roi_counts
    model.eval()
    records = defaultdict(lambda: defaultdict(list))
    totals = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    for batch in loader:
        prediction = forward_batch(model, batch, device)
        ris, ois = batch['record_index'].numpy(), batch['origin_index'].numpy()
        p0 = np.stack([dataset.arrays(int(ri))['base_p'][int(oi)] for ri, oi in zip(ris, ois)])
        q0 = np.stack([dataset.arrays(int(ri))['base_q'][int(oi)] for ri, oi in zip(ris, ois)])
        p, q = numpy_prediction(prediction, p0, q0)
        # Match the common evaluator's original float64 targets and hard geometry.
        target_p = np.stack([dataset.arrays(int(ri))['target_p'][int(oi)] for ri, oi in zip(ris, ois)])
        target_q = np.stack([dataset.arrays(int(ri))['target_q'][int(oi)] for ri, oi in zip(ris, ois)])
        truth = batch['supervision']['target_mask'].numpy().astype(bool)
        tp, fp, fn, _ = roi_counts(dataset.scene.mask(p, q), truth)
        union = tp+fp+fn
        iou = np.divide(tp, union, out=np.ones_like(tp, dtype=float), where=union != 0)
        tr, rot = np.linalg.norm(p-target_p, axis=-1), quaternion_error_deg(q, target_q)
        for j, horizon in enumerate(dataset.horizons):
            totals[str(horizon)] += [tp[:, j].sum(), (tp[:, j]+fn[:, j]).sum()]
            for b, ri in enumerate(ris):
                records[str(horizon)][dataset.recording_id(int(ri))].append((iou[b, j], tr[b, j], rot[b, j]))
    result = {}
    for horizon, groups in records.items():
        macro = np.mean([np.mean(values, axis=0) for values in groups.values()], axis=0)
        tp, required = totals[horizon]
        result[horizon] = {'iou_mean': float(macro[0]), 'translation_mean_m': float(macro[1]),
                           'rotation_mean_deg': float(macro[2]), 'coverage_micro': float(tp/required) if required else 0.}
    return result


def train_motion(prepared, output, config, epochs, batch_size, lr, seed, device, roi_weight,
                 objective, aggregation, validation_limits):
    from .train import train_epoch
    if epochs < 1 or batch_size < 1 or not math.isfinite(lr) or lr <= 0 or not math.isfinite(roi_weight) or roi_weight < 0:
        raise ValueError('Invalid training hyperparameters.')
    if objective not in ('pose_roi', 'roi_iou'):
        raise ValueError('Unknown training objective.')
    limits = validate_limits(validation_limits)
    seed_all(seed)
    training, validation = ForecastDataset(prepared, 'train'), ForecastDataset(prepared, 'valid')
    weights = recording_weights(training, aggregation)
    if config.modality == 'rf' and (not training.manifest['normalizer']['sample_count']
            or not any(bool(training[i]['inputs']['rf_available']) for i in range(len(training)))):
        raise ValueError('No causally available training RF.')
    baseline = training.manifest['baseline_checkpoint']
    if file_hash(baseline) != training.manifest['baseline_sha256']:
        raise ValueError('The baseline checkpoint changed after preparation.')
    model = build_forecaster(config, baseline, device)
    # Preserve pretrained pose heads while new RF/context modules learn. A full
    # long rollout amplifies a small native-head update at every predicted step.
    heads = [p for expert in model.motion_experts
             for head in (expert.branch.translation_head, expert.branch.rotation_head)
             for p in head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in heads}
    fresh = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{'params': fresh, 'lr': lr}]
    if heads:
        groups.append({'params': heads, 'lr': lr*.1})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    valid_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    out = new_directory(output)
    selection, history, baseline_metrics = CheckpointSelection(out, 'adapter.pt'), [], None
    for epoch in range(epochs+1):
        loss = None
        if epoch:
            model.baseline_only.fill_(False)
            loss = train_epoch(model, train_loader, optimizer, training.scene, device, roi_weight, objective, weights)
        metrics = validation_metrics(model, validation, valid_loader, device)
        if epoch == 0:
            baseline_metrics = metrics
        eligible, reasons = validation_eligibility(metrics, baseline_metrics, limits)
        score = float(np.mean([value['iou_mean'] for value in metrics.values()]))
        history.append({'epoch': epoch, 'train_loss': loss, 'validation_recording_macro_iou': score,
                        'selection_eligible': eligible, 'regression_reasons': reasons,
                        'validation_by_horizon': metrics})
        bundle = {'format_version': 1, 'model': model.state_dict(), 'model_config': asdict(config),
                  'experiment_id': training.manifest['experiment_id'], 'baseline_sha256': training.manifest['baseline_sha256'],
                  'horizons_ms': training.horizons, 'input_config': training.manifest['input_config'],
                  'grid_id': training.rf['grid_id'], 'allow_exploratory': training.rf.get('allow_exploratory', False),
                  'normalizer': training.manifest['normalizer'], 'rf_age_basis': training.rf_age_basis,
                  'selected_epoch': epoch, 'validation_recording_macro_iou': score,
                  'objective': objective, 'aggregation': aggregation, 'seed': seed, 'roi_weight': roi_weight,
                  'selection_eligible': eligible, 'validation_regression_reasons': reasons,
                  'validation_by_horizon': metrics, 'baseline_validation_by_horizon': baseline_metrics,
                  'validation_limits': limits,
                  'selection': 'Highest validation recording-macro IoU among epochs satisfying every horizon regression limit; exact BAF epoch zero eligible.'}
        status = selection.update(bundle)
        write_json(out/'training.json', {'specification': model.specification(), 'history': history,
                   'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'native_head_learning_rate': lr*.1, 'seed': seed,
                   'objective': objective, 'aggregation': aggregation, 'validation_limits': limits, **status,
                   'warning': 'Validation limits do not guarantee test non-regression. Extra experts have unverified motion semantics.'})
        print(f'Motion experts epoch={epoch}/{epochs} loss={loss} validation_IoU={score:.6f} eligible={eligible} selected={status["selected_epoch"]}', flush=True)
    return str(out/'adapter.pt')
