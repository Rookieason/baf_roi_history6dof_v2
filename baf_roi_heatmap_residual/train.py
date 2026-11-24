"""Train CSI-conditioned latent residuals through the frozen BAF rollout.

BAF parameters remain frozen, but gradients pass through its original pose
heads and every later autoregressive step. No target or cached BAF endpoint
is an input to the residual model.
"""
from collections import defaultdict
from dataclasses import asdict
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from baf_roi.runtime import Predictor
from baf_roi_heatmap.contracts import file_hash, new_directory, seed_all, write_json
from baf_roi_heatmap.data import ForecastDataset
from baf_roi_heatmap.losses import task_loss
from baf_roi_heatmap.training_utils import CheckpointSelection, recording_weights
from roi_experiments.evaluation import roi_counts
from .model import InternalResidualFusion, ResidualConfig
from .runtime import predict_dataset_batch, reference_prediction


@torch.inference_mode()
def validation_iou(model, dataset, loader, device):
    """Hard-geometry mean IoU, equally weighting original recordings/horizons."""
    model.eval()
    by_record = defaultdict(list)
    for batch in loader:
        prediction = predict_dataset_batch(model, dataset, batch, device)
        ris, ois = batch['record_index'].numpy(), batch['origin_index'].numpy()
        # Baseline forecasts serve only the exact online fallback policy.
        # They never enter model.forward or the RF residual features.
        p0 = np.stack([dataset.arrays(int(ri))['base_p'][int(oi)] for ri, oi in zip(ris, ois)])
        q0 = np.stack([dataset.arrays(int(ri))['base_q'][int(oi)] for ri, oi in zip(ris, ois)])
        p, q = reference_prediction(model, prediction, batch['inputs']['rf_available'], p0, q0)
        truth = batch['supervision']['target_mask'].numpy().astype(bool)
        tp, fp, fn, _ = roi_counts(dataset.scene.mask(p, q), truth)
        union = tp+fp+fn
        values = np.divide(tp, union, out=np.ones_like(tp, dtype=float), where=union != 0)
        for ri, value in zip(ris, values):
            by_record[dataset.recording_id(int(ri))].append(float(np.mean(value)))
    score = float(np.mean([np.mean(values) for values in by_record.values()]))
    if not math.isfinite(score):
        raise FloatingPointError('Non-finite internal residual validation IoU.')
    return score


def train(prepared, output, config=None, epochs=20, batch_size=64, lr=1e-3,
          seed=42, device='cpu', roi_weight=.25, *, objective='roi_iou', aggregation='recording'):
    config = (config or ResidualConfig()).validate()
    if (not isinstance(epochs, int) or epochs < 1 or not isinstance(batch_size, int) or batch_size < 1
            or not math.isfinite(lr) or lr <= 0 or not math.isfinite(roi_weight) or roi_weight < 0):
        raise ValueError('Invalid training hyperparameters.')
    if objective not in ('pose_roi', 'roi_iou'):
        raise ValueError('Unknown training objective.')
    seed_all(seed)
    training, validation = ForecastDataset(prepared, 'train'), ForecastDataset(prepared, 'valid')
    weights = recording_weights(training, aggregation)
    if config.modality == 'rf':
        if not training.manifest['normalizer']['sample_count']:
            raise ValueError('There are no valid training RF maps.')
        if not any(bool(training[i]['inputs']['rf_available']) for i in range(len(training))):
            raise ValueError('No training heatmaps align causally; inspect support/clock/delivery settings.')
    checkpoint = training.manifest['baseline_checkpoint']
    if file_hash(checkpoint) != training.manifest['baseline_sha256']:
        raise ValueError('The frozen baseline checkpoint changed after preparation.')
    base = Predictor.load(checkpoint, device)
    model = InternalResidualFusion(base, config).to(device)
    optimizer = torch.optim.AdamW(model.residual.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    valid_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    out = new_directory(output)
    log = []
    selection = CheckpointSelection(out, 'residual.pt')
    for epoch in range(epochs+1):
        loss_value = None
        if epoch:
            model.train()
            total, count = 0., 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = predict_dataset_batch(model, training, batch, device)
                labels = {key: value.to(device) for key, value in batch['supervision'].items()}
                loss = task_loss(prediction, labels, training.scene, roi_weight=roi_weight,
                                 objective=objective, sample_weight=weights[batch['record_index']])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite internal residual training loss.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.residual.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                size = len(batch['record_index'])
                total += float(loss.detach())*size
                count += size
            loss_value = total/count
        score = validation_iou(model, validation, valid_loader, device)
        log.append({'epoch': epoch, 'train_loss': loss_value, 'validation_recording_macro_iou': score})
        print(f'Internal residual epoch={epoch}/{epochs} train_loss={loss_value} validation_IoU={score:.6f}', flush=True)
        bundle = {
            'format_version': 1, 'method': 'baf_roi_heatmap_residual',
            'residual': model.residual.state_dict(), 'model_config': asdict(config),
            'experiment_id': training.manifest['experiment_id'],
            'baseline_sha256': training.manifest['baseline_sha256'],
            'horizons_ms': training.horizons, 'input_config': training.manifest['input_config'],
            'grid_id': training.rf['grid_id'], 'allow_exploratory': training.rf.get('allow_exploratory', False),
            'normalizer': training.manifest['normalizer'], 'selected_epoch': epoch,
            'validation_recording_macro_iou': score, 'seed': seed, 'roi_weight': roi_weight,
            'objective': objective, 'aggregation': aggregation,
            'rf_age_basis': training.rf_age_basis,
            'fusion_site': 'expert_latent_before_original_pose_heads_each_rollout_step',
            'selection': 'validation recording-macro IoU, equally weighted trained horizons; epoch zero eligible',
        }
        status = selection.update(bundle)
        write_json(out/'training.json', {
            'specification': model.specification(), 'history': log,
            'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'seed': seed,
            'objective': objective, 'aggregation': aggregation, **status,
            'warning': 'Software execution is not evidence of real-data CSI improvement.',
        })
    print(f'Internal residual selection: {status}', flush=True)
    return str(out/'residual.pt')
