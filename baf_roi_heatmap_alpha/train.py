"""Train only the eight-feature heatmap alpha gate; all BAF weights stay frozen."""
from collections import defaultdict
from dataclasses import asdict
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from baf_roi.runtime import Predictor
from baf_roi_heatmap.contracts import file_hash, new_directory, seed_all, write_json
from baf_roi_heatmap.data import ForecastDataset
from baf_roi_heatmap.evaluation import roi_counts
from baf_roi_heatmap.losses import numpy_prediction, task_loss
from .model import AlphaConfig, AlphaFusion
from .runtime import predict_dataset_batch


@torch.inference_mode()
def validation_iou(model, dataset, loader, device):
    model.eval()
    by_record = defaultdict(list)
    for batch in loader:
        prediction = predict_dataset_batch(model, dataset, batch, device)
        ris, ois = batch['record_index'].numpy(), batch['origin_index'].numpy()
        p0 = np.stack([dataset.arrays(int(ri))['base_p'][int(oi)] for ri, oi in zip(ris, ois)])
        q0 = np.stack([dataset.arrays(int(ri))['base_q'][int(oi)] for ri, oi in zip(ris, ois)])
        p, q = numpy_prediction(prediction, p0, q0)
        truth = batch['supervision']['target_mask'].numpy().astype(bool)
        tp, fp, fn, _ = roi_counts(dataset.scene.mask(p, q), truth)
        union = tp+fp+fn
        values = np.divide(tp, union, out=np.ones_like(tp, dtype=float), where=union != 0)
        for ri, value in zip(ris, values):
            by_record[dataset.recording_id(int(ri))].append(float(np.mean(value)))
    return float(np.mean([np.mean(values) for values in by_record.values()]))


def train(prepared, output, config=None, epochs=20, batch_size=64, lr=1e-3,
          seed=42, device='cpu', roi_weight=.25):
    config = (config or AlphaConfig()).validate()
    if epochs < 1 or batch_size < 1 or not math.isfinite(lr) or lr <= 0 or not math.isfinite(roi_weight) or roi_weight < 0:
        raise ValueError('Invalid training hyperparameters.')
    seed_all(seed)
    training, validation = ForecastDataset(prepared, 'train'), ForecastDataset(prepared, 'valid')
    if not training.manifest['normalizer']['sample_count']:
        raise ValueError('There are no valid training RF maps.')
    if not any(bool(training[i]['inputs']['rf_available']) for i in range(len(training))):
        raise ValueError('No training heatmaps align causally; inspect support/clock/delivery settings.')
    checkpoint = training.manifest['baseline_checkpoint']
    if file_hash(checkpoint) != training.manifest['baseline_sha256']:
        raise ValueError('The frozen baseline checkpoint changed after preparation.')
    base = Predictor.load(checkpoint, device)
    model = AlphaFusion(base, config).to(device)
    optimizer = torch.optim.AdamW(model.gate.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    valid_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    out = new_directory(output)
    best, log = -float('inf'), []
    for epoch in range(epochs+1):
        loss_value = None
        if epoch:
            model.train()
            total, count = 0., 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = predict_dataset_batch(model, training, batch, device)
                labels = {key: value.to(device) for key, value in batch['supervision'].items()}
                loss = task_loss(prediction, labels, training.scene, roi_weight=roi_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite alpha training loss.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                size = len(batch['record_index'])
                total += float(loss.detach())*size; count += size
            loss_value = total/count
        score = validation_iou(model, validation, valid_loader, device)
        log.append({'epoch': epoch, 'train_loss': loss_value, 'validation_recording_macro_iou': score})
        print(f'Alpha-only epoch={epoch}/{epochs} train_loss={loss_value} validation_IoU={score:.6f}', flush=True)
        if score > best+1e-12:
            best = score
            bundle = {'format_version': 1, 'method': 'baf_roi_heatmap_alpha',
                      'gate': model.gate.state_dict(), 'model_config': asdict(config),
                      'experiment_id': training.manifest['experiment_id'],
                      'baseline_sha256': training.manifest['baseline_sha256'],
                      'horizons_ms': training.horizons, 'input_config': training.manifest['input_config'],
                      'grid_id': training.rf['grid_id'], 'allow_exploratory': training.rf.get('allow_exploratory', False),
                      'normalizer': training.manifest['normalizer'], 'selected_epoch': epoch,
                      'validation_recording_macro_iou': score, 'seed': seed, 'roi_weight': roi_weight,
                      'selection': 'validation recording-macro IoU, equally weighted trained horizons; epoch zero eligible'}
            torch.save(bundle, out/'alpha.pt')
        write_json(out/'training.json', {'specification': model.specification(), 'history': log,
                                        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'seed': seed,
                                        'warning': 'Software execution is not evidence of real-data CSI improvement.'})
    return str(out/'alpha.pt')
