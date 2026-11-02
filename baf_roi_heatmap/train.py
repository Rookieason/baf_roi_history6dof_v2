"""Train only the small adapter; select checkpoints by validation recording-macro ROI IoU."""
from collections import defaultdict
from dataclasses import asdict
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .contracts import ModelConfig, new_directory, seed_all, write_json
from .data import ForecastDataset
from .model import ForecastAdapter
from .losses import task_loss, numpy_prediction
from .evaluation import to_device, roi_counts


def train_epoch(model, loader, optimizer, scene, device, roi_weight):
    model.train()
    total, count = 0., 0
    for batch in loader:
        inputs, labels = to_device(batch['inputs'], device), to_device(batch['supervision'], device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = task_loss(prediction, labels, scene, roi_weight=roi_weight)
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite adapter loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(inputs['hmd']); count += len(inputs['hmd'])
    return total / count


@torch.inference_mode()
def validation_iou(model, dataset, loader, device):
    model.eval()
    by_record = defaultdict(list)
    for batch in loader:
        prediction = model(to_device(batch['inputs'], device))
        ris, ois = batch['record_index'].numpy(), batch['origin_index'].numpy()
        p0 = np.stack([dataset.arrays(int(ri))['base_p'][int(oi)] for ri, oi in zip(ris, ois)])
        q0 = np.stack([dataset.arrays(int(ri))['base_q'][int(oi)] for ri, oi in zip(ris, ois)])
        p, q = numpy_prediction(prediction, p0, q0)
        truth = batch['supervision']['target_mask'].numpy().astype(bool)
        tp, fp, fn, _ = roi_counts(dataset.scene.mask(p, q), truth)
        union = tp+fp+fn
        values = np.divide(tp, union, out=np.ones_like(tp, dtype=float), where=union != 0)
        for ri, value in zip(ris, values): by_record[int(ri)].append(float(np.mean(value)))
    return float(np.mean([np.mean(values) for values in by_record.values()]))


def train(prepared, output, config=None, epochs=20, batch_size=64, lr=1e-3, seed=42, device='cpu', roi_weight=.25):
    config = (config or ModelConfig()).validate()
    if epochs < 1 or batch_size < 1 or not math.isfinite(lr) or lr <= 0 or not math.isfinite(roi_weight) or roi_weight < 0:
        raise ValueError('Invalid training hyperparameters.')
    seed_all(seed)
    training, validation = ForecastDataset(prepared, 'train'), ForecastDataset(prepared, 'valid')
    if config.modality == 'rf' and not training.manifest['normalizer']['sample_count']:
        raise ValueError('There are no valid training RF maps.')
    if config.modality == 'rf' and not any(bool(training[i]['inputs']['rf_available']) for i in range(len(training))):
        raise ValueError('Training maps exist but none align causally; inspect clocks/support/delivery constraints.')
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator)
    valid_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    model = ForecastAdapter(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    out = new_directory(output)
    log, best = [], -float('inf')
    for epoch in range(epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, training.scene, device, roi_weight) if epoch else None
        score = validation_iou(model, validation, valid_loader, device)
        log.append({'epoch': epoch, 'train_loss': loss, 'validation_recording_macro_iou': score})
        if score > best + 1e-12:
            best = score
            bundle = {'format_version': 1, 'model': model.state_dict(), 'model_config': asdict(config),
                      'experiment_id': training.manifest['experiment_id'], 'baseline_sha256': training.manifest['baseline_sha256'],
                      'horizons_ms': training.horizons, 'input_config': training.manifest['input_config'],
                      'grid_id': training.rf['grid_id'], 'allow_exploratory': training.rf.get('allow_exploratory', False),
                      'normalizer': training.manifest['normalizer'], 'selected_epoch': epoch,
                      'validation_recording_macro_iou': score, 'seed': seed, 'roi_weight': roi_weight,
                      'selection': 'validation recording-macro IoU, equally weighted trained horizons; epoch zero is eligible'}
            torch.save(bundle, out/'adapter.pt')
        write_json(out/'training.json', {'specification': model.specification(), 'history': log,
                                        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'seed': seed,
                                        'warning': 'No evidence of CSI gain until matched real held-out recordings are evaluated.'})
    return str(out/'adapter.pt')
