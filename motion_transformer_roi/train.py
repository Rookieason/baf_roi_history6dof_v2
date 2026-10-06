"""Author optimizer/loss protocol on the SAME cache as BAF, without label filtering."""
from pathlib import Path
from dataclasses import asdict
import math
import torch
from torch.utils.data import DataLoader
from baf_roi.config import dump_json
from baf_roi.data import Cache, OriginDataset
from baf_roi.train import seed_everything
from .models import build_model, utility, source_hashes


def compute_losses(out, batch):
    tr = utility.translation_l1_loss(out['p_hat_tr'], batch['target_tr'])
    rot = utility.quaternion_cosine_loss(out['q_hat_norm'], batch['target_q'])
    return {'loss_total': tr + .3 * rot, 'loss_tr': tr, 'loss_rot': rot}


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, count = {}, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = {k: v.to(device=device, dtype=torch.float32) for k, v in batch.items()}
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(model(batch['pose']), batch)
            if not all(torch.isfinite(v).item() for v in losses.values()):
                raise FloatingPointError('Motion Transformer produced a non-finite loss.')
            if optimizer is not None:
                losses['loss_total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            n = len(batch['pose'])
            count += n
            for key, value in losses.items():
                total[key] = total.get(key, 0.) + float(value.detach()) * n
    if not count:
        raise ValueError('Empty training/validation loader.')
    return {key: value / count for key, value in total.items()}


def train(cache_path, output, device='cpu', epochs=30, batch_size=64, seed=None):
    if not isinstance(epochs, int) or epochs < 1 or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Epochs and batch size must be positive integers.')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new run directory; previous checkpoints will not be overwritten.')
    cache = Cache(cache_path)
    datasets = {s: OriginDataset(cache_path, s, 'all') for s in ('train', 'valid')}
    counts = {s: len(d) for s, d in datasets.items()}
    if min(counts.values()) == 0:
        raise ValueError(f'Need next-step samples in train AND valid: {counts}.')
    seed = int(cache.manifest['config']['training'].get('seed', 42) if seed is None else seed)
    seed_everything(seed)
    model = build_model().to(device)
    loaders = {s: DataLoader(d, batch_size=batch_size, shuffle=s == 'train', num_workers=0)
               for s, d in datasets.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=2)
    protocol = {'epochs_limit': epochs, 'batch_size': batch_size, 'seed': seed,
                'optimizer': 'AdamW', 'lr': 5e-4, 'weight_decay': 1e-4, 'grad_clip': 1.,
                'lambda_tr': 1., 'lambda_rot': .3, 'patience': 6,
                'scheduler': {'name': 'ReduceLROnPlateau', 'factor': .5, 'patience': 2},
                'samples': 'all next-step transitions, including unlabeled and regime changes'}
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / 'data_manifest.json', cache.manifest)
    dump_json(output / 'eligible_counts.json', counts)
    best, stale, history = math.inf, 0, []
    checkpoint = output / 'motion_transformer_roi.pt'
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, loaders['train'], device, optimizer)
        val = run_epoch(model, loaders['valid'], device)
        scheduler.step(val['loss_total'])
        history.append({'epoch': epoch, 'train': tr, 'valid': val, 'lr': optimizer.param_groups[0]['lr']})
        print(f"motion_transformer epoch={epoch} train={tr['loss_total']:.6f} valid={val['loss_total']:.6f}", flush=True)
        if val['loss_total'] < best:
            best, stale = val['loss_total'], 0
            torch.save({'format_version': 2, 'method': 'motion_transformer_roi',
                        'model_state_dict': model.state_dict(), 'model_config': asdict(model.config),
                        'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'best_val_loss': best,
                        'normalization': cache.stats, 'config': cache.manifest['config'],
                        'cache_id': cache.manifest['cache_id'], 'split': cache.manifest['split'],
                        'supervision': cache.manifest['supervision'], 'training_protocol': protocol,
                        'source_sha256': source_hashes(), 'selection': 'validation_loss_only',
                        'model_input': 'history_6dof_only', 'torch_version': str(torch.__version__)}, checkpoint)
        else:
            stale += 1
        dump_json(output / 'training_history.json', {'protocol': protocol, 'history': history, 'best_val_loss': best})
        if stale >= 6:
            break
    return checkpoint
