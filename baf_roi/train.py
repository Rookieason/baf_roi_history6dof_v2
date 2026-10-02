"""Four original training stages; original model/loss functions are called directly."""
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import confidence, focused, explorer
from .upstream_ops import weighted_bce_loss
from .data import Cache, OriginDataset
from .config import dump_json

STAGES = {
    'confidence': ('confidence', 10, 1e-3, None),
    'focused_pretrain': ('all', 8, 1e-3, 4),
    'focused_finetune': ('focused_clean', 12, 1e-4, 5),
    'explorer': ('explorer_clean', 20, 1e-3, 5),
}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_component(stage):
    if stage == 'confidence':
        return confidence.build_confidence_estimator(input_dim=6, seq_len=30)
    if stage == 'explorer':
        return explorer.build_explorer_branch(pose_dim=7, aux_dim=0)
    return focused.FocusedResidualLSTM(focused.FocusedBranchConfig())


def stage_losses(stage, model, batch):
    if stage == 'confidence':
        out = model(batch['conf'])
        return {'loss_total': weighted_bce_loss(out['logits'], batch['label'], focused_weight=3., explorer_weight=1.)}
    x = batch['pose']
    out = model(x)
    args = dict(delta_hat_tr=out['delta_hat_tr'], current_translation=out['current_translation'],
                target_translation=batch['target_tr'], q_hat_norm=out['q_hat_norm'],
                target_quaternion=batch['target_q'], current_quaternion=x[:, -1, 3:])
    if stage == 'explorer':
        return explorer.explorer_total_loss(**args, prev_quaternion=x[:, -2, 3:], gamma_rot=.3, eta_tail=.1)
    return focused.focused_total_loss(**args, prev_translation=x[:, -2, :3],
                                      lambda_tr=1., lambda_rot=.3, lambda_smooth=.1)


def run_epoch(stage, model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, n = {}, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = {k: v.to(device=device, dtype=torch.float32) for k, v in batch.items()}
            if optimizer is not None:
                optimizer.zero_grad()
            losses = stage_losses(stage, model, batch)
            loss = losses['loss_total']
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{stage}: non-finite loss; inspect units, normalization and data.')
            if optimizer is not None:
                loss.backward()
                if stage != 'confidence':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            bs = len(next(iter(batch.values())))
            for key, value in losses.items():
                total[key] = total.get(key, 0.) + float(value.detach()) * bs
            n += bs
    if n == 0:
        raise ValueError('Empty training/validation loader.')
    return {k: v / n for k, v in total.items()}


def train(cache_path, run_dir, device='cpu', epochs_override=None):
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('Use a new run directory; previous checkpoints will not be overwritten.')
    cache = Cache(cache_path)
    cfg = cache.manifest['config']
    # Preflight every regime BEFORE starting any training. No silent branch fallback.
    counts = {}
    for stage, (mode, _, _, _) in STAGES.items():
        counts[stage] = {s: len(OriginDataset(cache_path, s, mode)) for s in ('train', 'valid')}
        if any(v == 0 for v in counts[stage].values()):
            raise ValueError(f'{stage} has no eligible samples: {counts[stage]}. Do not silently relabel data or train on test.')
    for group in ('train', 'valid'):
        focused_count = sum(r['counts']['focused'] for r in cache.records if r['split'] == group)
        explorer_count = sum(r['counts']['explorer'] for r in cache.records if r['split'] == group)
        if min(focused_count, explorer_count) == 0:
            raise ValueError(f'{group} confidence data needs BOTH Focused and Explorer labels.')
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_json(run_dir / 'data_manifest.json', cache.manifest)
    dump_json(run_dir / 'eligible_counts.json', counts)
    histories, final_states = {}, {}
    for stage, (mode, default_epochs, lr, patience) in STAGES.items():
        seed_everything(int(cfg['training'].get('seed', 42)))
        model = create_component(stage).to(device)
        if stage == 'focused_finetune':
            previous = torch.load(run_dir / 'focused_pretrain_best.pt', map_location=device, weights_only=True)
            model.load_state_dict(previous['model_state_dict'], strict=True)
        loaders = {group: DataLoader(OriginDataset(cache_path, group, mode),
                                    batch_size=cfg['training'].get('batch_size', 64),
                                    shuffle=group == 'train', num_workers=0, drop_last=False)
                   for group in ('train', 'valid')}
        if stage == 'confidence':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = None
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=2)
        epochs = epochs_override if epochs_override is not None else cfg['training'].get('epochs', {}).get(stage, default_epochs)
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError('Epoch counts must be positive integers.')
        best, stale, history = float('inf'), 0, []
        for epoch in range(1, epochs + 1):
            tr = run_epoch(stage, model, loaders['train'], device, optimizer)
            val = run_epoch(stage, model, loaders['valid'], device)
            if scheduler:
                scheduler.step(val['loss_total'])
            row = {'epoch': epoch, 'train': tr, 'valid': val, 'lr': optimizer.param_groups[0]['lr']}
            history.append(row)
            print(f"{stage} epoch={epoch} train={tr['loss_total']:.6f} valid={val['loss_total']:.6f}", flush=True)
            if val['loss_total'] < best:
                best, stale = val['loss_total'], 0
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                            'stage': stage, 'epoch': epoch, 'best_val_loss': best}, run_dir / f'{stage}_best.pt')
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    break
        histories[stage] = history
        dump_json(run_dir / 'training_history.json', histories)
        best_ckpt = torch.load(run_dir / f'{stage}_best.pt', map_location='cpu', weights_only=True)
        key = 'focused' if stage.startswith('focused') else stage
        final_states[key] = best_ckpt['model_state_dict']
    bundle = {'format_version': 2, 'supervision': cache.manifest['supervision'], 'models': final_states, 'normalization': cache.stats,
              'config': cfg, 'cache_id': cache.manifest['cache_id'], 'split': cache.manifest['split'],
              'epochs_override': epochs_override, 'model_input': 'history_6dof_only',
              'torch_version': str(torch.__version__), 'selection': 'validation_loss_only'}
    torch.save(bundle, run_dir / 'baf_roi.pt')
    return run_dir / 'baf_roi.pt'
