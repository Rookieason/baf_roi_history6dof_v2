"""Freeze matched baseline predictions once; never tune on held-out targets."""
from dataclasses import asdict
import math
from pathlib import Path
import numpy as np
from baf_roi.config import validate_horizons
from baf_roi.data import Cache
from baf_roi.runtime import Predictor
from baf_roi.pose import ypr_to_quat, quat_to_ypr, interpolate_predictions
from baf_roi.roi import load_scene
from baf_roi.upstream_ops import apply_decision_fusion
from .contracts import InputConfig, digest, file_hash, new_directory, write_json
from .data import load_rf_manifest, fit_normalizer


def fixed_forecast(predictor, histories, horizons, weight):
    """Independent recursive expert / fixed per-step fusion; no future teacher forcing."""
    if not 0 <= weight <= 1:
        raise ValueError('Fixed explorer contribution must be in [0,1].')
    hs = validate_horizons(horizons)
    h = np.asarray(histories, float).copy()
    ps, qs = [h[:, -1, :3].copy()], [ypr_to_quat(h[:, -1, 3:])]
    alpha = np.full(len(h), weight, np.float32)
    for _ in range(math.ceil(max(hs) * 30 / 1000 - 1e-10)):
        raw, fp, fq, ep, eq = predictor._step(h)
        p, q, _ = apply_decision_fusion('full_slerp', alpha, np.asarray(raw).reshape(-1), fp, ep, fq, eq, .5, .5)
        ps.append(p); qs.append(q)
        h = np.concatenate((h[:, 1:], np.concatenate((p, quat_to_ypr(q)), axis=-1)[:, None]), axis=1)
    return interpolate_predictions(np.stack(ps, 1), np.stack(qs, 1), hs)


def observed_weights(raw, times, mu, max_gap, state=None):
    """Commit only actual decision-origin EMA; reset on gaps, never use rollout alpha."""
    result = []
    for value, timestamp in zip(np.asarray(raw).reshape(-1), times):
        if state is not None and timestamp <= state[0]:
            raise ValueError('Observed decision timestamps must strictly increase.')
        alpha = float(value) if state is None or timestamp - state[0] > max_gap else mu * float(value) + (1 - mu) * state[1]
        state = (float(timestamp), alpha)
        result.append(alpha)
    return np.asarray(result, np.float32), state


def prepare(cache_path, baseline_checkpoint, rf_manifest_path, output, horizons=(50, 100, 150, 200, 250, 400, 600, 800, 1000),
            input_config=None, device='cpu', batch_size=64, scene_config=None):
    cfg = (input_config or InputConfig()).validate()
    hs = validate_horizons(horizons)
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    cache = Cache(cache_path)
    predictor = Predictor.load(baseline_checkpoint, device)
    if predictor.bundle['cache_id'] != cache.manifest['cache_id'] or predictor.stats != cache.stats:
        raise ValueError('BAF checkpoint must have been trained on exactly this cache.')
    rf = load_rf_manifest(rf_manifest_path, cache, cfg)
    normalizer = fit_normalizer(rf)
    scene = load_scene(scene_config or cache.manifest['config']['scene'])
    out = new_directory(output); (out / 'frozen').mkdir()
    records = []
    source_splits = {}
    for ti, record in enumerate(cache.records):
        source = record['source_sha256']
        if source in source_splits and source_splits[source] != record['split']:
            raise ValueError('HMD source bytes cross dataset splits.')
        source_splits[source] = record['split']
        arrays = cache.arrays(ti); series = arrays['series']
        chunks, state = [], None
        for start in range(0, len(arrays['anchors']), batch_size):
            anchors = arrays['anchors'][start:start + batch_size]
            histories = np.stack([series.history(int(i)) for i in anchors])
            alpha, state = observed_weights(predictor.confidence(histories), series.t[anchors], predictor.mu,
                                            series.max_gap, state)
            targets, chosen = [], []
            for j, anchor in enumerate(anchors):
                try:
                    target = series.at(series.t[anchor] + np.asarray(hs) / 1000)
                except ValueError:
                    continue
                targets.append(target); chosen.append(j)
            if not chosen:
                continue
            selected = np.asarray(chosen)
            h = histories[selected]
            prediction = predictor.forecast_batch(h, hs, initial_alpha=alpha[selected])
            fp, fq = fixed_forecast(predictor, h, hs, 0.)
            ep, eq = fixed_forecast(predictor, h, hs, 1.)
            xp, xq = fixed_forecast(predictor, h, hs, .5)
            targets = np.stack(targets)
            tq = ypr_to_quat(targets[..., 3:])
            chunks.append({'anchors': anchors[selected], 'origin_s': series.t[anchors[selected]] + record['epoch_start_s'],
                           'alpha': alpha[selected], 'base_p': prediction['positions_m'], 'base_q': prediction['quaternions_wxyz'],
                           'focused_p': fp, 'focused_q': fq, 'explorer_p': ep, 'explorer_q': eq,
                           'fixed_p': xp, 'fixed_q': xq, 'target_p': targets[..., :3], 'target_q': tq,
                           'target_mask': scene.mask(targets[..., :3], tq)})
        if not chunks:
            continue
        merged = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
        destination = out / 'frozen' / (record['key'] + '.npz')
        np.savez_compressed(destination, **merged)
        r = next(x for x in rf['records'] if x['key'] == record['key'])
        records.append({'key': record['key'], 'cache_index': ti, 'split': record['split'], 'session': record['session'],
                        'group_id': r['group_id'], 'scenario': r['scenario'], 'count': len(merged['anchors']),
                        'sha256': file_hash(destination)})
    for split in ('train', 'valid', 'test'):
        if not any(r['split'] == split for r in records):
            raise ValueError(f'No common-horizon origins in {split}; reduce horizons or collect longer recordings.')
    manifest = {'format_version': 1, 'base_repository_commit': '2a75637da2e814205a56484a1e195c2922d9f9c0',
                'pose_cache': str(Path(cache_path).resolve()), 'hmd_cache_id': cache.manifest['cache_id'],
                'baseline_checkpoint': str(Path(baseline_checkpoint).resolve()), 'baseline_sha256': file_hash(baseline_checkpoint),
                'split': cache.manifest['split'], 'horizons_ms': hs, 'input_config': asdict(cfg),
                'rf_manifest': rf, 'normalizer': normalizer, 'scene': scene.snapshot(), 'records': records,
                'notes': 'Frozen pretrained experts; cached targets are supervision only. Original BAF sources unchanged.'}
    manifest['experiment_id'] = digest(manifest)
    write_json(out / 'manifest.json', manifest)
    return manifest
