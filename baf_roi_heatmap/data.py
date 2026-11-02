"""Causal numeric heatmap loading, train-only normalization and paired datasets."""
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from baf_roi.data import Cache
from baf_roi.pose import model_inputs
from baf_roi.roi import Scene
from .contracts import InputConfig, digest, read_json, file_hash


def to_db(values, unit):
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Valid maps must be finite.')
    if unit == 'linear':
        if np.any(values < 0):
            raise ValueError('Linear MUSIC spectra cannot be negative.')
        return 10 * np.log10(np.maximum(values, 1e-12))
    if unit == 'db':
        return values.copy()
    raise ValueError('Heatmap unit must be explicitly linear or db.')


class HeatmapSeries:
    def __init__(self, path, offset_s=0., uncertainty_s=0., allow_exploratory=False):
        self.path = str(path)
        if not np.isfinite(offset_s) or not np.isfinite(uncertainty_s) or uncertainty_s < 0:
            raise ValueError('Clock offset must be finite; uncertainty must be nonnegative.')
        with np.load(path, allow_pickle=False) as data:
            self.maps = np.asarray(data['maps'], np.float32)
            self.valid = np.asarray(data['valid'], bool)
            self.start = np.asarray(data['support_start_s'], float) + offset_s
            self.end = np.asarray(data['support_end_s'], float) + offset_s
            self.ready = np.asarray(data['available_at_s'], float) + offset_s
            self.theta, self.tau = np.asarray(data['theta_deg'], float), np.asarray(data['tau_s'], float)
            self.metadata = json.loads(str(data['metadata_json'].item()))
        n = len(self.maps)
        if not n or self.maps.shape != (n, len(self.theta), len(self.tau)):
            raise ValueError('maps must be nonempty [frames, angle, delay].')
        if any(x.shape != (n,) for x in (self.valid, self.start, self.end, self.ready)):
            raise ValueError('One validity/support/availability value is required per map.')
        for axis in (self.theta, self.tau):
            if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
                raise ValueError('Physical axes must be finite and increasing.')
            if not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-4, atol=1e-15):
                raise ValueError('CNN resizing assumes uniform axes; explicitly regrid irregular maps first.')
        if not all(np.isfinite(x).all() for x in (self.start, self.end, self.ready)):
            raise ValueError('Map timing contains non-finite values.')
        if np.any(self.start > self.end) or np.any(self.ready < self.end) or np.any(np.diff(self.end) <= 0):
            raise ValueError('Invalid support, availability, or repeated/out-of-order measurement times.')
        if self.metadata.get('format_version') != 1 or self.metadata.get('representation') != 'aoa_tof':
            raise ValueError('Expected version 1 numeric aoa_tof maps.')
        if not self.metadata.get('clock_domain') or not self.metadata.get('availability_source'):
            raise ValueError('Declare source clock and how available_at_s was obtained.')
        if not self.metadata.get('physical_axes_verified', False) and not allow_exploratory:
            raise ValueError('Unverified physical axes: explicitly allow exploratory RF, or calibrate first.')
        self.db = np.zeros_like(self.maps)
        self.db[self.valid] = to_db(self.maps[self.valid], self.metadata.get('unit'))
        self.uncertainty = float(uncertainty_s)
        self.grid_id = digest({'theta': self.theta.tolist(), 'tau': self.tau.tolist(), 'representation': 'aoa_tof'})

    def indices(self, origin_s, cfg, delivery_delay_s=0.):
        """Latest *available measurement*, not nearest future or linearly interpolated RF."""
        if not np.isfinite(origin_s) or not np.isfinite(delivery_delay_s) or delivery_delay_s < 0:
            raise ValueError('Decision time must be finite; delivery delay must be nonnegative.')
        queries = np.linspace(origin_s - cfg.map_history_s, origin_s, cfg.frames)
        ids = np.full(cfg.frames, -1, np.int64)
        if self.uncertainty > cfg.max_clock_uncertainty_s:
            return queries, ids
        lo = np.searchsorted(self.end, origin_s - cfg.map_history_s - cfg.max_age_s - self.uncertainty - 1e-9)
        hi = np.searchsorted(self.end, origin_s + 1e-9, side='right')
        candidates = np.arange(lo, hi)
        eligible = self.valid[candidates] & (self.start[candidates] - self.uncertainty >= origin_s - cfg.raw_history_s - 1e-9)
        for j, query in enumerate(queries):
            okay = eligible & (self.end[candidates] + self.uncertainty <= query + 1e-9)
            okay &= self.ready[candidates] + delivery_delay_s + self.uncertainty <= query + 1e-9
            okay &= query - (self.end[candidates] - self.uncertainty) <= cfg.max_age_s + 1e-9
            found = candidates[okay]
            if len(found):
                ids[j] = found[-1]
        return queries, ids


def empty_window(cfg):
    return {'maps': np.zeros((cfg.frames, 3, cfg.size, cfg.size), np.float32),
            'mask': np.zeros(cfg.frames, bool), 'timing': np.zeros((cfg.frames, 2), np.float32),
            'rf_available': np.bool_(False)}


def encode_window(series, origin_s, normalizer, cfg, delivery_delay_s=0.):
    """Channels: level, causal within-window background residual, signed temporal difference."""
    if series is None:
        return empty_window(cfg)
    queries, ids = series.indices(origin_s, cfg, delivery_delay_s)
    mask = ids >= 0
    result = empty_window(cfg)
    result['mask'] = mask
    levels = np.zeros((cfg.frames, len(series.theta), len(series.tau)), np.float32)
    levels[mask] = np.clip((series.db[ids[mask]] - normalizer['median']) / normalizer['scale'], -cfg.clip, cfg.clip)
    features = np.zeros((cfg.frames, 3, *levels.shape[1:]), np.float32)
    prior, previous_id, previous_level, previous_end = [], -1, None, None
    for j, idx in enumerate(ids):
        if idx < 0:
            prior, previous_id, previous_level, previous_end = [], -1, None, None
            continue
        level = levels[j]
        features[j, 0] = level
        if prior:
            features[j, 1] = level - np.mean(prior, axis=0)
        if previous_level is not None and idx != previous_id and series.end[idx] - previous_end <= cfg.max_age_s:
            features[j, 2] = level - previous_level
        if idx != previous_id:
            prior.append(level)
        previous_id, previous_level, previous_end = idx, level, series.end[idx]
        result['timing'][j] = ((queries[j] - series.end[idx]) / cfg.max_age_s,
                               (queries[j] - origin_s) / cfg.map_history_s)
    result['maps'] = F.interpolate(torch.from_numpy(features), (cfg.size, cfg.size),
                                   mode='bilinear', align_corners=True).numpy()
    result['rf_available'] = np.bool_(mask[-1] and mask.mean() >= cfg.min_valid_fraction)
    return result


def load_rf_manifest(path, cache, cfg):
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get('format_version') != 1 or manifest.get('hmd_cache_id') != cache.manifest['cache_id']:
        raise ValueError('RF manifest must explicitly identify this HMD cache.')
    if manifest.get('hmd_clock_domain') != 'unix_seconds' or manifest.get('synchronization_confirmed') is not True:
        raise ValueError('Confirm clock mapping to the HMD Unix-second timestamps.')
    entries = manifest['records']
    mapping = {r['key']: r.copy() for r in entries}
    if len(mapping) != len(entries) or set(mapping) != {r['key'] for r in cache.records}:
        raise ValueError('RF manifest must cover every HMD record exactly once; use file=null for missing RF.')
    groups, rf_files, raw_sources, grids = {}, {}, {}, set()
    for record in cache.records:
        r = mapping[record['key']]
        group = str(r.get('group_id') or record['session'])
        if group in groups and groups[group] != record['split']:
            raise ValueError('A subject/session group crosses dataset splits.')
        groups[group] = record['split']
        r.update(split=record['split'], session=record['session'], group_id=group,
                 scenario=str(r.get('scenario', 'unspecified')))
        if r.get('file'):
            p = Path(r['file']).expanduser()
            p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
            r['file'] = str(p)
            r['sha256'] = file_hash(p)
            if r['sha256'] in rf_files and rf_files[r['sha256']] != record['split']:
                raise ValueError('Identical RF recording bytes cross train/valid/test splits.')
            rf_files[r['sha256']] = record['split']
            if 'clock_offset_s' not in r or 'clock_uncertainty_s' not in r:
                raise ValueError('Declare per-record clock_offset_s and clock_uncertainty_s, even when zero.')
            series = HeatmapSeries(p, r['clock_offset_s'], r['clock_uncertainty_s'], manifest.get('allow_exploratory', False))
            raw_source = series.metadata.get('raw_sha256')
            if raw_source:
                if raw_source in raw_sources and raw_sources[raw_source] != record['split']:
                    raise ValueError('Reprocessed versions of the same raw CSI recording cross splits.')
                raw_sources[raw_source] = record['split']
            r['raw_sha256'] = raw_source
            if series.metadata['clock_domain'] != r.get('source_clock_domain'):
                raise ValueError('The declared source clock disagrees with map metadata.')
            if series.uncertainty > cfg.max_clock_uncertainty_s:
                raise ValueError('Clock uncertainty exceeds the configured research tolerance.')
            grids.add(series.grid_id)
            r['physical_axes_verified'] = series.metadata.get('physical_axes_verified', False)
        else:
            r.update(file=None, sha256=None, physical_axes_verified=False)
    if len(grids) > 1:
        raise ValueError('Different physical grids must be explicitly regridded; do not silently resize them together.')
    manifest['grid_id'] = next(iter(grids), None)
    manifest['records'] = [mapping[r['key']] for r in cache.records]
    return manifest


def fit_normalizer(rf_manifest):
    values, source_keys = [], []
    for r in rf_manifest['records']:
        if r['split'] != 'train' or not r['file']:
            continue
        series = HeatmapSeries(r['file'], r['clock_offset_s'], r['clock_uncertainty_s'], rf_manifest.get('allow_exploratory', False))
        flat = series.db[series.valid].reshape(-1)
        if len(flat):
            values.append(flat[np.linspace(0, len(flat) - 1, min(len(flat), 4096), dtype=int)])
            source_keys.append(r['key'])
    if not values:
        return {'median': 0., 'scale': 1., 'fit_split': 'train', 'record_keys': [], 'sample_count': 0}
    x = np.concatenate(values)
    median = float(np.median(x))
    return {'median': median, 'scale': max(float(1.4826 * np.median(np.abs(x - median))), 1e-3),
            'fit_split': 'train', 'record_keys': source_keys, 'sample_count': len(x)}


class ForecastDataset(Dataset):
    """Inputs and future supervision are separate. Labels never enter model.forward()."""
    def __init__(self, root, split, delivery_delay_s=0.):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / 'manifest.json')
        identity = self.manifest.copy(); identity.pop('experiment_id', None)
        if self.manifest.get('experiment_id') != digest(identity):
            raise ValueError('Prepared manifest was modified after preparation.')
        if split not in ('train', 'valid', 'test'):
            raise ValueError('split must be train, valid or test.')
        self.cache = Cache(self.manifest['pose_cache'])
        if self.cache.manifest['cache_id'] != self.manifest['hmd_cache_id']:
            raise ValueError('Underlying pose cache changed.')
        self.cfg = InputConfig(**self.manifest['input_config']).validate()
        self.rf = self.manifest['rf_manifest']
        self.rf_by_key = {r['key']: r for r in self.rf['records']}
        scene = self.manifest['scene']
        self.scene = Scene(np.array(scene['centers_m']), scene['partition_ids'], scene['config'])
        if not np.isfinite(delivery_delay_s) or delivery_delay_s < 0:
            raise ValueError('Delivery delay must be finite and nonnegative.')
        self.split, self.delivery_delay_s = split, float(delivery_delay_s)
        self.horizons = self.manifest['horizons_ms']
        self.records = [r for r in self.manifest['records'] if r['split'] == split]
        index = [(ri, oi) for ri, r in enumerate(self.records) for oi in range(r['count'])]
        self.index = np.asarray(index, np.int64).reshape(-1, 2)
        if not len(self.index):
            raise ValueError(f'No common-horizon origins in {split}.')

    @lru_cache(maxsize=4)
    def arrays(self, record_index):
        r = self.records[record_index]
        path = self.root / 'frozen' / (r['key'] + '.npz')
        if file_hash(path) != r['sha256']:
            raise ValueError('Frozen predictions/targets changed after preparation.')
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    @lru_cache(maxsize=4)
    def rf_series(self, key):
        r = self.rf_by_key[key]
        if not r['file']:
            return None
        if file_hash(r['file']) != r['sha256']:
            raise ValueError('RF data changed after preparation.')
        return HeatmapSeries(r['file'], r['clock_offset_s'], r['clock_uncertainty_s'], self.rf.get('allow_exploratory', False))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        ri, oi = map(int, self.index[index])
        r, a = self.records[ri], self.arrays(ri)
        raw = self.cache.arrays(r['cache_index'])
        history = raw['series'].history(int(a['anchors'][oi]))
        _, pose = model_inputs(history, self.cache.stats)
        window = encode_window(self.rf_series(r['key']), float(a['origin_s'][oi]),
                               self.manifest['normalizer'], self.cfg, self.delivery_delay_s)
        inputs = {'hmd': pose, **window, 'horizon_s': np.asarray(self.horizons, np.float32) / 1000,
                  'base_p': a['base_p'][oi].astype(np.float32), 'base_q': a['base_q'][oi].astype(np.float32),
                  'focused_p': a['focused_p'][oi].astype(np.float32), 'focused_q': a['focused_q'][oi].astype(np.float32),
                  'explorer_p': a['explorer_p'][oi].astype(np.float32), 'explorer_q': a['explorer_q'][oi].astype(np.float32),
                  'alpha': np.float32(a['alpha'][oi])}
        supervision = {'target_p': a['target_p'][oi].astype(np.float32), 'target_q': a['target_q'][oi].astype(np.float32),
                       'target_mask': a['target_mask'][oi].astype(np.float32)}
        return {'inputs': inputs, 'supervision': supervision, 'record_index': ri, 'origin_index': oi}
