"""Pose-only recorder adapter, causal history sampling, session splits and cache."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from functools import lru_cache
import csv
import hashlib
import json
import random
import re
import numpy as np
from torch.utils.data import Dataset
from .config import dump_json, fingerprint
from .pose import model_inputs, ypr_to_quat
from .supervision import centered_pair, label_code, numeric_record, protocol_metadata, NUMERIC_FIELDS

CACHE_FORMAT_VERSION = 2


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_timestamp(value, numeric_unit='s'):
    text = str(value).strip()
    if 'T' in text:
        text = re.sub(r'T(\d{2})-(\d{2})-(\d{2})', r'T\1:\2:\3', text)
        date = datetime.fromisoformat(text.replace('Z', '+00:00'))
        if date.tzinfo is None:
            raise ValueError('ISO timestamps must contain a timezone; use Z for UTC.')
        return date.timestamp()
    result = float(text) / (1000 if numeric_unit == 'ms' else 1)
    if not np.isfinite(result):
        raise ValueError('Non-finite timestamp.')
    return result


def read_recording(path, data_cfg):
    """Strict schema, no heatmap/artifact dependency; bad rows are not fabricated."""
    times, poses = [], []
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'payload_json' not in reader.fieldnames:
            raise ValueError(f'{path}: missing payload_json CSV column.')
        for line, row in enumerate(reader, 2):
            try:
                payload = json.loads(row['payload_json'])
                if not isinstance(payload, dict):
                    raise ValueError('payload_json must contain a JSON object.')
                pos = payload['pos']
                rot = payload.get('rot_deg')
                if rot is None:
                    rot = payload.get('rot')
                if not isinstance(pos, list) or not isinstance(rot, list) or len(pos) != 3 or len(rot) != 3:
                    raise ValueError('Need pos[3] and rot_deg[3] (or rot[3]) in degrees.')
                pose = np.asarray([*pos, *rot], dtype=np.float64)
                if pose.shape != (6,) or not np.isfinite(pose).all():
                    raise ValueError('Pose must be six finite scalars.')
                ts = payload.get('ts_utc')
                if ts is None or ts == '':
                    ts = row.get('ts_utc')
                unit = data_cfg.get('timestamp_numeric_unit', 's')
                if ts is None or ts == '':
                    ts = row.get('recv_unix')
                    unit = 's'  # recorder Unix receive time is explicitly seconds
                if ts is None or ts == '':
                    raise ValueError('Missing ts_utc and recv_unix.')
                times.append(parse_timestamp(ts, unit))
                poses.append(pose)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                raise ValueError(f'{path}: CSV row {line}: {exc}') from exc
    if len(times) < 2:
        raise ValueError(f'{path}: fewer than two usable poses.')
    order = np.argsort(times, kind='stable')
    t, x = np.asarray(times, np.float64)[order], np.asarray(poses, np.float64)[order]
    duplicate = np.flatnonzero(np.diff(t) == 0)
    for i in duplicate:
        if not np.array_equal(x[i], x[i + 1]):
            raise ValueError(f'{path}: conflicting poses at the same timestamp {t[i]}.')
    keep = np.r_[True, np.diff(t) > 0]
    t, x = t[keep], x[keep]
    if len(t) < 2:
        raise ValueError(f'{path}: fewer than two distinct timestamps after deduplication.')
    x[:, :3] *= .01 if data_cfg['position_unit'] == 'cm' else 1.
    x[:, 3:] = (x[:, 3:] + 180) % 360 - 180
    return t, x, {'identical_duplicates_removed': int(len(duplicate)), 'rows': len(t)}


@dataclass
class Series:
    t: np.ndarray  # relative seconds; strictly increasing
    pose: np.ndarray  # [N,6], meters and dataset yaw,pitch,roll degrees
    max_gap: float = .1
    unwrapped: np.ndarray | None = None
    validate_values: bool = True

    def __post_init__(self):
        if len(self.t) < 2 or self.pose.shape != (len(self.t), 6):
            raise ValueError('Invalid series shape or insufficient data.')
        if self.validate_values and (not np.isfinite(self.t).all() or not np.isfinite(self.pose).all() or np.any(np.diff(self.t) <= 0)):
            raise ValueError('Non-finite values or non-increasing timestamps.')
        if self.unwrapped is None:
            # Unwrap is prefix-causal: appending future samples never changes its prefix.
            self.unwrapped = np.rad2deg(np.unwrap(np.deg2rad(self.pose[:, 3:]), axis=0))

    def at(self, query, cutoff_index=None):
        query = np.atleast_1d(np.asarray(query, np.float64))
        eps = 1e-8  # seconds, not a frame/horizon tolerance
        if query.min() < self.t[0] - eps or query.max() > self.t[-1] + eps:
            raise ValueError('Query outside trajectory; extrapolation is forbidden.')
        query = np.clip(query, self.t[0], self.t[-1])
        hi = np.searchsorted(self.t, query, side='left').clip(0, len(self.t) - 1)
        exact = np.abs(self.t[hi] - query) <= eps
        lo = np.where(exact, hi, hi - 1)
        if (lo < 0).any() or ((self.t[hi] - self.t[lo]) > self.max_gap + eps).any():
            raise ValueError('Interpolation would cross a recording gap.')
        if cutoff_index is not None and (hi > cutoff_index).any():
            raise ValueError('History interpolation attempted to read beyond the decision time.')
        denom = self.t[hi] - self.t[lo]
        w = np.divide(query - self.t[lo], denom, out=np.zeros_like(query), where=denom > eps)
        pos = self.pose[lo, :3] * (1 - w[:, None]) + self.pose[hi, :3] * w[:, None]
        angles = self.unwrapped[lo] * (1 - w[:, None]) + self.unwrapped[hi] * w[:, None]
        return np.concatenate([pos, (angles + 180) % 360 - 180], axis=1)

    def history(self, anchor_index):
        anchor = self.t[anchor_index]
        return self.at(anchor - np.arange(29, -1, -1) / 30, cutoff_index=anchor_index)

    def future_history(self, anchor_index):
        """Offline next-step pose history utility; NOT the centered supervision window."""
        return self.at(self.t[anchor_index] + 1 / 30 - np.arange(29, -1, -1) / 30)

    def origin_indices(self):
        # Approx. 30-Hz decision opportunities, snapped FORWARD to an observed row.
        # The actual observed row time is t; horizons are relative to that actual t.
        ticks = np.arange(self.t[0] + 29 / 30, self.t[-1] + 1e-9, 1 / 30)
        indices = np.searchsorted(self.t, ticks - 1e-8).clip(0, len(self.t) - 1)
        return np.unique(indices)


def discover_recordings(root):
    db = Path(root) / 'db'
    if not db.is_dir():
        raise ValueError(f'Dataset must contain db/<session>/agent6*: {db}')
    result = {}
    for session in sorted(db.iterdir()):
        if not session.is_dir():
            continue
        paths = []
        for candidate in sorted(session.glob('agent6*')):
            if candidate.is_file() and candidate.suffix.lower() == '.csv':
                paths.append(candidate)
            elif candidate.is_dir():
                paths.extend(sorted(candidate.rglob('*.csv')))
        if paths:
            result[session.name] = sorted(set(paths))
    if not result:
        raise ValueError(f'No agent6 CSV recordings found below {db}.')
    return result


def make_split(session_ids, cfg):
    ids = sorted(session_ids)
    manifest = cfg.get('manifest')
    if manifest:
        split = json.loads(Path(manifest).read_text(encoding='utf-8'))
        split = {'train': split['train'], 'valid': split.get('valid', split.get('val')), 'test': split['test']}
    else:
        random.Random(cfg.get('seed', 57)).shuffle(ids)
        n = len(ids)
        ntr, nv = int(n * .7), int(n * .15)
        split = {'train': ids[:ntr], 'valid': ids[ntr:ntr + nv], 'test': ids[ntr + nv:]}
    if any(not isinstance(split.get(k), list) or not split[k] for k in ('train', 'valid', 'test')):
        raise ValueError('Need nonempty train/valid/test session splits. Supply an explicit split manifest for small data.')
    flat = [s for group in split.values() for s in group]
    if len(flat) != len(set(flat)) or set(flat) != set(session_ids):
        raise ValueError('Split must cover each discovered session exactly once; no overlap or stale/missing IDs.')
    return split


class Moments:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, np.zeros(3), np.zeros(3)

    def add(self, values):
        values = np.asarray(values, np.float64)
        n = len(values)
        delta = values.mean(axis=0) - self.mean
        self.m2 += ((values - values.mean(axis=0)) ** 2).sum(axis=0) + delta ** 2 * self.n * n / (self.n + n)
        self.mean += delta * n / (self.n + n)
        self.n += n

    def result(self):
        if not self.n:
            raise ValueError('No training poses for normalization.')
        return {'mean': self.mean.tolist(), 'std': (np.sqrt(self.m2 / self.n) + 1e-6).tolist(),
                'count': self.n, 'fit_scope': 'training_sessions_only; valid raw poses in retained segments'}


def prepare(cfg, output):
    if not cfg['data'].get('coordinate_contract_confirmed', False):
        raise ValueError('Confirm position_unit and legacy_ypr_xforward against YOUR recorder, then set coordinate_contract_confirmed=true.')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty. Use a NEW cache path; existing experiments are never overwritten.')
    recordings = discover_recordings(cfg['data']['dataset_root'])
    split = make_split(recordings, cfg['split'])
    membership = {s: key for key, sessions in split.items() for s in sessions}
    output.mkdir(parents=True, exist_ok=True)
    max_gap = cfg['data'].get('max_gap_ms', 100) / 1000
    stats, records, excluded = Moments(), [], []
    for session, files in recordings.items():
        for path in files:
            t, pose, audit = read_recording(path, cfg['data'])
            bounds = np.r_[0, np.flatnonzero(np.diff(t) > max_gap) + 1, len(t)]
            file_hash = sha256_file(path)
            for segment_id, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
                if right - left < 2 or t[right - 1] - t[left] < 29 / 30:
                    excluded.append({'file': str(path), 'segment': segment_id, 'reason': 'insufficient history'})
                    continue
                series = Series(t[left:right] - t[left], pose[left:right], max_gap)
                anchors, labels, next_labels, targets, scores = [], [], [], [], []
                current_meta, next_meta = [], []
                for i in series.origin_indices():
                    try:
                        history = series.history(i)
                    except ValueError:
                        continue
                    # Author's centered LABELS are offline supervision only.
                    # The causal history above never enters this scoring path.
                    current_result, next_result = centered_pair(series, int(i))
                    label = label_code(current_result)
                    score = current_result['explorer_score'] if current_result is not None else np.nan
                    try:
                        target = series.at([series.t[i] + 1 / 30])[0]
                        nl = label_code(next_result)
                    except ValueError:
                        nl, target = -2, np.full(6, np.nan)
                    anchors.append(i); labels.append(label); next_labels.append(nl)
                    targets.append(target); scores.append(score)
                    current_meta.append(numeric_record(current_result))
                    next_meta.append(numeric_record(next_result))
                if not anchors:
                    excluded.append({'file': str(path), 'segment': segment_id, 'reason': 'no causal windows'})
                    continue
                key = f'trajectory_{len(records):05d}'
                target_dir = output / key
                target_dir.mkdir()
                for name, arr in {'t': series.t, 'pose': series.pose, 'unwrapped': series.unwrapped,
                                  'anchors': np.array(anchors, np.int64), 'labels': np.array(labels, np.int8),
                                  'next_labels': np.array(next_labels, np.int8), 'targets': np.array(targets),
                                  'scores': np.array(scores)}.items():
                    np.save(target_dir / (name + '.npy'), arr, allow_pickle=False)
                # Audit-only copies of the author's original supervision fields.
                # head_rate_yaw_q90 remains rad/s; no model consumes this archive.
                np.savez_compressed(target_dir / 'supervision_metadata.npz',
                    **{prefix + field: np.array([row[field] for row in rows], np.float64)
                       for prefix, rows in (('current_', current_meta), ('next_', next_meta))
                       for field in NUMERIC_FIELDS})
                label, nl = np.array(labels), np.array(next_labels)
                counts = {'input_windows': len(label), 'confidence': int(np.sum(label >= 0)),
                          'all_next': int(np.sum(nl != -2)),
                          'focused_clean': int(np.sum((label == 0) & (nl == 0))),
                          'explorer_clean': int(np.sum((label == 1) & (nl == 1))),
                          'focused': int(np.sum(label == 0)), 'explorer': int(np.sum(label == 1)),
                          'unlabeled': int(np.sum(label == -1))}
                records.append({'key': key, 'session': session, 'source': str(path), 'source_sha256': file_hash,
                                'segment': segment_id, 'split': membership[session], 'epoch_start_s': float(t[left]),
                                'raw_rows': int(right - left), 'median_raw_interval_ms': float(np.median(np.diff(series.t)) * 1000),
                                'counts': counts, 'read_audit': audit})
                if membership[session] == 'train':
                    stats.add(series.pose[:, :3])
    for group in ('train', 'valid', 'test'):
        if not any(r['split'] == group for r in records):
            raise ValueError(f'No usable trajectories in {group}; inspect missing data/gaps.')
    result = {'format_version': CACHE_FORMAT_VERSION, 'supervision': protocol_metadata(), 'config': cfg, 'split': split, 'trajectories': records,
              'excluded_segments': excluded, 'normalization': stats.result(),
              'data_use': 'history 6DoF only; no CSI, MAT, PNG, gaze, scene or ROI features'}
    result['cache_id'] = fingerprint(result)
    dump_json(output / 'manifest.json', result)
    print(json.dumps({'cache_id': result['cache_id'], 'trajectories': len(records),
                      'counts': {g: {k: sum(r['counts'][k] for r in records if r['split'] == g)
                                     for k in records[0]['counts']} for g in ('train', 'valid', 'test')}}, indent=2))
    return result


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('format_version') != CACHE_FORMAT_VERSION:
            raise ValueError('Cache protocol changed to author-centered supervision v2. '
                             'Run prepare into a NEW cache and retrain all four stages; do not reuse v1.')
        if self.manifest.get('supervision', {}).get('id') != protocol_metadata()['id']:
            raise ValueError('Unexpected supervision protocol in cache; rebuild with this release.')
        if self.manifest['supervision']['source_sha256'] != protocol_metadata()['source_sha256']:
            raise ValueError('Author supervision source differs from this cache; rebuild after source changes.')
        self.stats = self.manifest['normalization']
        self.records = self.manifest['trajectories']

    @lru_cache(maxsize=4)
    def arrays(self, index):
        root = self.path / self.records[index]['key']
        arrays = {name: np.load(root / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                  for name in ('t', 'pose', 'unwrapped', 'anchors', 'labels', 'next_labels', 'targets', 'scores')}
        # Cache was validated during prepare. Avoid re-scanning an entire file on
        # every LRU miss during globally shuffled training; each model window is
        # still checked for finite values by model_inputs. Do not edit cache files.
        arrays['series'] = Series(arrays['t'], arrays['pose'],
                                  self.manifest['config']['data']['max_gap_ms'] / 1000, arrays['unwrapped'], validate_values=False)
        return arrays


class OriginDataset(Dataset):
    """Compact integer index + memory-mapped trajectories, never all windows in RAM."""
    def __init__(self, cache, split, mode):
        self.cache = Cache(cache)
        chunks = []
        for i, record in enumerate(self.cache.records):
            if record['split'] != split:
                continue
            arrays = self.cache.arrays(i)
            label, nl = arrays['labels'], arrays['next_labels']
            if mode == 'confidence':
                valid = label >= 0
            elif mode == 'all':
                valid = nl != -2
            elif mode == 'focused_clean':
                valid = (label == 0) & (nl == 0)
            elif mode == 'explorer_clean':
                valid = (label == 1) & (nl == 1)
            else:
                raise ValueError(mode)
            ids = np.flatnonzero(valid)
            chunks.append(np.column_stack((np.full(len(ids), i, np.int64), ids)))
        self.index = np.concatenate(chunks) if chunks else np.empty((0, 2), np.int64)
        self.mode = mode

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ti, oi = map(int, self.index[idx])
        a = self.cache.arrays(ti)
        h = a['series'].history(int(a['anchors'][oi]))
        conf, pose = model_inputs(h, self.cache.stats)
        if self.mode == 'confidence':
            return {'conf': conf, 'label': np.float32(a['labels'][oi])}
        target = a['targets'][oi]
        return {'pose': pose, 'target_tr': ((target[:3] - self.cache.stats['mean']) / self.cache.stats['std']).astype(np.float32),
                'target_q': ypr_to_quat(target[3:]).astype(np.float32)}
