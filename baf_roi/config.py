"""Explicit data/geometry contracts; no inferred units or hidden fallbacks."""
from pathlib import Path
import hashlib
import json
import math


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def validate_horizons(horizons_ms):
    """Runtime/evaluation horizons; independent of the training cache config."""
    hs = list(horizons_ms)
    if not hs or len(set(hs)) != len(hs) or any(not math.isfinite(h) or not 50 <= h <= 1000 for h in hs):
        raise ValueError('Distinct horizons must be within [50,1000] ms.')
    return hs


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    for block, key in [('data', 'dataset_root'), ('split', 'manifest'),
                       ('scene', 'adaptive_config_path'), ('scene', 'partition_centers_file')]:
        value = cfg.get(block, {}).get(key)
        if value:
            p = Path(value).expanduser()
            cfg[block][key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    data = cfg['data']
    if data.get('position_unit') not in ('cm', 'm'):
        raise ValueError('data.position_unit must be explicitly cm or m.')
    if data.get('rotation_convention') != 'legacy_ypr_xforward':
        raise ValueError('Only the supplied ROI convention is implemented: yaw,pitch,roll; +X forward, +Z up.')
    if data.get('timestamp_numeric_unit', 's') not in ('s', 'ms'):
        raise ValueError('timestamp_numeric_unit must be s or ms.')
    if not math.isfinite(data.get('max_gap_ms', 100)) or data.get('max_gap_ms', 100) <= 0:
        raise ValueError('max_gap_ms must be positive.')
    model = cfg['model']
    if model.get('sample_rate_hz') != 30 or model.get('history_frames') != 30:
        raise ValueError('This reproduction deliberately fixes 30 Hz / 30 frames. Changing them is a new experiment.')
    hs = cfg['evaluation']['horizons_ms']
    if not hs or len(set(hs)) != len(hs) or any(not math.isfinite(h) or h < 50 or h > 250 for h in hs):
        raise ValueError('Specify distinct horizons in [50, 250] ms.')
    if not 0 <= model.get('temporal_mu', .6) <= 1:
        raise ValueError('temporal_mu must be in [0,1].')
    if cfg['training'].get('batch_size', 64) < 1:
        raise ValueError('Training batch_size must be positive.')
