"""Explicit experiment configuration and reproducibility helpers."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import random
import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def read_scene_config(path):
    path = Path(path).resolve()
    obj = read_json(path)
    scene = obj.get('scene', obj).copy()
    for key in ('adaptive_config_path', 'partition_centers_file'):
        if scene.get(key):
            p = Path(scene[key]).expanduser()
            scene[key] = str(p.resolve() if p.is_absolute() else (path.parent / p).resolve())
    return scene


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def new_directory(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Refusing to overwrite a nonempty experiment directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class InputConfig:
    raw_history_s: float = 29 / 30
    map_history_s: float = 0.4
    frames: int = 8
    size: int = 24
    max_age_s: float = 0.15
    min_valid_fraction: float = 0.5
    max_clock_uncertainty_s: float = 0.01
    clip: float = 6.0

    def validate(self):
        values = asdict(self)
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError('Input configuration values must be finite and positive.')
        if self.map_history_s > self.raw_history_s or self.raw_history_s > 29 / 30 + 1e-9:
            raise ValueError('RF raw support must fit the 30-pose HMD history budget.')
        if self.frames < 2 or self.size < 4 or not isinstance(self.frames, int) or not isinstance(self.size, int):
            raise ValueError('frames >= 2 and size >= 4 must be integers.')
        if self.min_valid_fraction > 1:
            raise ValueError('min_valid_fraction must be <= 1.')
        return self


@dataclass(frozen=True)
class ModelConfig:
    method: str = 'residual'
    modality: str = 'rf'
    ablation: str = 'full'
    hidden: int = 32
    max_translation_m: float = 0.2
    max_rotation_deg: float = 20.0
    max_logit_delta: float = 2.0

    def validate(self):
        if self.method not in ('residual', 'expert_gate') or self.modality not in ('rf', 'hmd'):
            raise ValueError('method=residual|expert_gate; modality=rf|hmd.')
        if self.ablation not in ('full', 'spatial_mean', 'last_frame', 'angle_only', 'delay_only', 'abs_change', 'metadata_only'):
            raise ValueError('Unknown representation ablation.')
        if not isinstance(self.hidden, int) or self.hidden < 4:
            raise ValueError('hidden must be an integer >= 4.')
        for value in (self.max_translation_m, self.max_rotation_deg, self.max_logit_delta):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Correction bounds must be finite and positive.')
        if self.max_rotation_deg >= 180:
            raise ValueError('Rotation correction must stay below 180 degrees.')
        return self
