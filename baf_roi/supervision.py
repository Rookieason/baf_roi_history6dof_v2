"""OFFLINE supervision using the supplied author's unmodified scoring function.

The labels use a CENTERED one-second window, not the model's causal history.
Never call this module to supply alpha or other features during inference.
The original source remains byte-identical under vendor/data_preparation/.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .data import Series

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vendor/data_preparation/supervision.py'
PROTOCOL_ID = 'author_centered_supervision_v2'
LABEL_CODES = {'Focused': 0, 'Explorer': 1, 'Unlabeled': -1}
NUMERIC_FIELDS = (
    'binary_target', 'valid_for_confidence', 'confidence_label_time',
    'confidence_window_end_time', 'confidence_window_start_time',
    'supervision_window_center_time', 'supervision_window_duration_sec',
    'supervision_is_centered_window', 'score_valid', 'speed_mean',
    'head_rate_yaw_q90', 'yaw_zero_cross_per_s', 'head_rate_yaw_hi_prop',
    'delta_rx_deg_s', 'delta_ry_deg_s', 'delta_rz_deg_s', 'explorer_score',
)


@lru_cache(maxsize=1)
def author_module():
    """Import only definitions; the author's filesystem/CLI main is NOT executed."""
    name = '_roi_author_data_preparation_supervision'
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load author supervision from {SOURCE}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def semantic_supervisor_rotations(ypr: np.ndarray) -> np.ndarray:
    """Data adapter ONLY: the author hard-codes HeadRY as semantic yaw.

    Our recorder contract is [yaw,pitch,roll] with yaw about world +Z.
    Supply [pitch,yaw,roll] to this SCORING-ONLY interface. The extra angular
    debug channels retain that semantic ordering. This is NOT an xyz Euler
    transform and must not be used for branch quaternions or confidence input.
    Those keep the existing pose.py [roll,-pitch,yaw] geometric conversion.
    """
    ypr = np.asarray(ypr, np.float64)
    if ypr.ndim != 2 or ypr.shape[1] != 3 or not np.isfinite(ypr).all():
        raise ValueError('Need finite supervisor rotations [N,3] in degrees.')
    return ypr[:, [1, 0, 2]].copy()


def score_resampled_index(timer, poses_m_ypr, index: int):
    """Run EXACT author score_one_index on a supplied resampled sequence.

    Shapes: Timer [N], poses [N,6] = meters and dataset yaw/pitch/roll degrees.
    No label formula, zero handling, thresholds, or feature units are redefined.
    A None result has the original meaning: insufficient/invalid scoring support.
    """
    timer, poses = np.asarray(timer, np.float64), np.asarray(poses_m_ypr, np.float64)
    if timer.ndim != 1 or poses.shape != (len(timer), 6):
        raise ValueError('Need Timer[N] and poses[N,6] for offline supervision.')
    if not 0 <= int(index) < len(timer):
        raise ValueError('Supervision center index is out of range.')
    if not np.isfinite(timer).all() or not np.isfinite(poses).all() or np.any(np.diff(timer) <= 0):
        raise ValueError('Supervision needs finite data and strictly increasing timestamps.')
    module = author_module()
    rotation = semantic_supervisor_rotations(poses[:, 3:])
    return module.score_one_index(timer, poses[:, :3], rotation,
                                  module.to_radians(rotation), int(index))


def centered_pair(series: Series, anchor_index: int):
    """Offline labels centered at the ACTUAL decision t and exact t+1/30.

    Preserve v1's causal model-input/origin protocol for irregular recordings.
    The scoring-only 30-Hz grid is re-anchored at this same observed t. A small
    union covers both centered windows; trimming unused surrounding samples
    does not change the author's score_one_index mask or its formulas.

    At a file/gap boundary we retain only measured segment coverage, as the
    author does, and retain MIN_SAMPLES=5. No padding or extrapolation. Inputs
    to the prediction models are built separately by Series.history(cutoff=t).
    """
    t = float(series.t[int(anchor_index)])
    offsets = np.arange(-16, 18, dtype=np.int64)
    timer = t + offsets / 30.0
    # Use the actual observed coverage. There is no nearest-time label matching.
    keep = (timer >= series.t[0]) & (timer <= series.t[-1])
    offsets, timer = offsets[keep], timer[keep]
    poses = series.at(timer)  # OFFLINE label data; never model input.
    result = []
    for offset in (0, 1):
        where = np.flatnonzero(offsets == offset)
        result.append(score_resampled_index(timer, poses, int(where[0])) if len(where) else None)
    return tuple(result)


def label_code(result) -> int:
    if result is None:
        return -1
    return LABEL_CODES[result['Label']]


def numeric_record(result) -> dict:
    """Preserve author's metadata for audit; this is not a feature vector."""
    if result is not None:
        return {key: float(result[key]) for key in NUMERIC_FIELDS}
    # Defaults from author's ensure_required_columns for an unscorable row.
    values = dict.fromkeys(NUMERIC_FIELDS, np.nan)
    values.update(valid_for_confidence=0., score_valid=0.,
                  supervision_window_duration_sec=1., supervision_is_centered_window=1.)
    return values


def protocol_metadata() -> dict:
    return {
        'id': PROTOCOL_ID,
        'source': 'vendor/data_preparation/supervision.py::score_one_index',
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'supervision_window': '[center-0.5s, center+0.5s], inclusive author mask',
        'model_input_window': '[t-29/30s,...,t] with raw support <= t',
        'sampling_adapter': '30-Hz scoring grid re-anchored to each actual observed decision t',
        'next_label_center': 't+1/30s; may need label support up to t+1/30+0.5s',
        'boundary_policy': 'available within the same segment; author MIN_SAMPLES=5; no extrapolation',
        'zero_rate_policy': 'author: np.sign(0) replaced by +1',
        'yaw_mapping': 'recorder rot_deg[0] -> scoring-only HeadRY; not geometric xyz input',
        'head_rate_yaw_q90_unit': 'rad/s (author storage), not deg/s',
        'delta_debug_units': 'deg/s; unwrapped difference NOT substituted for author np.diff',
        'usage': 'OFFLINE labels and branch selection only; never input or oracle alpha at inference',
    }
