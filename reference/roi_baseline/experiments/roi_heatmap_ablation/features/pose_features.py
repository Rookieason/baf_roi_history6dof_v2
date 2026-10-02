from __future__ import annotations

import numpy as np


def pose_history_features(timestamps_ms: np.ndarray, positions: np.ndarray, rotations_deg: np.ndarray) -> np.ndarray:
    ts = timestamps_ms.astype(np.float64)
    pos = positions.astype(np.float64)
    rot = rotations_deg.astype(np.float64)
    dt = np.diff(ts, prepend=ts[0]) / 1000.0
    dt_safe = np.where(dt <= 1e-6, 1.0, dt)

    vel = np.vstack([np.zeros(3), np.diff(pos, axis=0) / dt_safe[1:, None]])
    ang = np.vstack([np.zeros(3), np.diff(rot, axis=0) / dt_safe[1:, None]])
    rel_pos = pos - pos[-1]
    rel_rot = rot - rot[-1]
    rel_time = (ts - ts[-1])[:, None] / 1000.0

    return np.hstack([rel_time, rel_pos, rel_rot, vel, ang]).reshape(-1).astype(np.float32)

