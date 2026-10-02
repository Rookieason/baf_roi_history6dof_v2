"""Dataset coordinates and geometric conversions, not learned features."""
import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from .upstream_ops import slerp_quaternion_wxyz


def finite_array(x, shape_tail=None):
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError('Pose/geometry contains NaN or infinity.')
    if shape_tail is not None and x.shape[-len(shape_tail):] != tuple(shape_tail):
        raise ValueError(f'Expected trailing shape {shape_tail}, got {x.shape}.')
    return x


def ypr_to_xyz(ypr):
    """R = Rz(yaw) Ry(-pitch) Rx(roll), matching the supplied ROI source."""
    ypr = finite_array(ypr, (3,))
    return np.stack((ypr[..., 2], -ypr[..., 1], ypr[..., 0]), axis=-1)


def ypr_to_quat(ypr):
    ypr = finite_array(ypr, (3,))
    shape = ypr.shape[:-1]
    q = Rotation.from_euler('xyz', ypr_to_xyz(ypr).reshape(-1, 3), degrees=True).as_quat()
    return q[:, [3, 0, 1, 2]].reshape(*shape, 4)


def quat_to_ypr(q):
    q = finite_array(q, (4,))
    shape = q.shape[:-1]
    q = q.reshape(-1, 4)
    if np.any(np.linalg.norm(q, axis=-1) < 1e-10):
        raise ValueError('Zero quaternion.')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Gimbal lock detected')
        xyz = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)
    return np.stack((xyz[:, 2], -xyz[:, 1], xyz[:, 0]), axis=-1).reshape(*shape, 3)


def model_inputs(history, stats):
    """history: [...,30,6], meters + [yaw,pitch,roll] degrees."""
    x = finite_array(history, (30, 6))
    conf = np.concatenate((x[..., :3], ypr_to_xyz(x[..., 3:])), axis=-1)
    pos = (x[..., :3] - np.asarray(stats['mean'])) / np.asarray(stats['std'])
    pose7 = np.concatenate((pos, ypr_to_quat(x[..., 3:])), axis=-1)
    return conf.astype(np.float32), pose7.astype(np.float32)


def interpolate_predictions(positions, quaternions, horizons_ms, rate=30):
    """Index zero is observed pose; indices 1..K are predictions only."""
    p, q = np.asarray(positions), np.asarray(quaternions)
    # [B, K+1, 3/4]; fractional horizons are never rounded to the nearest frame.
    ps, qs = [], []
    for h in horizons_ms:
        step = float(h) * rate / 1000
        lo = int(np.floor(step + 1e-10))
        hi = int(np.ceil(step - 1e-10))
        if hi >= p.shape[1]:
            raise ValueError('Insufficient rollout steps.')
        w = max(0., min(1., step - lo))
        ps.append((1 - w) * p[:, lo] + w * p[:, hi])
        qs.append(slerp_quaternion_wxyz(q[:, lo], q[:, hi], np.full(len(p), w)))
    return np.stack(ps, axis=1), np.stack(qs, axis=1)


def quaternion_error_deg(a, b):
    """Evaluation metric: zero for identical rotations, unlike clamped TRAIN loss."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    # atan2 avoids acos cancellation at small errors and respects q == -q.
    b = np.where((a * b).sum(axis=-1, keepdims=True) < 0, -b, b)
    return np.rad2deg(4 * np.arctan2(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1)))
