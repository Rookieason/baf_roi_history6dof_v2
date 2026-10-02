from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _stream_config(default_config: dict, stream: dict) -> dict:
    config = dict(default_config)
    stream_config = stream.get("config")
    if isinstance(stream_config, dict):
        config.update(stream_config)
    else:
        config.update({k: v for k, v in stream.items() if k.startswith("obj_transform_")})
    return config


def load_scene_object_centers(adaptive_config_path: str | Path) -> np.ndarray:
    with Path(adaptive_config_path).open("r", encoding="utf-8") as f:
        root = json.load(f)

    default_config = (root.get("defaults") or {}).get("config") or {}
    streams = root.get("streams") or root.get("objects") or []
    centers: list[list[float]] = []

    for stream in streams:
        if not isinstance(stream, dict):
            continue
        config = _stream_config(default_config, stream)
        centers.append(
            [
                float(config.get("obj_transform_pos_x_cm", 0.0)),
                float(config.get("obj_transform_pos_y_cm", 0.0)),
                float(config.get("obj_transform_pos_z_cm", 0.0)),
            ]
        )

    if not centers:
        raise RuntimeError(f"No streams found in adaptive config: {adaptive_config_path}")
    return np.asarray(centers, dtype=np.float64)


def _size_xyz(object_size_cm: float | list[float] | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    size = np.asarray(object_size_cm, dtype=np.float64)
    if size.ndim == 0:
        size = np.repeat(float(size), 3)
    if size.shape != (3,):
        raise ValueError("object_size_cm must be a scalar or a length-3 sequence.")
    if np.any(size <= 0.0) or not np.all(np.isfinite(size)):
        raise ValueError("object_size_cm must contain positive finite values.")
    return size


def generate_partition_centers(
    object_center: np.ndarray,
    object_size_cm: float | list[float] | tuple[float, float, float] | np.ndarray,
    grid_dim: int,
) -> np.ndarray:
    grid_dim = int(grid_dim)
    if grid_dim <= 0:
        raise ValueError("grid_dim must be positive.")

    center = np.asarray(object_center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("object_center must have shape (3,).")

    size = _size_xyz(object_size_cm)
    mins = center - 0.5 * size
    cell = size / float(grid_dim)
    axes = [mins[axis] + (np.arange(grid_dim, dtype=np.float64) + 0.5) * cell[axis] for axis in range(3)]

    centers = []
    for iz in range(grid_dim):
        for iy in range(grid_dim):
            for ix in range(grid_dim):
                centers.append([axes[0][ix], axes[1][iy], axes[2][iz]])
    return np.asarray(centers, dtype=np.float64)


def build_scene_partition_centers(
    adaptive_config_path: str | Path,
    grid_dim: int,
    object_size_cm: float | list[float] | tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    object_centers = load_scene_object_centers(adaptive_config_path)
    parts = [generate_partition_centers(center, object_size_cm, grid_dim) for center in object_centers]
    return np.vstack(parts).astype(np.float64)


def _rotation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    roll = np.deg2rad(roll_deg)

    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)

    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def camera_forward_vector(rotation_deg: np.ndarray) -> np.ndarray:
    yaw_deg, pitch_deg, roll_deg = np.asarray(rotation_deg, dtype=np.float64)[:3]
    forward = _rotation_matrix(-pitch_deg, yaw_deg, roll_deg) @ np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    norm = np.linalg.norm(forward)
    return forward / norm if norm > 1e-12 else np.asarray([1.0, 0.0, 0.0], dtype=np.float64)


def pose_to_roi_mask(
    future_position: np.ndarray,
    future_rotation_deg: np.ndarray,
    partition_centers: np.ndarray,
    fov_deg: float,
    max_distance_cm: float | None = None,
) -> np.ndarray:
    centers = np.asarray(partition_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("partition_centers must have shape (N, 3).")

    position = np.asarray(future_position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("future_position must have shape (3,).")

    fov = float(fov_deg)
    if not np.isfinite(fov) or fov <= 0.0 or fov >= 180.0:
        raise ValueError("fov_deg must be in (0, 180).")

    direction = centers - position
    distance = np.linalg.norm(direction, axis=1)
    valid = distance > 1e-9
    unit = np.zeros_like(direction)
    unit[valid] = direction[valid] / distance[valid, None]

    forward = camera_forward_vector(future_rotation_deg)
    cos_half_fov = np.cos(np.deg2rad(0.5 * fov))
    inside = np.zeros(centers.shape[0], dtype=bool)
    inside[valid] = unit[valid] @ forward >= cos_half_fov
    inside[~valid] = True

    if max_distance_cm is not None:
        inside &= distance <= float(max_distance_cm)

    return inside.astype(np.float32)
