from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from scipy.io import loadmat

from .discover_sessions import SessionInfo


@dataclass(frozen=True)
class PoseSeries:
    timestamps_ms: np.ndarray
    positions: np.ndarray
    rotations_deg: np.ndarray


@dataclass(frozen=True)
class HeatmapSeries:
    timestamps_ms: np.ndarray
    frames: np.ndarray

    def nearest_at_or_before(self, timestamp_ms: float, tolerance_ms: float) -> Optional[np.ndarray]:
        if self.timestamps_ms.size == 0:
            return None
        idx = int(np.searchsorted(self.timestamps_ms, timestamp_ms, side="right") - 1)
        if idx < 0 or timestamp_ms - self.timestamps_ms[idx] > tolerance_ms:
            return None
        return self.frames[idx]


@dataclass(frozen=True)
class AlignedSample:
    timestamp_ms: float
    pose_history_timestamps_ms: np.ndarray
    pose_history_positions: np.ndarray
    pose_history_rotations_deg: np.ndarray
    heatmap: Optional[np.ndarray]
    target_position: np.ndarray
    target_rotation_deg: np.ndarray


_FIGURE_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+Z)__")


def parse_timestamp_ms(value: str) -> Optional[float]:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    if "T" in text:
        text = re.sub(r"T(\d{2})-(\d{2})-(\d{2})", r"T\1:\2:\3", text)
        try:
            return datetime.fromisoformat(text).timestamp() * 1000.0
        except ValueError:
            return None
    try:
        return float(text) * 1000.0
    except ValueError:
        return None


def load_pose_series(agent6_csv: str | Path) -> PoseSeries:
    timestamps: list[float] = []
    positions: list[list[float]] = []
    rotations: list[list[float]] = []

    with Path(agent6_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            payload_text = row.get("payload_json") or ""
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            pos = payload.get("pos")
            rot = payload.get("rot_deg") or payload.get("rot") or [0.0, 0.0, 0.0]
            ts_ms = parse_timestamp_ms(payload.get("ts_utc") or row.get("ts_utc") or row.get("recv_unix") or "")
            if ts_ms is None or pos is None or len(pos) != 3:
                continue
            timestamps.append(ts_ms)
            positions.append([float(v) for v in pos[:3]])
            rotations.append([float(v) for v in rot[:3]])

    order = np.argsort(np.asarray(timestamps, dtype=np.float64))
    return PoseSeries(
        timestamps_ms=np.asarray(timestamps, dtype=np.float64)[order],
        positions=np.asarray(positions, dtype=np.float64)[order],
        rotations_deg=np.asarray(rotations, dtype=np.float64)[order],
    )


def _figure_timestamps(figures_dir: str | Path) -> np.ndarray:
    timestamps: list[float] = []
    for png in sorted(Path(figures_dir).glob("*.png")):
        match = _FIGURE_TS_RE.match(png.name)
        if not match:
            continue
        ts_ms = parse_timestamp_ms(match.group(1))
        if ts_ms is not None:
            timestamps.append(ts_ms)
    return np.asarray(timestamps, dtype=np.float64)


def load_heatmap_series(mat_path: str | Path, figures_dir: str | Path) -> HeatmapSeries:
    timestamps = _figure_timestamps(figures_dir)
    if timestamps.size == 0:
        return HeatmapSeries(timestamps_ms=timestamps, frames=np.empty((0, 0), dtype=np.float32))

    mat = loadmat(str(mat_path), variable_names=["spectrum"])
    if "spectrum" not in mat:
        return HeatmapSeries(timestamps_ms=np.empty(0), frames=np.empty((0, 0), dtype=np.float32))
    spectrum = np.asarray(mat["spectrum"], dtype=np.float32)
    if spectrum.ndim != 2:
        return HeatmapSeries(timestamps_ms=np.empty(0), frames=np.empty((0, 0), dtype=np.float32))

    if spectrum.shape[0] >= timestamps.size:
        frames = spectrum[: timestamps.size]
    elif spectrum.shape[1] >= timestamps.size:
        frames = spectrum[:, : timestamps.size].T
    else:
        n = min(spectrum.shape[0], timestamps.size)
        frames = spectrum[:n]
        timestamps = timestamps[:n]
    return HeatmapSeries(timestamps_ms=timestamps, frames=frames)


def nearest_pose_index(timestamps_ms: np.ndarray, target_ms: float, tolerance_ms: float) -> Optional[int]:
    idx = int(np.searchsorted(timestamps_ms, target_ms))
    candidates = [i for i in (idx - 1, idx) if 0 <= i < timestamps_ms.size]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(timestamps_ms[i] - target_ms))
    return best if abs(timestamps_ms[best] - target_ms) <= tolerance_ms else None


def iter_aligned_samples(
    session: SessionInfo,
    horizon_ms: int,
    pose_history_ms: int,
    pose_history_size: int,
    tolerance_ms: int,
    sample_stride: int = 1,
    require_heatmap: bool = True,
) -> Iterator[AlignedSample]:
    poses = load_pose_series(session.agent6_csv)
    if poses.timestamps_ms.size <= pose_history_size:
        return

    heatmaps = load_heatmap_series(session.heatmap_mat, session.heatmap_figures_dir) if require_heatmap else None
    stride = max(int(sample_stride), 1)

    for current_idx in range(pose_history_size - 1, poses.timestamps_ms.size, stride):
        current_ts = poses.timestamps_ms[current_idx]
        start_ts = current_ts - pose_history_ms
        first_idx = int(np.searchsorted(poses.timestamps_ms, start_ts, side="left"))
        if current_idx + 1 - first_idx < pose_history_size:
            continue

        future_idx = nearest_pose_index(poses.timestamps_ms, current_ts + horizon_ms, tolerance_ms)
        if future_idx is None:
            continue

        heatmap = None
        if heatmaps is not None:
            heatmap = heatmaps.nearest_at_or_before(current_ts, tolerance_ms)
            if heatmap is None:
                continue

        idxs = np.linspace(first_idx, current_idx, pose_history_size, dtype=int)
        yield AlignedSample(
            timestamp_ms=current_ts,
            pose_history_timestamps_ms=poses.timestamps_ms[idxs],
            pose_history_positions=poses.positions[idxs],
            pose_history_rotations_deg=poses.rotations_deg[idxs],
            heatmap=heatmap,
            target_position=poses.positions[future_idx],
            target_rotation_deg=poses.rotations_deg[future_idx],
        )


def iter_session_positions(session: SessionInfo) -> Iterator[np.ndarray]:
    poses = load_pose_series(session.agent6_csv)
    for pos in poses.positions:
        if np.all(np.isfinite(pos)):
            yield pos
