from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def infer_heatmap_shape(flat_size: int) -> tuple[int, int]:
    root = int(math.sqrt(flat_size))
    for h in range(root, 0, -1):
        if flat_size % h == 0:
            return h, flat_size // h
    return 1, flat_size


def _as_grid(heatmap: np.ndarray, configured_shape: Optional[Sequence[int]]) -> np.ndarray:
    arr = np.asarray(heatmap, dtype=np.float32).squeeze()
    if arr.ndim == 2:
        return arr
    flat = arr.reshape(-1)
    if configured_shape:
        h, w = int(configured_shape[0]), int(configured_shape[1])
        if h * w == flat.size:
            return flat.reshape(h, w)
    h, w = infer_heatmap_shape(flat.size)
    return flat.reshape(h, w)


def heatmap_compact_features(
    heatmap: np.ndarray,
    configured_shape: Optional[Sequence[int]] = None,
    top_k: int = 5,
    pooled_shape: Sequence[int] = (8, 8),
) -> np.ndarray:
    grid = np.nan_to_num(_as_grid(heatmap, configured_shape), copy=False)
    h, w = grid.shape
    flat = grid.reshape(-1)
    total = float(np.sum(np.maximum(flat, 0.0)))
    denom = total if total > 1e-9 else float(flat.size)

    top_k = int(top_k)
    if flat.size >= top_k:
        top_idx = np.argpartition(flat, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
    else:
        top_idx = np.argsort(flat)[::-1]
    top_features: list[float] = []
    for idx in top_idx[:top_k]:
        y, x = divmod(int(idx), w)
        top_features.extend([y / max(h - 1, 1), x / max(w - 1, 1), float(flat[idx])])
    while len(top_features) < top_k * 3:
        top_features.extend([0.0, 0.0, 0.0])

    yy, xx = np.indices((h, w), dtype=np.float32)
    weights = np.maximum(grid, 0.0)
    com_y = float(np.sum(yy * weights) / denom / max(h - 1, 1))
    com_x = float(np.sum(xx * weights) / denom / max(w - 1, 1))

    ph, pw = int(pooled_shape[0]), int(pooled_shape[1])
    pooled = np.array(
        [cell.mean() for row in np.array_split(grid, ph, axis=0) for cell in np.array_split(row, pw, axis=1)],
        dtype=np.float32,
    )

    stats = np.array([total, float(flat.mean()), float(flat.std()), float(flat.max(initial=0.0)), com_y, com_x], dtype=np.float32)
    return np.concatenate([np.asarray(top_features, dtype=np.float32), stats, pooled.astype(np.float32)])

