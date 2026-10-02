from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from .data.discover_sessions import SessionInfo
from .data.roi_label import pose_to_roi_mask
from .data.timestamp_alignment import iter_aligned_samples
from .models.baseline_6dof import BinarySigmoidClassifier
from .train import sample_features


def _safe_div(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def evaluate_one(
    sessions: list[SessionInfo],
    horizon_ms: int,
    model_kind: str,
    config: dict,
    model_path: Path,
    partition_centers: np.ndarray,
) -> dict:
    model, standardizer = BinarySigmoidClassifier.load(model_path)
    threshold = float(config["evaluation"].get("threshold", 0.5))
    scene_cfg = config["scene"]

    latencies: list[float] = []
    total = 0
    exact_matches = 0
    cell_correct = 0
    cell_total = 0
    tp = 0
    fp = 0
    fn = 0

    tracemalloc.start()
    for session in sessions:
        for sample in iter_aligned_samples(
            session,
            horizon_ms=horizon_ms,
            pose_history_ms=config["pose_history_ms"],
            pose_history_size=config["pose_history_size"],
            tolerance_ms=config["alignment_tolerance_ms"],
            sample_stride=config.get("sample_stride", 1),
            require_heatmap=(model_kind != "6dof"),
        ):
            y = pose_to_roi_mask(
                sample.target_position,
                sample.target_rotation_deg,
                partition_centers,
                float(scene_cfg["fov_deg"]),
                scene_cfg.get("max_distance_cm"),
            )
            x = sample_features(sample, model_kind, config)[None, :]
            x = standardizer.transform(x)
            start = time.perf_counter()
            pred = model.predict_mask(x, threshold)[0]
            latencies.append((time.perf_counter() - start) * 1000.0)

            y_bool = y.astype(bool)
            pred_bool = pred.astype(bool)
            exact_matches += int(np.array_equal(pred_bool, y_bool))
            cell_correct += int(np.count_nonzero(pred_bool == y_bool))
            cell_total += int(y_bool.size)
            tp += int(np.count_nonzero(pred_bool & y_bool))
            fp += int(np.count_nonzero(pred_bool & ~y_bool))
            fn += int(np.count_nonzero(~pred_bool & y_bool))
            total += 1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)

    return {
        "samples": total,
        "exact_match_accuracy": _safe_div(exact_matches, total),
        "per_cell_accuracy": _safe_div(cell_correct, cell_total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
        "peak_memory_mb": peak / (1024 * 1024),
    }


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

