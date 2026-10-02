### step 5
### generate confidence supervision data and split the files

"""
python supervision.py \
  --input-dir "../data/mm23_movement_csv_resampled_30Hz" \
  --output-dir "../data/mm23_movement_csv_resampled_30Hz_with_conf_supervision" \
  --split_after_supervision \
  --split_seed 57 \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --test_ratio 0.15
"""

### label behavior type on data (rotation + translation version, Scheme A Final)

### 是專門服務 confidence estimator 的 supervisor 模組，評分方式同 csv_label_generator，生成的 CSV 會新增 confidence supervision 欄位


import os
import json
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import random
import shutil

# ============================================================
# Confidence-ready Behavior Supervision Generator
# ------------------------------------------------------------
# Goal:
#   1. Preserve Chapter 3 scoring-and-labeling rule
#   2. Add confidence-ready supervision columns
#   3. Make label usage explicit for downstream causal confidence training
#   4. Output per-file summaries for quick sanity check
#
# Notes:
#   - This script keeps the original centered scoring window for compatibility
#     with Chapter 3 statistics and oracle analyses.
#   - For Chapter 5 confidence training, labels are attached to each timestamp
#     and should be consumed by a causal dataset builder that uses past-only
#     pose windows ending at time t.
#   - Unlabeled samples are excluded from confidence training by default.
# ============================================================


# =========================
# Configurations
# =========================

WINDOW_SEC = 1.0
MIN_SAMPLES = 5
ANGLE_UNIT = "deg"

# --- Chapter 3 scoring parameters ---
HI_HEAD_RATE_THRESH_DEG_S = 60.0
EXPLORER_YAW_Q90_DEG_S = 180.0
FOCUSED_YAW_Q90_DEG_S = 60.0  # kept for completeness / debugging
EXPLORER_ZERO_CROSS_S = 3.0
FOCUSED_ZERO_CROSS_S = 1.0    # kept for completeness / debugging
T_SPEED_NORM = 0.2
EXPLORER_SCORE_UPPER = 0.55
FOCUSED_SCORE_LOWER = 0.45

# --- output behavior for confidence training ---
CONFIDENCE_USE_UNLABELED = False
BINARY_TARGET_EXPLORER = 1
BINARY_TARGET_FOCUSED = 0

# --- optional split tagging ---
# only metadata tag; does NOT physically split files
DEFAULT_SPLIT = "unspecified"


# =========================
# Utilities
# =========================

def to_radians(x: np.ndarray) -> np.ndarray:
    return np.deg2rad(x) if ANGLE_UNIT == "deg" else x


def circular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = (~np.isnan(values)) & (~np.isnan(weights))
    if np.sum(mask) == 0:
        return np.nan
    w_sum = np.sum(weights[mask])
    if w_sum == 0:
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / w_sum)


def zero_crossing_count(x: np.ndarray) -> float:
    x = np.asarray(x)
    if len(x) < 2:
        return np.nan
    s = np.sign(x)
    s[s == 0] = 1
    return float(np.sum(s[1:] * s[:-1] < 0))


def safe_percentile(x: np.ndarray, q: float) -> float:
    if len(x) == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanpercentile(x, q))


def label_to_binary_target(label: str) -> float:
    if label == "Explorer":
        return float(BINARY_TARGET_EXPLORER)
    if label == "Focused":
        return float(BINARY_TARGET_FOCUSED)
    return np.nan


def label_to_valid_for_confidence(label: str) -> int:
    if label in ("Explorer", "Focused"):
        return 1
    return 1 if CONFIDENCE_USE_UNLABELED else 0


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_defaults = {
        "Label": "Unlabeled",
        "binary_target": np.nan,
        "valid_for_confidence": 0,
        "confidence_label_time": np.nan,
        "confidence_window_end_time": np.nan,
        "confidence_window_start_time": np.nan,
        "supervision_window_center_time": np.nan,
        "supervision_window_duration_sec": WINDOW_SEC,
        "supervision_is_centered_window": 1,
        "score_valid": 0,
        "head_rate_yaw_q90": np.nan,
        "yaw_zero_cross_per_s": np.nan,
        "head_rate_yaw_hi_prop": np.nan,
        "speed_mean": np.nan,
        "explorer_score": np.nan,
        "delta_rx_deg_s": np.nan,
        "delta_ry_deg_s": np.nan,
        "delta_rz_deg_s": np.nan,
        "split": DEFAULT_SPLIT,
        "scene_name": "",
    }

    for col, default_value in required_defaults.items():
        if col not in df.columns:
            df[col] = default_value
    return df


def summarize_file(df: pd.DataFrame, file_name: str, scene_name: str) -> Dict:
    label_counts = df["Label"].value_counts(dropna=False).to_dict()
    valid_conf = int(df["valid_for_confidence"].sum())
    total_rows = int(len(df))
    valid_score = int(df["score_valid"].sum())

    binary_dist = (
        df.loc[df["valid_for_confidence"] == 1, "binary_target"]
        .value_counts(dropna=False)
        .to_dict()
    )

    summary = {
        "file_name": file_name,
        "scene_name": scene_name,
        "num_rows": total_rows,
        "num_score_valid": valid_score,
        "num_confidence_valid": valid_conf,
        "score_valid_ratio": round(valid_score / total_rows, 6) if total_rows > 0 else 0.0,
        "confidence_valid_ratio": round(valid_conf / total_rows, 6) if total_rows > 0 else 0.0,
        "label_counts": label_counts,
        "binary_target_distribution_on_valid_confidence_rows": binary_dist,
        "explorer_score_mean": (
            float(df["explorer_score"].dropna().mean())
            if df["explorer_score"].notna().any()
            else None
        ),
    }
    return summary

# =========================
# Split Utilities
# =========================

def clean_split_dirs(output_dir: str) -> None:
    """
    清空舊的 train / val / test 子資料夾，避免重跑 supervision 後混到舊切分結果。
    只會刪除 output_dir 底下的 train / val / test，不會刪除 output_dir 本身。
    """
    import shutil

    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split_name)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)


def split_conf_ready_csv_files(
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 57,
    clean_output: bool = True,
    manifest_name: str = "split_manifest.json",
) -> Dict:
    """
    將 supervision 產生的 *_conf_ready.csv 檔案切分成 train / val / test。

    使用方式：
        split_conf_ready_csv_files(output_dir)

    設計原則：
        1. 不改動原本 output_dir 底下的 *_conf_ready.csv
        2. 只把檔案複製到 train / val / test 子資料夾
        3. 使用固定 seed，確保切分可重現
        4. 輸出 split_manifest.json，方便後續確認三個模組使用同一批資料

    注意：
        這是 file-level split，不是 window-level split。
        後續 confidence、Focused branch、Explorer branch 都應該讀取這三個子資料夾。
    """

    if abs(train_ratio + val_ratio + test_ratio - 1.0) >= 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    csv_files = sorted([
        f for f in os.listdir(output_dir)
        if f.endswith("_conf_ready.csv") and os.path.isfile(os.path.join(output_dir, f))
    ])

    if not csv_files:
        print(f"[WARN] No *_conf_ready.csv files found in {output_dir}. Skip split.")
        return {
            "output_dir": output_dir,
            "num_total": 0,
            "train": [],
            "val": [],
            "test": [],
        }

    rng = random.Random(seed)
    rng.shuffle(csv_files)

    total_files = len(csv_files)
    num_train = int(total_files * train_ratio)
    num_val = int(total_files * val_ratio)

    train_files = csv_files[:num_train]
    val_files = csv_files[num_train:num_train + num_val]
    test_files = csv_files[num_train + num_val:]

    if clean_output:
        clean_split_dirs(output_dir)
    else:
        for split_name in ["train", "val", "test"]:
            os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name, files in split_map.items():
        split_dir = os.path.join(output_dir, split_name)

        for fname in files:
            src = os.path.join(output_dir, fname)
            dst = os.path.join(split_dir, fname)
            shutil.copy2(src, dst)

    manifest = {
        "output_dir": output_dir,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "num_total": total_files,
        "num_train": len(train_files),
        "num_val": len(val_files),
        "num_test": len(test_files),
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    manifest_path = os.path.join(output_dir, manifest_name)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Split completed for: {output_dir}")
    print(f"[INFO] Total: {total_files}")
    print(f"[INFO] Train: {len(train_files)} -> {os.path.join(output_dir, 'train')}")
    print(f"[INFO] Val  : {len(val_files)} -> {os.path.join(output_dir, 'val')}")
    print(f"[INFO] Test : {len(test_files)} -> {os.path.join(output_dir, 'test')}")
    print(f"[INFO] Split manifest written to: {manifest_path}")

    return manifest

# =========================
# Core scoring function
# =========================

def score_one_index(
    timer: np.ndarray,
    pos: np.ndarray,
    rot_deg: np.ndarray,
    rot_rad: np.ndarray,
    i: int,
) -> Optional[Dict]:
    t_i = timer[i]

    # Keep Chapter 3-compatible centered supervision window
    mask = (timer >= t_i - WINDOW_SEC / 2) & (timer <= t_i + WINDOW_SEC / 2)
    idx = np.where(mask)[0]

    if len(idx) < MIN_SAMPLES:
        return None

    t_seg = timer[idx]
    pos_seg = pos[idx]
    rot_seg_rad = rot_rad[idx]

    dt = np.diff(t_seg)
    dt[dt <= 0] = np.nan

    if len(dt) == 0 or np.any(np.isnan(dt)):
        return None

    duration = float(t_seg[-1] - t_seg[0])
    if duration <= 0:
        return None

    # ---- Translational speed ----
    disp = np.linalg.norm(np.diff(pos_seg, axis=0), axis=1)
    v = disp / dt
    speed_mean = weighted_mean(v, dt)

    # ---- Yaw-dominant rotational features ----
    # Assuming HeadRY is yaw, aligned with your current Chapter 3 script usage
    d_yaw = circular_diff(rot_seg_rad[1:, 1], rot_seg_rad[:-1, 1])
    yaw_rate = np.abs(d_yaw) / dt
    yaw_rate_signed = d_yaw / dt

    hi_thresh = np.deg2rad(HI_HEAD_RATE_THRESH_DEG_S)
    explorer_yaw_q90 = np.deg2rad(EXPLORER_YAW_Q90_DEG_S)

    head_rate_yaw_q90 = safe_percentile(yaw_rate, 90)
    head_rate_yaw_hi_prop = float(np.mean(yaw_rate > hi_thresh)) if len(yaw_rate) > 0 else np.nan
    switches = zero_crossing_count(yaw_rate_signed)
    yaw_zero_cross_per_s = float(switches / duration) if not np.isnan(switches) else np.nan

    # ---- Instant angular velocities for debugging / future downstream use ----
    if len(rot_seg_rad) > 1:
        d_rot = np.diff(rot_seg_rad, axis=0)
        ang_vel = np.abs(d_rot) / dt[:, None]
        drx = float(np.mean(ang_vel[:, 0]))
        dry = float(np.mean(ang_vel[:, 1]))
        drz = float(np.mean(ang_vel[:, 2]))
    else:
        drx = dry = drz = np.nan

    # ---- Chapter 3 scoring rule ----
    r1 = np.clip(head_rate_yaw_q90 / explorer_yaw_q90, 0, 1) if not np.isnan(head_rate_yaw_q90) else np.nan
    r2 = np.clip(yaw_zero_cross_per_s / EXPLORER_ZERO_CROSS_S, 0, 1) if not np.isnan(yaw_zero_cross_per_s) else np.nan
    r3 = np.clip(head_rate_yaw_hi_prop / 0.5, 0, 1) if not np.isnan(head_rate_yaw_hi_prop) else np.nan

    if np.any(np.isnan([r1, r2, r3, speed_mean])):
        return None

    R = 0.5 * r1 + 0.3 * r2 + 0.2 * r3
    T = np.clip(speed_mean / T_SPEED_NORM, 0, 1)
    S = 0.6 * R + 0.4 * T
    explorer_score = float(1.0 / (1.0 + np.exp(-6 * (S - 0.5))))

    if explorer_score >= EXPLORER_SCORE_UPPER:
        label = "Explorer"
    elif explorer_score <= FOCUSED_SCORE_LOWER:
        label = "Focused"
    else:
        label = "Unlabeled"

    binary_target = label_to_binary_target(label)
    valid_for_confidence = label_to_valid_for_confidence(label)

    result = {
        "Label": label,
        "binary_target": binary_target,
        "valid_for_confidence": valid_for_confidence,
        "confidence_label_time": float(t_i),
        # explicit end-aligned timestamp for downstream causal dataset builder
        "confidence_window_end_time": float(t_i),
        # downstream causal builder should slice [t_i - WINDOW_SEC, t_i]
        "confidence_window_start_time": float(t_i - WINDOW_SEC),
        "supervision_window_center_time": float(t_i),
        "supervision_window_duration_sec": float(WINDOW_SEC),
        "supervision_is_centered_window": 1,
        "score_valid": 1,
        "speed_mean": float(speed_mean),
        "head_rate_yaw_q90": float(head_rate_yaw_q90),
        "yaw_zero_cross_per_s": float(yaw_zero_cross_per_s),
        "head_rate_yaw_hi_prop": float(head_rate_yaw_hi_prop),
        "delta_rx_deg_s": float(np.rad2deg(drx)) if not np.isnan(drx) else np.nan,
        "delta_ry_deg_s": float(np.rad2deg(dry)) if not np.isnan(dry) else np.nan,
        "delta_rz_deg_s": float(np.rad2deg(drz)) if not np.isnan(drz) else np.nan,
        "explorer_score": explorer_score,
    }
    return result


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confidence-ready behavior supervision generator"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the resampled 30 Hz trajectory CSV files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the generated confidence supervision data"
    )
    parser.add_argument(
        "--summary_json",
        default="supervision_summary.json",
        help="Filename of summary JSON written to each output folder"
    )
    parser.add_argument(
        "--split_after_supervision",
        action="store_true",
        help="Split generated *_conf_ready.csv files into train / val / test folders after supervision"
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=57,
        help="Random seed for file-level train / val / test split"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Training file ratio"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation file ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test file ratio"
    )
    parser.add_argument(
        "--keep_old_split",
        action="store_true",
        help="Do not clean existing train / val / test folders before splitting"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    scene_name = "trajectory"
    print(f"\n[INFO] Processing folder: {input_dir}")

    folder_summaries: List[Dict] = []

    for fname in csv_files:
        input_path = os.path.join(input_dir, fname)

        try:
            df = pd.read_csv(input_path)

            if "Timer" not in df.columns:
                raise ValueError("Missing required column: Timer")
            required_pose_cols = ["HeadX", "HeadY", "HeadZ", "HeadRX", "HeadRY", "HeadRZ"]
            missing_pose_cols = [c for c in required_pose_cols if c not in df.columns]
            if missing_pose_cols:
                raise ValueError(f"Missing required pose columns: {missing_pose_cols}")

            df = ensure_required_columns(df)
            df["scene_name"] = scene_name

            timer = df["Timer"].to_numpy(dtype=float)
            pos = df[["HeadX", "HeadY", "HeadZ"]].to_numpy(dtype=float)
            rot_deg = df[["HeadRX", "HeadRY", "HeadRZ"]].to_numpy(dtype=float)
            rot_rad = to_radians(rot_deg)

            n = len(df)

            for i in range(n):
                result = score_one_index(timer, pos, rot_deg, rot_rad, i)
                if result is None:
                    continue

                for k, v in result.items():
                    df.at[i, k] = v

            base = os.path.splitext(fname)[0]
            output_path = os.path.join(output_dir, f"{base}_conf_ready.csv")
            df.to_csv(output_path, index=False, float_format="%.6f")

            summary = summarize_file(df, fname, scene_name)
            folder_summaries.append(summary)

            label_counts = summary["label_counts"]
            print(
                f"[OK] {fname}: "
                f"labels={label_counts}, "
                f"score_valid={summary['num_score_valid']}/{summary['num_rows']}, "
                f"conf_valid={summary['num_confidence_valid']}/{summary['num_rows']}"
            )

        except Exception as e:
            print(f"[FAIL] {fname}: {e}")

    # write per-folder summary
    summary_path = os.path.join(output_dir, args.summary_json)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(folder_summaries, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Summary written to: {summary_path}")

    # split data to train/val/test
    if args.split_after_supervision:
        split_conf_ready_csv_files(
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
            clean_output=not args.keep_old_split,
        )

    print("\n[DONE] Confidence-ready supervision generation completed.")


if __name__ == "__main__":
    main()
