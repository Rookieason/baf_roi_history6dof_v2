from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


POSE_COLS = ["HeadX", "HeadY", "HeadZ", "HeadRX", "HeadRY", "HeadRZ"]
AUX_COLS = [
    "speed_mean",
    "head_rate_yaw_q90",
    "yaw_zero_cross_per_s",
    "head_rate_yaw_hi_prop",
]
REQUIRED_COLS = [
    "Timer",
    *POSE_COLS,
    *AUX_COLS,
    "Label",
]


@dataclass
class ExplorerSample:
    pose_x: np.ndarray              # [T, 7] = [x, y, z, qw, qx, qy, qz]
    aux_x: np.ndarray               # [4]
    target_translation: np.ndarray  # [3]
    target_quaternion: np.ndarray   # [4] -> [qw, qx, qy, qz]
    current_quaternion: np.ndarray  # [4]
    prev_quaternion: np.ndarray     # [4]
    file_name: str
    end_row_index: int
    target_row_index: int
    time_t: float
    time_t1: float
    trans_mean: np.ndarray          # [3]
    trans_std: np.ndarray           # [3]
    aux_mean: np.ndarray            # [4]
    aux_std: np.ndarray             # [4]


def euler_deg_to_quaternion_wxyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """
    Convert Euler angles in degrees to quaternion [qw, qx, qy, qz].

    scipy Rotation.as_quat() returns [x, y, z, w],
    so we reorder it to [w, x, y, z].
    """
    r = R.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True)
    q_xyzw = r.as_quat().astype(np.float32)  # [x, y, z, w]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

    norm = np.linalg.norm(q_wxyz)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q_wxyz / norm


def euler_array_deg_to_quaternion_wxyz(euler_deg: np.ndarray) -> np.ndarray:
    """
    Convert Euler array [N, 3] in degrees to quaternion array [N, 4] in [qw, qx, qy, qz].
    """
    r = R.from_euler("xyz", euler_deg, degrees=True)
    q_xyzw = r.as_quat().astype(np.float32)  # [N, 4] -> [x, y, z, w]
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1)
    norms = np.linalg.norm(q_wxyz, axis=1, keepdims=True) + 1e-8
    return (q_wxyz / norms).astype(np.float32)


def quaternion_geodesic_distance_np(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2) + 1e-8)
    inner = np.abs(np.dot(q1, q2))
    inner = np.clip(inner, 0.0, 1.0)
    return float(2.0 * np.arccos(inner))


class ExplorerDatasetBuilder:
    def __init__(
        self,
        seq_len: int = 30,
        use_timer_sort: bool = True,
        normalize_translation: bool = True,
        normalize_aux: bool = True,
    ) -> None:
        self.seq_len = seq_len
        self.use_timer_sort = use_timer_sort
        self.normalize_translation = normalize_translation
        self.normalize_aux = normalize_aux

    def _validate_columns(self, df: pd.DataFrame, file_path: str) -> None:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {file_path}: {missing}")

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.use_timer_sort:
            df = df.sort_values("Timer").reset_index(drop=True)

        df = df.dropna(subset=REQUIRED_COLS)
        df = df.reset_index(drop=True)
        df["Label"] = df["Label"].astype(str)
        return df

    def build_from_file(self, file_path: str) -> List[ExplorerSample]:
        df = pd.read_csv(file_path)
        self._validate_columns(df, file_path)
        df = self._prepare_dataframe(df)

        file_name = os.path.basename(file_path)
        samples: List[ExplorerSample] = []

        trans_array = df[["HeadX", "HeadY", "HeadZ"]].to_numpy(dtype=np.float32)
        euler_array = df[["HeadRX", "HeadRY", "HeadRZ"]].to_numpy(dtype=np.float32)
        quat_array = euler_array_deg_to_quaternion_wxyz(euler_array)   # [N, 4]
        aux_array = df[AUX_COLS].to_numpy(dtype=np.float32)

        if self.normalize_translation:
            trans_mean = trans_array.mean(axis=0).astype(np.float32)
            trans_std = (trans_array.std(axis=0) + 1e-6).astype(np.float32)
            trans_array_out = ((trans_array - trans_mean) / trans_std).astype(np.float32)
        else:
            trans_mean = np.zeros((3,), dtype=np.float32)
            trans_std = np.ones((3,), dtype=np.float32)
            trans_array_out = trans_array

        if self.normalize_aux:
            aux_mean = aux_array.mean(axis=0).astype(np.float32)
            aux_std = (aux_array.std(axis=0) + 1e-6).astype(np.float32)
            aux_array_out = ((aux_array - aux_mean) / aux_std).astype(np.float32)
        else:
            aux_mean = np.zeros((len(AUX_COLS),), dtype=np.float32)
            aux_std = np.ones((len(AUX_COLS),), dtype=np.float32)
            aux_array_out = aux_array

        # final pose representation per timestep: [x, y, z, qw, qx, qy, qz]
        pose7_array = np.concatenate([trans_array_out, quat_array], axis=1).astype(np.float32)

        for end_idx in range(len(df)):
            start_idx = end_idx - self.seq_len + 1
            target_idx = end_idx + 1
            prev_idx = end_idx - 1

            if start_idx < 0 or prev_idx < 0 or target_idx >= len(df):
                continue

            row_t = df.iloc[end_idx]
            row_t1 = df.iloc[target_idx]

            # clean Explorer -> Explorer transitions
            if row_t["Label"] != "Explorer":
                continue
            if row_t1["Label"] != "Explorer":
                continue

            pose_x = pose7_array[start_idx:end_idx + 1]
            if pose_x.shape[0] != self.seq_len:
                continue

            aux_x = aux_array_out[end_idx]

            current_translation = trans_array_out[end_idx]
            target_translation = trans_array_out[target_idx]

            prev_quaternion = quat_array[prev_idx]
            current_quaternion = quat_array[end_idx]
            target_quaternion = quat_array[target_idx]

            sample = ExplorerSample(
                pose_x=pose_x.astype(np.float32),
                aux_x=aux_x.astype(np.float32),
                target_translation=target_translation.astype(np.float32),
                target_quaternion=target_quaternion.astype(np.float32),
                current_quaternion=current_quaternion.astype(np.float32),
                prev_quaternion=prev_quaternion.astype(np.float32),
                file_name=file_name,
                end_row_index=end_idx,
                target_row_index=target_idx,
                time_t=float(df.iloc[end_idx]["Timer"]),
                time_t1=float(df.iloc[target_idx]["Timer"]),
                trans_mean=trans_mean.copy(),
                trans_std=trans_std.copy(),
                aux_mean=aux_mean.copy(),
                aux_std=aux_std.copy(),
            )
            samples.append(sample)

        return samples

    def build_from_glob(self, pattern: str) -> List[ExplorerSample]:
        file_paths = sorted(glob.glob(pattern))
        if not file_paths:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")

        all_samples: List[ExplorerSample] = []
        for file_path in file_paths:
            all_samples.extend(self.build_from_file(file_path))
        return all_samples

    def build_per_file_stats(self, pattern: str) -> Dict[str, int]:
        file_paths = sorted(glob.glob(pattern))
        if not file_paths:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")

        stats: Dict[str, int] = {}
        for file_path in file_paths:
            stats[os.path.basename(file_path)] = len(self.build_from_file(file_path))
        return stats


def summarize_dataset(samples: List[ExplorerSample]) -> Dict[str, int]:
    return {"num_total": len(samples)}


def print_per_file_stats(per_file_stats: Dict[str, int]) -> None:
    print("\n=== Per-file Clean Explorer Sample Count ===")
    for file_name, count in per_file_stats.items():
        print(f"{file_name}: {count}")


def print_dataset_preview(samples: List[ExplorerSample], num_preview: int = 3) -> None:
    if not samples:
        print("Dataset is empty.")
        return

    print("\n=== Dataset Summary ===")
    stats = summarize_dataset(samples)
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\n=== Preview Samples ===")
    for i, s in enumerate(samples[:num_preview]):
        print(f"\n--- Sample {i} ---")
        print(f"file_name           : {s.file_name}")
        print(f"end_row_index       : {s.end_row_index}")
        print(f"target_row_index    : {s.target_row_index}")
        print(f"time_t              : {s.time_t:.6f}")
        print(f"time_t1             : {s.time_t1:.6f}")
        print(f"pose_x.shape        : {s.pose_x.shape}")
        print(f"aux_x.shape         : {s.aux_x.shape}")
        print(f"target_translation  : {s.target_translation}")
        print(f"target_quaternion   : {s.target_quaternion}")
        print(f"current_quaternion  : {s.current_quaternion}")
        print(f"prev_quaternion     : {s.prev_quaternion}")

        print("pose window head (first 3 rows):")
        print(s.pose_x[:3])

        print("pose window tail (last 3 rows):")
        print(s.pose_x[-3:])

        print("aux_x (normalized if enabled):")
        print(s.aux_x)

        print("quaternion norms:")
        print(
            "prev={:.6f}, current={:.6f}, target={:.6f}".format(
                np.linalg.norm(s.prev_quaternion),
                np.linalg.norm(s.current_quaternion),
                np.linalg.norm(s.target_quaternion),
            )
        )

        d_prev_curr = quaternion_geodesic_distance_np(s.prev_quaternion, s.current_quaternion)
        d_curr_tgt = quaternion_geodesic_distance_np(s.current_quaternion, s.target_quaternion)
        print("quaternion geodesic distance (rad):")
        print(f"prev->curr = {d_prev_curr:.6f}, curr->target = {d_curr_tgt:.6f}")

        print("translation mean/std:")
        print(s.trans_mean)
        print(s.trans_std)

        print("aux mean/std:")
        print(s.aux_mean)
        print(s.aux_std)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_glob",
        type=str,
        required=True,
        help="Glob pattern for supervision CSV files",
    )
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--num_preview", type=int, default=3)
    parser.add_argument("--no_normalize_translation", action="store_true")
    parser.add_argument("--no_normalize_aux", action="store_true")
    args = parser.parse_args()

    builder = ExplorerDatasetBuilder(
        seq_len=args.seq_len,
        normalize_translation=not args.no_normalize_translation,
        normalize_aux=not args.no_normalize_aux,
    )

    per_file_stats = builder.build_per_file_stats(args.input_glob)
    samples = builder.build_from_glob(args.input_glob)

    print_per_file_stats(per_file_stats)
    print_dataset_preview(samples, num_preview=args.num_preview)


if __name__ == "__main__":
    main()