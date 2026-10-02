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
REQUIRED_COLS = [
    "Timer",
    *POSE_COLS,
    "Label",
]


@dataclass
class FocusedSample:
    x_window: np.ndarray
    target_translation: np.ndarray
    target_quaternion: np.ndarray
    file_name: str
    end_row_index: int
    target_row_index: int
    time_t: float
    time_t1: float
    trans_mean: np.ndarray
    trans_std: np.ndarray

def euler_array_deg_to_quaternion_wxyz(euler_deg: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles [N, 3] in degrees to quaternion [N, 4].
    Output format: [qw, qx, qy, qz].
    """
    r = R.from_euler("xyz", euler_deg, degrees=True)
    q_xyzw = r.as_quat().astype(np.float32)
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1)
    q_wxyz = q_wxyz / (np.linalg.norm(q_wxyz, axis=1, keepdims=True) + 1e-8)
    return q_wxyz.astype(np.float32)

class FocusedDatasetBuilder:
#     def __init__(
#         self,
#         seq_len: int = 30,
#         use_timer_sort: bool = True,
#         normalize: bool = True,
#     ) -> None:
#         self.seq_len = seq_len
#         self.use_timer_sort = use_timer_sort
#         self.normalize = normalize
    def __init__(
        self,
        seq_len: int = 30,
        use_timer_sort: bool = True,
        normalize: bool = True,
        sample_mode: str = "focused_clean",
    ) -> None:
        self.seq_len = seq_len
        self.use_timer_sort = use_timer_sort
        self.normalize = normalize
        self.sample_mode = sample_mode

        if self.sample_mode not in {"focused_clean", "all"}:
            raise ValueError(
                f"Unsupported sample_mode={self.sample_mode}. "
                "Use 'focused_clean' or 'all'."
            )
    def _validate_columns(self, df: pd.DataFrame, file_path: str) -> None:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in {file_path}: {missing}"
            )

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.use_timer_sort:
            df = df.sort_values("Timer").reset_index(drop=True)

        df = df.dropna(subset=["Timer", *POSE_COLS, "Label"])
        df = df.reset_index(drop=True)

        df["Label"] = df["Label"].astype(str)

        return df

    def build_from_file(self, file_path: str) -> List[FocusedSample]:
        df = pd.read_csv(file_path)
        self._validate_columns(df, file_path)
        df = self._prepare_dataframe(df)

        samples: List[FocusedSample] = []
        file_name = os.path.basename(file_path)

        # pose_array = df[POSE_COLS].to_numpy(dtype=np.float32)

        # if self.normalize:
        #     pose_mean = pose_array.mean(axis=0).astype(np.float32)
        #     pose_std = (pose_array.std(axis=0) + 1e-6).astype(np.float32)
        #     pose_array_norm = ((pose_array - pose_mean) / pose_std).astype(np.float32)
        # else:
        #     pose_mean = np.zeros((len(POSE_COLS),), dtype=np.float32)
        #     pose_std = np.ones((len(POSE_COLS),), dtype=np.float32)
        #     pose_array_norm = pose_array.astype(np.float32)
        pose_array = df[POSE_COLS].to_numpy(dtype=np.float32)

        trans_raw = df[["HeadX", "HeadY", "HeadZ"]].to_numpy(dtype=np.float32)
        euler_raw = df[["HeadRX", "HeadRY", "HeadRZ"]].to_numpy(dtype=np.float32)
        quat_raw = euler_array_deg_to_quaternion_wxyz(euler_raw)

        if self.normalize:
            trans_mean = trans_raw.mean(axis=0).astype(np.float32)
            trans_std = (trans_raw.std(axis=0) + 1e-6).astype(np.float32)
            trans_norm = ((trans_raw - trans_mean) / trans_std).astype(np.float32)
        else:
            trans_mean = np.zeros((3,), dtype=np.float32)
            trans_std = np.ones((3,), dtype=np.float32)
            trans_norm = trans_raw.astype(np.float32)

        pose_array_norm = np.concatenate([trans_norm, quat_raw], axis=1).astype(np.float32)

        for end_idx in range(len(df)):
            row = df.iloc[end_idx]

            start_idx = end_idx - self.seq_len + 1
            target_idx = end_idx + 1

            if start_idx < 0:
                continue
            if target_idx >= len(df):
                continue

            target_row = df.iloc[target_idx]

            # # keep only clean Focused -> Focused transitions
            # if row["Label"] != "Focused":
            #     continue
            # if target_row["Label"] != "Focused":
            #     continue
            if self.sample_mode == "focused_clean":
                # keep only clean Focused -> Focused transitions
                if row["Label"] != "Focused":
                    continue
                if target_row["Label"] != "Focused":
                    continue
            elif self.sample_mode == "all":
                # keep all valid transitions for generic motion pre-training
                pass
            else:
                raise ValueError(f"Unsupported sample_mode={self.sample_mode}")

            x_window = pose_array_norm[start_idx:end_idx + 1]
            y_next = pose_array_norm[target_idx]

            if x_window.shape[0] != self.seq_len:
                continue

            time_t = float(df.iloc[end_idx]["Timer"])
            time_t1 = float(df.iloc[target_idx]["Timer"])

            # samples.append(
            #     FocusedSample(
            #         x_window=x_window.astype(np.float32),
            #         y_next=y_next.astype(np.float32),
            #         file_name=file_name,
            #         end_row_index=end_idx,
            #         target_row_index=target_idx,
            #         time_t=time_t,
            #         time_t1=time_t1,
            #         pose_mean=pose_mean.copy(),
            #         pose_std=pose_std.copy(),
            #     )
            # )
            samples.append(
                FocusedSample(
                    x_window=x_window.astype(np.float32),
                    target_translation=trans_norm[target_idx].astype(np.float32),
                    target_quaternion=quat_raw[target_idx].astype(np.float32),
                    file_name=file_name,
                    end_row_index=end_idx,
                    target_row_index=target_idx,
                    time_t=time_t,
                    time_t1=time_t1,
                    trans_mean=trans_mean.copy(),
                    trans_std=trans_std.copy(),
                )
            )

        return samples

    def build_from_glob(self, pattern: str) -> List[FocusedSample]:
        file_paths = sorted(glob.glob(pattern))
        if not file_paths:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")

        all_samples: List[FocusedSample] = []
        for file_path in file_paths:
            file_samples = self.build_from_file(file_path)
            all_samples.extend(file_samples)

        return all_samples

    def build_per_file_stats(self, pattern: str) -> Dict[str, int]:
        file_paths = sorted(glob.glob(pattern))
        if not file_paths:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")

        stats: Dict[str, int] = {}
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            stats[file_name] = len(self.build_from_file(file_path))
        return stats


def summarize_dataset(samples: List[FocusedSample]) -> Dict[str, int]:
    return {
        "num_total": len(samples),
    }


def print_per_file_stats(per_file_stats: Dict[str, int]) -> None:
    print("\n=== Per-file Clean Focused Sample Count ===")
    for file_name, count in per_file_stats.items():
        print(f"{file_name}: {count}")


def print_dataset_preview(samples: List[FocusedSample], num_preview: int = 3) -> None:
    if not samples:
        print("Dataset is empty.")
        return

    print("\n=== Dataset Summary ===")
    stats = summarize_dataset(samples)
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\n=== Preview Samples ===")
    for i, sample in enumerate(samples[:num_preview]):
        print(f"\n--- Sample {i} ---")
        print(f"file_name         : {sample.file_name}")
        print(f"end_row_index     : {sample.end_row_index}")
        print(f"target_row_index  : {sample.target_row_index}")
        print(f"time_t            : {sample.time_t:.6f}")
        print(f"time_t1           : {sample.time_t1:.6f}")
        print(f"x_window.shape    : {sample.x_window.shape}")
        print(f"target_translation.shape: {sample.target_translation.shape}")
        print(f"target_quaternion.shape : {sample.target_quaternion.shape}")

        print("first pose row (normalized):")
        print(sample.x_window[0])

        print("current pose row (p_t, normalized):")
        print(sample.x_window[-1])

        print("target translation (normalized):")
        print(sample.target_translation)

        print("target quaternion [qw, qx, qy, qz]:")
        print(sample.target_quaternion)

        print("window head (first 3 rows, normalized):")
        print(sample.x_window[:3])

        print("window tail (last 3 rows, normalized):")
        print(sample.x_window[-3:])

        print("translation mean:")
        print(sample.trans_mean)

        print("translation standard deviation:")
        print(sample.trans_std)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_glob",
        type=str,
        required=True,
        help="Glob pattern for supervision CSV files, e.g. './data/supervision/*.csv'",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=30,
        help="Causal history window length T",
    )
    parser.add_argument(
        "--num_preview",
        type=int,
        default=3,
        help="Number of preview samples to print",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply per-file normalization to pose channels",
    )
    args = parser.parse_args()

    builder = FocusedDatasetBuilder(
        seq_len=args.seq_len,
        normalize=args.normalize,
    )

    per_file_stats = builder.build_per_file_stats(args.input_glob)
    samples = builder.build_from_glob(args.input_glob)

    print_per_file_stats(per_file_stats)
    print_dataset_preview(samples, num_preview=args.num_preview)


if __name__ == "__main__":
    main()
