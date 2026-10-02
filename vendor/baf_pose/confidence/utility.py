from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


POSE_COLS = ["HeadX", "HeadY", "HeadZ", "HeadRX", "HeadRY", "HeadRZ"]
REQUIRED_COLS = [
    "Timer",
    *POSE_COLS,
    "binary_target",
    "valid_for_confidence",
]


@dataclass
class ConfidenceSample:
    x_window: np.ndarray         # shape: [T, 6]
    y_label: int                 # 0 = Focused, 1 = Explorer
    file_name: str
    end_row_index: int
    time_t: float


class ConfidenceDatasetBuilder:
    def __init__(
        self,
        seq_len: int = 30,
        use_timer_sort: bool = True,
    ) -> None:
        self.seq_len = seq_len
        self.use_timer_sort = use_timer_sort

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

        # keep only rows that have all required pose values
        df = df.dropna(subset=["Timer", *POSE_COLS, "binary_target", "valid_for_confidence"])
        df = df.reset_index(drop=True)

        # normalize types
        df["binary_target"] = df["binary_target"].astype(int)
        df["valid_for_confidence"] = df["valid_for_confidence"].astype(int)

        return df

    def build_from_file(self, file_path: str) -> List[ConfidenceSample]:
        df = pd.read_csv(file_path)
        self._validate_columns(df, file_path)
        df = self._prepare_dataframe(df)

        samples: List[ConfidenceSample] = []
        file_name = os.path.basename(file_path)

        for end_idx in range(len(df)):
            row = df.iloc[end_idx]

            # use only rows explicitly marked as valid
            if int(row["valid_for_confidence"]) != 1:
                continue

            # require a full past-only window [t-T+1, ..., t]
            start_idx = end_idx - self.seq_len + 1
            if start_idx < 0:
                continue

            window_df = df.iloc[start_idx:end_idx + 1]

            # safeguard: window must be contiguous after preprocessing
            if len(window_df) != self.seq_len:
                continue

            x_window = window_df[POSE_COLS].to_numpy(dtype=np.float32)
            y_label = int(row["binary_target"])
            time_t = float(row["Timer"])

            samples.append(
                ConfidenceSample(
                    x_window=x_window,
                    y_label=y_label,
                    file_name=file_name,
                    end_row_index=end_idx,
                    time_t=time_t,
                )
            )

        return samples

    def build_from_glob(self, pattern: str) -> List[ConfidenceSample]:
        file_paths = sorted(glob.glob(pattern))
        if not file_paths:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")

        all_samples: List[ConfidenceSample] = []
        for file_path in file_paths:
            file_samples = self.build_from_file(file_path)
            all_samples.extend(file_samples)

        return all_samples


def summarize_dataset(samples: List[ConfidenceSample]) -> Dict[str, int]:
    num_total = len(samples)
    num_focused = sum(1 for s in samples if s.y_label == 0)
    num_explorer = sum(1 for s in samples if s.y_label == 1)

    return {
        "num_total": num_total,
        "num_focused": num_focused,
        "num_explorer": num_explorer,
    }


def print_dataset_preview(samples: List[ConfidenceSample], num_preview: int = 3) -> None:
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
        print(f"file_name     : {sample.file_name}")
        print(f"end_row_index : {sample.end_row_index}")
        print(f"time_t        : {sample.time_t:.6f}")
        print(f"y_label       : {sample.y_label} ({'Explorer' if sample.y_label == 1 else 'Focused'})")
        print(f"x_window.shape: {sample.x_window.shape}")

        print("first pose row :")
        print(sample.x_window[0])

        print("last pose row  :")
        print(sample.x_window[-1])

        print("window head (first 3 rows):")
        print(sample.x_window[:3])

        print("window tail (last 3 rows):")
        print(sample.x_window[-3:])


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
    args = parser.parse_args()

    builder = ConfidenceDatasetBuilder(seq_len=args.seq_len)
    samples = builder.build_from_glob(args.input_glob)

    print_dataset_preview(samples, num_preview=args.num_preview)


if __name__ == "__main__":
    main()
