from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model import build_confidence_estimator


POSE_COLS = ["HeadX", "HeadY", "HeadZ", "HeadRX", "HeadRY", "HeadRZ"]


class ConfidenceInferenceDataset(Dataset):
    """
    Inference dataset for the confidence module.

    Difference from training dataset:
        Training uses only rows with valid_for_confidence == 1.
        Inference uses every row that can form a full causal window.

    Input:
        x_t = [p_{t-T+1}, ..., p_t], shape [T, 6]

    Output:
        metadata at decision time t.
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int = 30,
        use_timer_sort: bool = True,
    ) -> None:
        self.csv_path = csv_path
        self.file_name = os.path.basename(csv_path)
        self.seq_len = seq_len

        df = pd.read_csv(csv_path)

        required_cols = ["Timer", *POSE_COLS]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        if use_timer_sort:
            df = df.sort_values("Timer").reset_index(drop=True)

        df = df.dropna(subset=["Timer", *POSE_COLS]).reset_index(drop=True)

        self.df = df
        self.indices = []

        for end_idx in range(len(df)):
            start_idx = end_idx - seq_len + 1
            if start_idx < 0:
                continue

            window_df = df.iloc[start_idx:end_idx + 1]
            if len(window_df) != seq_len:
                continue

            self.indices.append((start_idx, end_idx))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start_idx, end_idx = self.indices[idx]

        window_df = self.df.iloc[start_idx:end_idx + 1]
        row = self.df.iloc[end_idx]

        x_window = window_df[POSE_COLS].to_numpy(dtype=np.float32)
        x = torch.tensor(x_window, dtype=torch.float32)

        meta = {
            "file_name": self.file_name,
            "row_index": int(end_idx),
            "window_start_index": int(start_idx),
            "window_end_index": int(end_idx),
            "time_t": float(row["Timer"]),
        }

        # Optional metadata for debugging and analysis.
        optional_cols = [
            "Label",
            "binary_target",
            "valid_for_confidence",
            "explorer_score",
            "scene_name",
            "split",
            "confidence_window_start_time",
            "confidence_window_end_time",
        ]

        for col in optional_cols:
            if col in self.df.columns:
                meta[col] = row[col]

        # Keep current pose at decision time t for alignment/debugging.
        for col in POSE_COLS:
            meta[col] = float(row[col])

        return x, meta


def collate_fn(batch):
    xs = torch.stack([item[0] for item in batch], dim=0)
    metas = [item[1] for item in batch]
    return xs, metas


@torch.no_grad()
def infer_one_file(
    model,
    csv_path: str,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    threshold: float,
) -> pd.DataFrame:
    dataset = ConfidenceInferenceDataset(
        csv_path=csv_path,
        seq_len=seq_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    rows: List[Dict] = []

    model.eval()

    for x, metas in loader:
        x = x.to(device)

        out = model(x)
        logits = out["logits"].detach().cpu().numpy()
        alpha = out["confidence"].detach().cpu().numpy()

        for i, meta in enumerate(metas):
            alpha_i = float(alpha[i])

            row = dict(meta)
            row["confidence_logit"] = float(logits[i])
            row["alpha_t"] = alpha_i
            row["threshold"] = float(threshold)
            row["pred_label"] = int(alpha_i >= threshold)
            row["pred_behavior"] = "Explorer" if alpha_i >= threshold else "Focused"

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_glob", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    file_paths = sorted(glob.glob(args.input_glob))
    if not file_paths:
        raise FileNotFoundError(f"No files matched pattern: {args.input_glob}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)

    # Use checkpoint seq_len if available. Otherwise use command-line seq_len.
    seq_len = int(ckpt.get("seq_len", args.seq_len))

    model = build_confidence_estimator(input_dim=6, seq_len=seq_len).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_outputs = []

    for csv_path in file_paths:
        print(f"[INFO] Inference: {os.path.basename(csv_path)}")

        df_out = infer_one_file(
            model=model,
            csv_path=csv_path,
            device=device,
            seq_len=seq_len,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )

        all_outputs.append(df_out)

    final_df = pd.concat(all_outputs, axis=0, ignore_index=True)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(args.output_csv, index=False)

    print(f"[INFO] Saved inference output to: {args.output_csv}")
    print(f"[INFO] Number of rows: {len(final_df)}")
    print(f"[INFO] alpha_t mean: {final_df['alpha_t'].mean():.6f}")
    print(f"[INFO] alpha_t std : {final_df['alpha_t'].std():.6f}")
    print(f"[INFO] alpha_t min : {final_df['alpha_t'].min():.6f}")
    print(f"[INFO] alpha_t max : {final_df['alpha_t'].max():.6f}")

    if "Label" in final_df.columns:
        print("\n[INFO] alpha_t by Label:")
        print(final_df.groupby("Label")["alpha_t"].describe())

    if "binary_target" in final_df.columns:
        print("\n[INFO] alpha_t by binary_target:")
        print(final_df.groupby("binary_target")["alpha_t"].describe())


if __name__ == "__main__":
    main()
