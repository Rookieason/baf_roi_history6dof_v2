from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R

from model import build_explorer_branch, quaternion_geodesic_distance


POSE_COLS = ["HeadX", "HeadY", "HeadZ", "HeadRX", "HeadRY", "HeadRZ"]
AUX_COLS = [
    "speed_mean",
    "head_rate_yaw_q90",
    "yaw_zero_cross_per_s",
    "head_rate_yaw_hi_prop",
]


def euler_array_deg_to_quaternion_wxyz(euler_deg: np.ndarray) -> np.ndarray:
    r = R.from_euler("xyz", euler_deg, degrees=True)
    q_xyzw = r.as_quat().astype(np.float32)
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1)
    norms = np.linalg.norm(q_wxyz, axis=1, keepdims=True) + 1e-8
    return (q_wxyz / norms).astype(np.float32)


def quaternion_wxyz_to_euler_deg(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = q_wxyz.astype(np.float32)
    q_wxyz = q_wxyz / (np.linalg.norm(q_wxyz) + 1e-8)

    q_xyzw = np.array(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]],
        dtype=np.float32,
    )

    r = R.from_quat(q_xyzw)
    return r.as_euler("xyz", degrees=True).astype(np.float32)


def enforce_quaternion_sign_continuity(quat_array: np.ndarray) -> np.ndarray:
    """
    Enforce sign continuity for quaternion sequence.
    q and -q represent the same rotation, but sign flips are discontinuous for neural input.
    """
    q = quat_array.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def translation_error_m(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred - target))


def quaternion_geo_deg(q_pred: np.ndarray, q_target: np.ndarray) -> float:
    q_pred_t = torch.tensor(q_pred[None, :], dtype=torch.float32)
    q_target_t = torch.tensor(q_target[None, :], dtype=torch.float32)
    d_rad = quaternion_geodesic_distance(q_pred_t, q_target_t).item()
    return float(np.rad2deg(d_rad))


class ExplorerInferenceDataset(Dataset):
    """
    Inference dataset for Explorer branch.

    Training keeps clean Explorer -> Explorer samples.
    Inference keeps every row that can form a full causal window and has p_(t+1).
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int = 30,
        use_timer_sort: bool = True,
        normalize_translation: bool = True,
        normalize_aux: bool = True,
        use_aux: bool = False,
        enforce_quat_continuity: bool = False, #未來想使用 continuity，就必須從 training dataset builder 到 inference dataset builder 全部一致打開，並重新訓練 Explorer branch。
    ) -> None:
        self.csv_path = csv_path
        self.file_name = os.path.basename(csv_path)
        self.seq_len = seq_len
        self.normalize_translation = normalize_translation
        self.normalize_aux = normalize_aux
        self.use_aux = use_aux

        df = pd.read_csv(csv_path)

        required_cols = ["Timer", *POSE_COLS]
        if use_aux:
            required_cols += AUX_COLS

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        if use_timer_sort:
            df = df.sort_values("Timer").reset_index(drop=True)

        df = df.dropna(subset=required_cols).reset_index(drop=True)
        self.df = df

        trans_raw = df[["HeadX", "HeadY", "HeadZ"]].to_numpy(dtype=np.float32)
        euler_raw = df[["HeadRX", "HeadRY", "HeadRZ"]].to_numpy(dtype=np.float32)
        quat_raw = euler_array_deg_to_quaternion_wxyz(euler_raw)

        if enforce_quat_continuity:
            quat_raw = enforce_quaternion_sign_continuity(quat_raw)

        if normalize_translation:
            self.trans_mean = trans_raw.mean(axis=0).astype(np.float32)
            self.trans_std = (trans_raw.std(axis=0) + 1e-6).astype(np.float32)
            trans_out = ((trans_raw - self.trans_mean) / self.trans_std).astype(np.float32)
        else:
            self.trans_mean = np.zeros((3,), dtype=np.float32)
            self.trans_std = np.ones((3,), dtype=np.float32)
            trans_out = trans_raw.astype(np.float32)

        if use_aux:
            aux_raw = df[AUX_COLS].to_numpy(dtype=np.float32)
            if normalize_aux:
                self.aux_mean = aux_raw.mean(axis=0).astype(np.float32)
                self.aux_std = (aux_raw.std(axis=0) + 1e-6).astype(np.float32)
                self.aux_out = ((aux_raw - self.aux_mean) / self.aux_std).astype(np.float32)
            else:
                self.aux_mean = np.zeros((len(AUX_COLS),), dtype=np.float32)
                self.aux_std = np.ones((len(AUX_COLS),), dtype=np.float32)
                self.aux_out = aux_raw.astype(np.float32)
        else:
            self.aux_mean = np.zeros((len(AUX_COLS),), dtype=np.float32)
            self.aux_std = np.ones((len(AUX_COLS),), dtype=np.float32)
            self.aux_out = None

        self.trans_raw = trans_raw.astype(np.float32)
        self.euler_raw = euler_raw.astype(np.float32)
        self.quat_raw = quat_raw.astype(np.float32)
        self.pose7 = np.concatenate([trans_out, quat_raw], axis=1).astype(np.float32)

        self.indices = []
        for end_idx in range(len(df)):
            start_idx = end_idx - seq_len + 1
            target_idx = end_idx + 1
            prev_idx = end_idx - 1

            if start_idx < 0:
                continue
            if prev_idx < 0:
                continue
            if target_idx >= len(df):
                continue

            self.indices.append((start_idx, end_idx, target_idx, prev_idx))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start_idx, end_idx, target_idx, prev_idx = self.indices[idx]

        pose_x = torch.tensor(
            self.pose7[start_idx:end_idx + 1],
            dtype=torch.float32,
        )

        if self.use_aux:
            aux_x = torch.tensor(self.aux_out[end_idx], dtype=torch.float32)
        else:
            aux_x = None

        row_t = self.df.iloc[end_idx]
        row_t1 = self.df.iloc[target_idx]

        meta = {
            "file_name": self.file_name,
            "window_start_index": int(start_idx),
            "end_row_index": int(end_idx),
            "target_row_index": int(target_idx),
            "prev_row_index": int(prev_idx),
            "time_t": float(row_t["Timer"]),
            "time_t1": float(row_t1["Timer"]),
        }

        optional_cols = [
            "Label",
            "binary_target",
            "valid_for_confidence",
            "explorer_score",
            "scene_name",
            "split",
        ]

        for col in optional_cols:
            if col in self.df.columns:
                meta[f"{col}_t"] = row_t[col]
                meta[f"{col}_t1"] = row_t1[col]

        for i, col in enumerate(["HeadX", "HeadY", "HeadZ"]):
            meta[f"current_{col}"] = float(self.trans_raw[end_idx, i])
            meta[f"gt_{col}"] = float(self.trans_raw[target_idx, i])
            meta[f"trans_mean_{col}"] = float(self.trans_mean[i])
            meta[f"trans_std_{col}"] = float(self.trans_std[i])

        for i, col in enumerate(["HeadRX", "HeadRY", "HeadRZ"]):
            meta[f"current_{col}"] = float(self.euler_raw[end_idx, i])
            meta[f"gt_{col}"] = float(self.euler_raw[target_idx, i])

        for i, name in enumerate(["qw", "qx", "qy", "qz"]):
            meta[f"current_{name}"] = float(self.quat_raw[end_idx, i])
            meta[f"gt_{name}"] = float(self.quat_raw[target_idx, i])

        return {
            "pose_x": pose_x,
            "aux_x": aux_x,
            "meta": meta,
        }


def collate_fn(batch):
    pose_x = torch.stack([b["pose_x"] for b in batch], dim=0)

    use_aux = batch[0]["aux_x"] is not None
    aux_x = torch.stack([b["aux_x"] for b in batch], dim=0) if use_aux else None

    metas = [b["meta"] for b in batch]

    return {
        "pose_x": pose_x,
        "aux_x": aux_x,
        "meta": metas,
    }


def build_model_from_checkpoint(ckpt: Dict, device: torch.device):
    args = ckpt.get("args", {})

    use_aux = bool(args.get("use_aux", False))

    model = build_explorer_branch(
        pose_dim=7,
        aux_dim=4 if use_aux else 0,
        hidden_dim=int(args.get("hidden_dim", 128)),
        num_tcn_blocks=int(args.get("num_tcn_blocks", 3)),
        kernel_size=int(args.get("kernel_size", 3)),
        dropout=float(args.get("dropout", 0.1)),
        gamma_rot=float(args.get("gamma_rot", 0.3)),
        eta_tail=float(args.get("eta_tail", 0.1)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, use_aux


@torch.no_grad()
def infer_one_file(
    model,
    csv_path: str,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    use_aux: bool,
) -> pd.DataFrame:
    dataset = ExplorerInferenceDataset(
        csv_path=csv_path,
        seq_len=seq_len,
        use_aux=use_aux,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    rows: List[Dict] = []

    for batch in loader:
        pose_x = batch["pose_x"].to(device)
        aux_x = batch["aux_x"].to(device) if batch["aux_x"] is not None else None
        metas = batch["meta"]

        out = model(pose_x, aux_x)

        pred_tr_norm = out["p_hat_tr"].detach().cpu().numpy()
        delta_tr_norm = out["delta_hat_tr"].detach().cpu().numpy()
        pred_q = out["q_hat_norm"].detach().cpu().numpy()

        for i, meta in enumerate(metas):
            trans_mean = np.array(
                [
                    meta["trans_mean_HeadX"],
                    meta["trans_mean_HeadY"],
                    meta["trans_mean_HeadZ"],
                ],
                dtype=np.float32,
            )
            trans_std = np.array(
                [
                    meta["trans_std_HeadX"],
                    meta["trans_std_HeadY"],
                    meta["trans_std_HeadZ"],
                ],
                dtype=np.float32,
            )

            pred_translation_raw = pred_tr_norm[i] * trans_std + trans_mean
            delta_translation_raw = delta_tr_norm[i] * trans_std

            q_pred = pred_q[i]
            q_pred = q_pred / (np.linalg.norm(q_pred) + 1e-8)
            euler_pred_deg = quaternion_wxyz_to_euler_deg(q_pred)

            gt_translation = np.array(
                [meta["gt_HeadX"], meta["gt_HeadY"], meta["gt_HeadZ"]],
                dtype=np.float32,
            )
            gt_quat = np.array(
                [meta["gt_qw"], meta["gt_qx"], meta["gt_qy"], meta["gt_qz"]],
                dtype=np.float32,
            )

            row = dict(meta)

            for j, col in enumerate(["HeadX", "HeadY", "HeadZ"]):
                row[f"explorer_pred_{col}"] = float(pred_translation_raw[j])
                row[f"explorer_delta_{col}"] = float(delta_translation_raw[j])
                row[f"explorer_pred_norm_{col}"] = float(pred_tr_norm[i, j])
                row[f"explorer_delta_norm_{col}"] = float(delta_tr_norm[i, j])

            for j, name in enumerate(["qw", "qx", "qy", "qz"]):
                row[f"explorer_pred_{name}"] = float(q_pred[j])

            for j, col in enumerate(["HeadRX", "HeadRY", "HeadRZ"]):
                row[f"explorer_pred_{col}"] = float(euler_pred_deg[j])

            row["explorer_translation_error_m"] = translation_error_m(
                pred_translation_raw,
                gt_translation,
            )
            row["explorer_rotation_geo_deg"] = quaternion_geo_deg(
                q_pred,
                gt_quat,
            )

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_glob", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()

    file_paths = sorted(glob.glob(args.input_glob))
    if not file_paths:
        raise FileNotFoundError(f"No files matched pattern: {args.input_glob}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model, use_aux = build_model_from_checkpoint(ckpt, device=device)

    print(f"[INFO] use_aux from checkpoint: {use_aux}")

    all_outputs = []

    for csv_path in file_paths:
        print(f"[INFO] Inference: {os.path.basename(csv_path)}")

        df_out = infer_one_file(
            model=model,
            csv_path=csv_path,
            device=device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            use_aux=use_aux,
        )

        all_outputs.append(df_out)

    final_df = pd.concat(all_outputs, axis=0, ignore_index=True)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(args.output_csv, index=False)

    print(f"[INFO] Saved Explorer inference output to: {args.output_csv}")
    print(f"[INFO] Number of rows: {len(final_df)}")

    print("\n[INFO] Overall raw-space error:")
    print(f"translation_error_m mean : {final_df['explorer_translation_error_m'].mean():.6f}")
    print(f"translation_error_m p50  : {final_df['explorer_translation_error_m'].median():.6f}")
    print(f"translation_error_m p90  : {final_df['explorer_translation_error_m'].quantile(0.90):.6f}")
    print(f"rotation_geo_deg mean    : {final_df['explorer_rotation_geo_deg'].mean():.6f}")
    print(f"rotation_geo_deg p50     : {final_df['explorer_rotation_geo_deg'].median():.6f}")
    print(f"rotation_geo_deg p90     : {final_df['explorer_rotation_geo_deg'].quantile(0.90):.6f}")

    if "Label_t" in final_df.columns:
        print("\n[INFO] Raw-space error by Label_t:")
        grouped = final_df.groupby("Label_t")[
            ["explorer_translation_error_m", "explorer_rotation_geo_deg"]
        ].agg(["count", "mean", "median"])
        print(grouped)


if __name__ == "__main__":
    main()
