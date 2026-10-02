from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from model import FocusedBranchConfig, FocusedResidualLSTM
from scipy.spatial.transform import Rotation as R


POSE_COLS = ["HeadX", "HeadY", "HeadZ", "HeadRX", "HeadRY", "HeadRZ"]
TR_COLS = ["HeadX", "HeadY", "HeadZ"]
ROT_COLS = ["HeadRX", "HeadRY", "HeadRZ"]

def euler_array_deg_to_quaternion_wxyz(euler_deg: np.ndarray) -> np.ndarray:
    r = R.from_euler("xyz", euler_deg, degrees=True)
    q_xyzw = r.as_quat().astype(np.float32)
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1)
    q_wxyz = q_wxyz / (np.linalg.norm(q_wxyz, axis=1, keepdims=True) + 1e-8)
    return q_wxyz.astype(np.float32)


def quaternion_wxyz_to_euler_deg(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = q_wxyz.astype(np.float32)
    q_wxyz = q_wxyz / (np.linalg.norm(q_wxyz, axis=1, keepdims=True) + 1e-8)

    q_xyzw = np.stack(
        [q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]],
        axis=1,
    ).astype(np.float32)

    return R.from_quat(q_xyzw).as_euler("xyz", degrees=True).astype(np.float32)


def quaternion_geo_deg(q_pred: np.ndarray, q_target: np.ndarray) -> float:
    q_pred = q_pred.astype(np.float32)
    q_target = q_target.astype(np.float32)

    q_pred = q_pred / (np.linalg.norm(q_pred) + 1e-8)
    q_target = q_target / (np.linalg.norm(q_target) + 1e-8)

    inner = abs(float(np.sum(q_pred * q_target)))
    inner = np.clip(inner, 0.0, 1.0)

    return float(np.rad2deg(2.0 * np.arccos(inner)))

def circular_abs_diff_deg(pred_angle: np.ndarray, target_angle: np.ndarray) -> np.ndarray:
    """
    Circular absolute difference for Euler angles in degrees.
    """
    diff = pred_angle - target_angle
    diff = (diff + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def translation_error_m(pred_pose: np.ndarray, target_pose: np.ndarray) -> float:
    """
    Euclidean translation error in meters.
    """
    return float(np.linalg.norm(pred_pose[:3] - target_pose[:3]))


def rotation_mae_deg(pred_pose: np.ndarray, target_pose: np.ndarray) -> float:
    """
    Mean circular rotation error in degrees across RX, RY, RZ.
    """
    err = circular_abs_diff_deg(pred_pose[3:], target_pose[3:])
    return float(np.mean(err))


class FocusedInferenceDataset(Dataset):
    """
    Inference dataset for Focused branch.

    Difference from training dataset:
        Training keeps only clean Focused -> Focused samples.
        Inference keeps every row that can form a full causal window and has p_(t+1).

    Output:
        x_window_norm: [T, 6]
        meta: alignment and denormalization information
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int = 30,
        use_timer_sort: bool = True,
        normalize: bool = True,
    ) -> None:
        self.csv_path = csv_path
        self.file_name = os.path.basename(csv_path)
        self.seq_len = seq_len
        self.normalize = normalize

        df = pd.read_csv(csv_path)

        required_cols = ["Timer", *POSE_COLS]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        if use_timer_sort:
            df = df.sort_values("Timer").reset_index(drop=True)

        df = df.dropna(subset=["Timer", *POSE_COLS]).reset_index(drop=True)
        self.df = df

        # pose_raw = df[POSE_COLS].to_numpy(dtype=np.float32)

        # if normalize:
        #     self.pose_mean = pose_raw.mean(axis=0).astype(np.float32)
        #     self.pose_std = (pose_raw.std(axis=0) + 1e-6).astype(np.float32)
        #     self.pose_norm = ((pose_raw - self.pose_mean) / self.pose_std).astype(np.float32)
        # else:
        #     self.pose_mean = np.zeros((len(POSE_COLS),), dtype=np.float32)
        #     self.pose_std = np.ones((len(POSE_COLS),), dtype=np.float32)
        #     self.pose_norm = pose_raw.astype(np.float32)

        # self.pose_raw = pose_raw.astype(np.float32)
        pose_raw = df[POSE_COLS].to_numpy(dtype=np.float32)
        self.pose_raw = pose_raw.astype(np.float32)

        trans_raw = df[TR_COLS].to_numpy(dtype=np.float32)
        euler_raw = df[ROT_COLS].to_numpy(dtype=np.float32)
        quat_raw = euler_array_deg_to_quaternion_wxyz(euler_raw)

        if normalize:
            self.trans_mean = trans_raw.mean(axis=0).astype(np.float32)
            self.trans_std = (trans_raw.std(axis=0) + 1e-6).astype(np.float32)
            trans_norm = ((trans_raw - self.trans_mean) / self.trans_std).astype(np.float32)
        else:
            self.trans_mean = np.zeros((3,), dtype=np.float32)
            self.trans_std = np.ones((3,), dtype=np.float32)
            trans_norm = trans_raw.astype(np.float32)

        self.quat_raw = quat_raw.astype(np.float32)
        self.pose_norm = np.concatenate([trans_norm, quat_raw], axis=1).astype(np.float32)

        self.indices = []
        for end_idx in range(len(df)):
            start_idx = end_idx - seq_len + 1
            target_idx = end_idx + 1

            if start_idx < 0:
                continue
            if target_idx >= len(df):
                continue

            self.indices.append((start_idx, end_idx, target_idx))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start_idx, end_idx, target_idx = self.indices[idx]

        x_window_norm = self.pose_norm[start_idx:end_idx + 1]
        x = torch.tensor(x_window_norm, dtype=torch.float32)

        row_t = self.df.iloc[end_idx]
        row_t1 = self.df.iloc[target_idx]

        meta = {
            "file_name": self.file_name,
            "window_start_index": int(start_idx),
            "end_row_index": int(end_idx),
            "target_row_index": int(target_idx),
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

        current_pose_raw = self.pose_raw[end_idx]
        target_pose_raw = self.pose_raw[target_idx]

        for i, col in enumerate(POSE_COLS):
            meta[f"current_{col}"] = float(current_pose_raw[i])
            meta[f"gt_{col}"] = float(target_pose_raw[i])
        #     meta[f"pose_mean_{col}"] = float(self.pose_mean[i])
        #     meta[f"pose_std_{col}"] = float(self.pose_std[i])
        for i, col in enumerate(TR_COLS):
            meta[f"trans_mean_{col}"] = float(self.trans_mean[i])
            meta[f"trans_std_{col}"] = float(self.trans_std[i])

        current_quat = self.quat_raw[end_idx]
        target_quat = self.quat_raw[target_idx]

        for j, name in enumerate(["qw", "qx", "qy", "qz"]):
            meta[f"current_{name}"] = float(current_quat[j])
            meta[f"gt_{name}"] = float(target_quat[j])

        return x, meta


def collate_fn(batch):
    xs = torch.stack([item[0] for item in batch], dim=0)
    metas = [item[1] for item in batch]
    return xs, metas


def build_model_from_checkpoint(ckpt: Dict, device: torch.device) -> FocusedResidualLSTM:
    args = ckpt.get("args", {})

    config = FocusedBranchConfig(
        input_dim=7,
        hidden_dim=int(args.get("hidden_dim", 128)),
        num_layers=int(args.get("num_layers", 2)),
        dropout=float(args.get("dropout", 0.1)),
        use_layer_norm=True,
    )

    model = FocusedResidualLSTM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def infer_one_file(
    model: FocusedResidualLSTM,
    csv_path: str,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    normalize: bool,
) -> pd.DataFrame:
    dataset = FocusedInferenceDataset(
        csv_path=csv_path,
        seq_len=seq_len,
        normalize=normalize,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    rows: List[Dict] = []

    for x, metas in loader:
        x = x.to(device)

        # out = model(x)
        # y_hat_norm = out["y_hat"].detach().cpu().numpy()
        # delta_hat_norm = out["delta_hat"].detach().cpu().numpy()
        out = model(x)

        pred_tr_norm = out["p_hat_tr"].detach().cpu().numpy()
        delta_tr_norm = out["delta_hat_tr"].detach().cpu().numpy()
        pred_q = out["q_hat_norm"].detach().cpu().numpy()

        delta_q = out.get("delta_q_hat_norm", None)
        if delta_q is not None:
            delta_q = delta_q.detach().cpu().numpy()

        # for i, meta in enumerate(metas):
        #     pose_mean = np.array(
        #         [meta[f"pose_mean_{col}"] for col in POSE_COLS],
        #         dtype=np.float32,
        #     )
        #     pose_std = np.array(
        #         [meta[f"pose_std_{col}"] for col in POSE_COLS],
        #         dtype=np.float32,
        #     )

        #     pred_norm = y_hat_norm[i]
        #     delta_norm = delta_hat_norm[i]

        #     pred_raw = pred_norm * pose_std + pose_mean
        #     delta_raw = delta_norm * pose_std

        #     gt_raw = np.array(
        #         [meta[f"gt_{col}"] for col in POSE_COLS],
        #         dtype=np.float32,
        #     )

        #     row = dict(meta)

        #     for j, col in enumerate(POSE_COLS):
        #         row[f"focused_pred_norm_{col}"] = float(pred_norm[j])
        #         row[f"focused_delta_norm_{col}"] = float(delta_norm[j])
        #         row[f"focused_pred_{col}"] = float(pred_raw[j])
        #         row[f"focused_delta_{col}"] = float(delta_raw[j])

        #     row["focused_translation_error_m"] = translation_error_m(pred_raw, gt_raw)
        #     row["focused_rotation_mae_deg"] = rotation_mae_deg(pred_raw, gt_raw)

        #     rows.append(row)
        for i, meta in enumerate(metas):
            trans_mean = np.array(
                [meta[f"trans_mean_{col}"] for col in TR_COLS],
                dtype=np.float32,
            )
            trans_std = np.array(
                [meta[f"trans_std_{col}"] for col in TR_COLS],
                dtype=np.float32,
            )

            pred_translation_raw = pred_tr_norm[i] * trans_std + trans_mean
            delta_translation_raw = delta_tr_norm[i] * trans_std

            q_pred = pred_q[i].astype(np.float32)
            q_pred = q_pred / (np.linalg.norm(q_pred) + 1e-8)

            pred_euler = quaternion_wxyz_to_euler_deg(q_pred[None, :])[0]

            pred_pose_raw = np.concatenate(
                [pred_translation_raw, pred_euler],
                axis=0,
            ).astype(np.float32)

            gt_raw = np.array(
                [meta[f"gt_{col}"] for col in POSE_COLS],
                dtype=np.float32,
            )

            gt_quat = np.array(
                [meta[f"gt_{name}"] for name in ["qw", "qx", "qy", "qz"]],
                dtype=np.float32,
            )

            row = dict(meta)

            for j, col in enumerate(TR_COLS):
                row[f"focused_pred_norm_{col}"] = float(pred_tr_norm[i, j])
                row[f"focused_delta_norm_{col}"] = float(delta_tr_norm[i, j])
                row[f"focused_pred_{col}"] = float(pred_translation_raw[j])
                row[f"focused_delta_{col}"] = float(delta_translation_raw[j])
                row[f"gt_{col}"] = float(gt_raw[j])

            for j, col in enumerate(ROT_COLS):
                row[f"focused_pred_{col}"] = float(pred_euler[j])
                row[f"gt_{col}"] = float(gt_raw[j + 3])

            for j, name in enumerate(["qw", "qx", "qy", "qz"]):
                row[f"focused_pred_{name}"] = float(q_pred[j])
                row[f"gt_{name}"] = float(gt_quat[j])

            if delta_q is not None:
                for j, name in enumerate(["qw", "qx", "qy", "qz"]):
                    row[f"focused_delta_q_{name}"] = float(delta_q[i, j])

            row["focused_translation_error_m"] = translation_error_m(pred_pose_raw, gt_raw)
            row["focused_rotation_geo_deg"] = quaternion_geo_deg(q_pred, gt_quat)
            row["focused_rotation_mae_deg"] = rotation_mae_deg(pred_pose_raw, gt_raw)

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_glob", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Use this only if the model was trained without per-file normalization.",
    )

    args = parser.parse_args()

    file_paths = sorted(glob.glob(args.input_glob))
    if not file_paths:
        raise FileNotFoundError(f"No files matched pattern: {args.input_glob}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(ckpt, device=device)

    all_outputs = []

    for csv_path in file_paths:
        print(f"[INFO] Inference: {os.path.basename(csv_path)}")

        df_out = infer_one_file(
            model=model,
            csv_path=csv_path,
            device=device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            normalize=not args.no_normalize,
        )

        all_outputs.append(df_out)

    final_df = pd.concat(all_outputs, axis=0, ignore_index=True)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    final_df.to_csv(args.output_csv, index=False)

    print(f"[INFO] Saved Focused inference output to: {args.output_csv}")
    print(f"[INFO] Number of rows: {len(final_df)}")

    print("\n[INFO] Overall raw-space error:")
    print(f"translation_error_m mean : {final_df['focused_translation_error_m'].mean():.6f}")
    print(f"translation_error_m p50  : {final_df['focused_translation_error_m'].median():.6f}")
    print(f"translation_error_m p90  : {final_df['focused_translation_error_m'].quantile(0.90):.6f}")
    # print(f"rotation_mae_deg mean    : {final_df['focused_rotation_mae_deg'].mean():.6f}")
    # print(f"rotation_mae_deg p50     : {final_df['focused_rotation_mae_deg'].median():.6f}")
    # print(f"rotation_mae_deg p90     : {final_df['focused_rotation_mae_deg'].quantile(0.90):.6f}")
    print(f"rotation_geo_deg mean    : {final_df['focused_rotation_geo_deg'].mean():.6f}")
    print(f"rotation_geo_deg p50     : {final_df['focused_rotation_geo_deg'].median():.6f}")
    print(f"rotation_geo_deg p90     : {final_df['focused_rotation_geo_deg'].quantile(0.90):.6f}")

    print(f"rotation_euler_mae mean  : {final_df['focused_rotation_mae_deg'].mean():.6f}")

    if "Label_t" in final_df.columns:
        print("\n[INFO] Raw-space error by Label_t:")
        # grouped = final_df.groupby("Label_t")[
        #     ["focused_translation_error_m", "focused_rotation_mae_deg"]
        # ].agg(["count", "mean", "median"])
        # print(grouped)
        grouped = final_df.groupby("Label_t")[
            ["focused_translation_error_m", "focused_rotation_geo_deg", "focused_rotation_mae_deg"]
        ].agg(["count", "mean", "median"])
        print(grouped)


if __name__ == "__main__":
    main()
