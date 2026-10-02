from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader


POSE_COLS = ["HeadX", "HeadY", "HeadZ", "HeadRX", "HeadRY", "HeadRZ"]
TR_COLS = ["HeadX", "HeadY", "HeadZ"]
ROT_COLS = ["HeadRX", "HeadRY", "HeadRZ"]

AUX_COLS = [
    "speed_mean",
    "head_rate_yaw_q90",
    "yaw_zero_cross_per_s",
    "head_rate_yaw_hi_prop",
]


@dataclass
class DecisionSample:
    """
    A single decision-time sample shared by all modules.

    x_conf:
        Confidence input, raw 6DoF pose window.
        Shape: [T, 6]


    x_focused:
        Focused branch input, translation-normalized + quaternion pose window.
        Shape: [T, 7] = [x, y, z, qw, qx, qy, qz]

    x_explorer:
        Explorer branch input, translation-normalized + quaternion pose window.
        Shape: [T, 7] = [x, y, z, qw, qx, qy, qz]

    aux_explorer:
        Optional Explorer auxiliary feature at decision time t.
        Shape: [4] if enabled, otherwise None.

    gt_pose:
        Ground-truth next-step raw 6DoF pose p_(t+1).
        Shape: [6]
    """

    x_conf: np.ndarray
    x_focused: np.ndarray
    x_explorer: np.ndarray
    aux_explorer: Optional[np.ndarray]

    gt_pose: np.ndarray

    file_name: str
    window_start_index: int
    end_row_index: int
    target_row_index: int
    time_t: float
    time_t1: float

    # focused_pose_mean: np.ndarray
    # focused_pose_std: np.ndarray
    focused_trans_mean: np.ndarray
    focused_trans_std: np.ndarray

    explorer_trans_mean: np.ndarray
    explorer_trans_std: np.ndarray
    explorer_aux_mean: np.ndarray
    explorer_aux_std: np.ndarray

    current_pose_raw: np.ndarray
    current_quaternion: np.ndarray
    target_quaternion: np.ndarray

    label_t: Optional[str]
    label_t1: Optional[str]
    explorer_score_t: Optional[float]

    head_rate_yaw_q90_t: Optional[float]
    yaw_zero_cross_per_s_t: Optional[float]
    head_rate_yaw_hi_prop_t: Optional[float]
    head_speed_mean_t: Optional[float]
    score_valid_t: Optional[float]
    scene_name: Optional[str]
    split: Optional[str]


def euler_array_deg_to_quaternion_wxyz(euler_deg: np.ndarray) -> np.ndarray:
    """
    Convert Euler angle array [N, 3] in degrees to quaternion array [N, 4].
    Output quaternion format is [qw, qx, qy, qz].

    This follows the same convention used by the current Explorer inference script.
    """
    r = R.from_euler("xyz", euler_deg, degrees=True)
    q_xyzw = r.as_quat().astype(np.float32)  # scipy returns [x, y, z, w]
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1)
    q_wxyz = q_wxyz / (np.linalg.norm(q_wxyz, axis=1, keepdims=True) + 1e-8)
    return q_wxyz.astype(np.float32)


def enforce_quaternion_sign_continuity(quat_array: np.ndarray) -> np.ndarray:
    """
    Optional quaternion sign-continuity correction.

    Important:
        Keep this disabled by default because the current Explorer model was trained
        without this correction. Enabling it only at inference time changes the input
        distribution and can damage rotation prediction.
    """
    q = quat_array.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


class DecisionInferenceDataset(Dataset):
    """
    Dataset for integrated decision-module inference.

    This dataset does not filter by behavior label.
    It creates one sample for every row that satisfies:

        full causal window [t-T+1, ..., t] exists
        target p_(t+1) exists

    It prepares all module-specific input representations at the same decision time t.
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int = 30,
        use_timer_sort: bool = True,
        normalize_focused_pose: bool = True,
        normalize_explorer_translation: bool = True,
        use_explorer_aux: bool = False,
        normalize_explorer_aux: bool = True,
        enforce_explorer_quat_continuity: bool = False,
    ) -> None:
        self.csv_path = csv_path
        self.file_name = os.path.basename(csv_path)
        self.seq_len = seq_len
        self.use_explorer_aux = use_explorer_aux

        df = pd.read_csv(csv_path)

        required_cols = ["Timer", *POSE_COLS]
        if use_explorer_aux:
            required_cols += AUX_COLS

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        if use_timer_sort:
            df = df.sort_values("Timer").reset_index(drop=True)

        df = df.dropna(subset=required_cols).reset_index(drop=True)
        self.df = df

        self.pose_raw = df[POSE_COLS].to_numpy(dtype=np.float32)

        # Confidence input: raw 6DoF pose.
        self.pose_conf = self.pose_raw.astype(np.float32)

        # Focused input: per-file normalized 6DoF pose.
        # if normalize_focused_pose:
        #     self.focused_pose_mean = self.pose_raw.mean(axis=0).astype(np.float32)
        #     self.focused_pose_std = (self.pose_raw.std(axis=0) + 1e-6).astype(np.float32)
        #     self.pose_focused = (
        #         (self.pose_raw - self.focused_pose_mean) / self.focused_pose_std
        #     ).astype(np.float32)
        # else:
        #     self.focused_pose_mean = np.zeros((6,), dtype=np.float32)
        #     self.focused_pose_std = np.ones((6,), dtype=np.float32)
        #     self.pose_focused = self.pose_raw.astype(np.float32)
        # Focused input: translation-normalized + quaternion rotation.
        # This must match br_focused_residual training/inference format:
        # [x_norm, y_norm, z_norm, qw, qx, qy, qz].
        focused_trans_raw = df[TR_COLS].to_numpy(dtype=np.float32)
        focused_euler_raw = df[ROT_COLS].to_numpy(dtype=np.float32)
        focused_quat_raw = euler_array_deg_to_quaternion_wxyz(focused_euler_raw)

        if normalize_focused_pose:
            self.focused_trans_mean = focused_trans_raw.mean(axis=0).astype(np.float32)
            self.focused_trans_std = (focused_trans_raw.std(axis=0) + 1e-6).astype(np.float32)
            focused_trans_out = (
                (focused_trans_raw - self.focused_trans_mean) / self.focused_trans_std
            ).astype(np.float32)
        else:
            self.focused_trans_mean = np.zeros((3,), dtype=np.float32)
            self.focused_trans_std = np.ones((3,), dtype=np.float32)
            focused_trans_out = focused_trans_raw.astype(np.float32)

        self.pose_focused = np.concatenate(
            [focused_trans_out, focused_quat_raw],
            axis=1,
        ).astype(np.float32)

        # Explorer input: translation-normalized + quaternion rotation.
        trans_raw = df[TR_COLS].to_numpy(dtype=np.float32)
        euler_raw = df[ROT_COLS].to_numpy(dtype=np.float32)

        quat_raw = euler_array_deg_to_quaternion_wxyz(euler_raw)

        if enforce_explorer_quat_continuity:
            quat_raw = enforce_quaternion_sign_continuity(quat_raw)

        if normalize_explorer_translation:
            self.explorer_trans_mean = trans_raw.mean(axis=0).astype(np.float32)
            self.explorer_trans_std = (trans_raw.std(axis=0) + 1e-6).astype(np.float32)
            trans_out = (
                (trans_raw - self.explorer_trans_mean) / self.explorer_trans_std
            ).astype(np.float32)
        else:
            self.explorer_trans_mean = np.zeros((3,), dtype=np.float32)
            self.explorer_trans_std = np.ones((3,), dtype=np.float32)
            trans_out = trans_raw.astype(np.float32)

        self.quat_raw = quat_raw.astype(np.float32)
        self.pose_explorer = np.concatenate([trans_out, quat_raw], axis=1).astype(np.float32)

        # Optional Explorer auxiliary features.
        if use_explorer_aux:
            aux_raw = df[AUX_COLS].to_numpy(dtype=np.float32)

            if normalize_explorer_aux:
                self.explorer_aux_mean = aux_raw.mean(axis=0).astype(np.float32)
                self.explorer_aux_std = (aux_raw.std(axis=0) + 1e-6).astype(np.float32)
                self.aux_explorer = (
                    (aux_raw - self.explorer_aux_mean) / self.explorer_aux_std
                ).astype(np.float32)
            else:
                self.explorer_aux_mean = np.zeros((len(AUX_COLS),), dtype=np.float32)
                self.explorer_aux_std = np.ones((len(AUX_COLS),), dtype=np.float32)
                self.aux_explorer = aux_raw.astype(np.float32)
        else:
            self.explorer_aux_mean = np.zeros((len(AUX_COLS),), dtype=np.float32)
            self.explorer_aux_std = np.ones((len(AUX_COLS),), dtype=np.float32)
            self.aux_explorer = None

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

    def __getitem__(self, idx: int) -> DecisionSample:
        start_idx, end_idx, target_idx = self.indices[idx]

        row_t = self.df.iloc[end_idx]
        row_t1 = self.df.iloc[target_idx]

        x_conf = self.pose_conf[start_idx:end_idx + 1]
        x_focused = self.pose_focused[start_idx:end_idx + 1]
        x_explorer = self.pose_explorer[start_idx:end_idx + 1]

        if self.use_explorer_aux:
            aux_explorer = self.aux_explorer[end_idx]
        else:
            aux_explorer = None

        gt_pose = self.pose_raw[target_idx]
        current_pose_raw = self.pose_raw[end_idx]

        label_t = str(row_t["Label"]) if "Label" in self.df.columns else None
        label_t1 = str(row_t1["Label"]) if "Label" in self.df.columns else None

        if "explorer_score" in self.df.columns:
            explorer_score_t = float(row_t["explorer_score"])
        else:
            explorer_score_t = None

        head_rate_yaw_q90_t = (
            float(row_t["head_rate_yaw_q90"])
            if "head_rate_yaw_q90" in self.df.columns and not pd.isna(row_t["head_rate_yaw_q90"])
            else None
        )

        yaw_zero_cross_per_s_t = (
            float(row_t["yaw_zero_cross_per_s"])
            if "yaw_zero_cross_per_s" in self.df.columns and not pd.isna(row_t["yaw_zero_cross_per_s"])
            else None
        )

        head_rate_yaw_hi_prop_t = (
            float(row_t["head_rate_yaw_hi_prop"])
            if "head_rate_yaw_hi_prop" in self.df.columns and not pd.isna(row_t["head_rate_yaw_hi_prop"])
            else None
        )

        head_speed_mean_t = (
            float(row_t["speed_mean"])
            if "speed_mean" in self.df.columns and not pd.isna(row_t["speed_mean"])
            else None
        )

        score_valid_t = (
            float(row_t["score_valid"])
            if "score_valid" in self.df.columns and not pd.isna(row_t["score_valid"])
            else None
        )

        scene_name = (
            str(row_t["scene_name"])
            if "scene_name" in self.df.columns and not pd.isna(row_t["scene_name"])
            else ""
        )

        split = (
            str(row_t["split"])
            if "split" in self.df.columns and not pd.isna(row_t["split"])
            else ""
        )

        return DecisionSample(
            x_conf=x_conf.astype(np.float32),
            x_focused=x_focused.astype(np.float32),
            x_explorer=x_explorer.astype(np.float32),
            aux_explorer=None if aux_explorer is None else aux_explorer.astype(np.float32),
            gt_pose=gt_pose.astype(np.float32),
            file_name=self.file_name,
            window_start_index=int(start_idx),
            end_row_index=int(end_idx),
            target_row_index=int(target_idx),
            time_t=float(row_t["Timer"]),
            time_t1=float(row_t1["Timer"]),
            # focused_pose_mean=self.focused_pose_mean.copy(),
            # focused_pose_std=self.focused_pose_std.copy(),
            focused_trans_mean=self.focused_trans_mean.copy(),
            focused_trans_std=self.focused_trans_std.copy(),
            explorer_trans_mean=self.explorer_trans_mean.copy(),
            explorer_trans_std=self.explorer_trans_std.copy(),
            explorer_aux_mean=self.explorer_aux_mean.copy(),
            explorer_aux_std=self.explorer_aux_std.copy(),
            current_pose_raw=current_pose_raw.astype(np.float32),
            current_quaternion=self.quat_raw[end_idx].astype(np.float32),
            target_quaternion=self.quat_raw[target_idx].astype(np.float32),
            label_t=label_t,
            label_t1=label_t1,
            explorer_score_t=explorer_score_t,
            head_rate_yaw_q90_t=head_rate_yaw_q90_t,
            yaw_zero_cross_per_s_t=yaw_zero_cross_per_s_t,
            head_rate_yaw_hi_prop_t=head_rate_yaw_hi_prop_t,
            head_speed_mean_t=head_speed_mean_t,
            score_valid_t=score_valid_t,
            scene_name=scene_name,
            split=split,
        )


def decision_collate_fn(batch: List[DecisionSample]) -> Dict:
    x_conf = torch.tensor(
        np.stack([s.x_conf for s in batch], axis=0),
        dtype=torch.float32,
    )

    x_focused = torch.tensor(
        np.stack([s.x_focused for s in batch], axis=0),
        dtype=torch.float32,
    )

    x_explorer = torch.tensor(
        np.stack([s.x_explorer for s in batch], axis=0),
        dtype=torch.float32,
    )

    if batch[0].aux_explorer is None:
        aux_explorer = None
    else:
        aux_explorer = torch.tensor(
            np.stack([s.aux_explorer for s in batch], axis=0),
            dtype=torch.float32,
        )

    gt_pose = torch.tensor(
        np.stack([s.gt_pose for s in batch], axis=0),
        dtype=torch.float32,
    )

    meta = []
    for s in batch:
        meta.append(
            {
                "file_name": s.file_name,
                "window_start_index": s.window_start_index,
                "end_row_index": s.end_row_index,
                "target_row_index": s.target_row_index,
                "time_t": s.time_t,
                "time_t1": s.time_t1,
                "label_t": s.label_t,
                "label_t1": s.label_t1,
                "explorer_score_t": s.explorer_score_t,
                "head_rate_yaw_q90_t": s.head_rate_yaw_q90_t,
                "yaw_zero_cross_per_s_t": s.yaw_zero_cross_per_s_t,
                "head_rate_yaw_hi_prop_t": s.head_rate_yaw_hi_prop_t,
                "head_speed_mean_t": s.head_speed_mean_t,
                "score_valid_t": s.score_valid_t,
                "scene_name": s.scene_name,
                "split": s.split,
                "focused_trans_mean": s.focused_trans_mean,
                "focused_trans_std": s.focused_trans_std,
                "explorer_trans_mean": s.explorer_trans_mean,
                "explorer_trans_std": s.explorer_trans_std,
                "explorer_aux_mean": s.explorer_aux_mean,
                "explorer_aux_std": s.explorer_aux_std,
                "current_pose_raw": s.current_pose_raw,
                "current_quaternion": s.current_quaternion,
                "target_quaternion": s.target_quaternion,
            }
        )

    return {
        "x_conf": x_conf,
        "x_focused": x_focused,
        "x_explorer": x_explorer,
        "aux_explorer": aux_explorer,
        "gt_pose": gt_pose,
        "meta": meta,
    }


def print_batch_preview(batch: Dict) -> None:
    print("\n=== Decision Batch Preview ===")
    print("x_conf.shape      :", tuple(batch["x_conf"].shape))
    print("x_focused.shape   :", tuple(batch["x_focused"].shape))
    print("x_explorer.shape  :", tuple(batch["x_explorer"].shape))

    if batch["aux_explorer"] is None:
        print("aux_explorer      : None")
    else:
        print("aux_explorer.shape:", tuple(batch["aux_explorer"].shape))

    print("gt_pose.shape     :", tuple(batch["gt_pose"].shape))

    print("\n=== First Sample Metadata ===")
    meta = batch["meta"][0]
    for key in [
        "file_name",
        "window_start_index",
        "end_row_index",
        "target_row_index",
        "time_t",
        "time_t1",
        "label_t",
        "label_t1",
        "explorer_score_t",
    ]:
        print(f"{key}: {meta[key]}")

    print("\n=== First Sample Tensor Check ===")
    print("confidence input first row:")
    print(batch["x_conf"][0, 0].numpy())

    print("confidence input last row:")
    print(batch["x_conf"][0, -1].numpy())

    # print("\nfocused input first row:")
    # print(batch["x_focused"][0, 0].numpy())

    # print("focused input last row:")
    # print(batch["x_focused"][0, -1].numpy())
    print("\nfocused input first row [x,y,z,qw,qx,qy,qz]:")
    print(batch["x_focused"][0, 0].numpy())

    print("focused input last row [x,y,z,qw,qx,qy,qz]:")
    print(batch["x_focused"][0, -1].numpy())

    print("\nexplorer input first row [x,y,z,qw,qx,qy,qz]:")
    print(batch["x_explorer"][0, 0].numpy())

    print("explorer input last row [x,y,z,qw,qx,qy,qz]:")
    print(batch["x_explorer"][0, -1].numpy())

    print("\ngt_pose:")
    print(batch["gt_pose"][0].numpy())

    # print("\nfocused_pose_mean:")
    # print(meta["focused_pose_mean"])

    # print("focused_pose_std:")
    # print(meta["focused_pose_std"])
    print("\nfocused_trans_mean:")
    print(meta["focused_trans_mean"])

    print("focused_trans_std:")
    print(meta["focused_trans_std"])

    print("\nexplorer_trans_mean:")
    print(meta["explorer_trans_mean"])

    print("explorer_trans_std:")
    print(meta["explorer_trans_std"])

    print("\ncurrent_quaternion [qw,qx,qy,qz]:")
    print(meta["current_quaternion"])

    print("target_quaternion [qw,qx,qy,qz]:")
    print(meta["target_quaternion"])


def build_decision_loader(
    input_glob: str,
    seq_len: int = 30,
    batch_size: int = 4,
    use_explorer_aux: bool = False,
    enforce_explorer_quat_continuity: bool = False,
) -> DataLoader:
    file_paths = sorted(glob.glob(input_glob))
    if not file_paths:
        raise FileNotFoundError(f"No files matched pattern: {input_glob}")

    datasets = [
        DecisionInferenceDataset(
            csv_path=path,
            seq_len=seq_len,
            use_explorer_aux=use_explorer_aux,
            enforce_explorer_quat_continuity=enforce_explorer_quat_continuity,
        )
        for path in file_paths
    ]

    # Keep implementation simple for checking.
    # For full inference, a ConcatDataset is acceptable because per-sample metadata
    # already stores file-level normalization statistics.
    from torch.utils.data import ConcatDataset

    concat_dataset = ConcatDataset(datasets)

    loader = DataLoader(
        concat_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=decision_collate_fn,
    )

    return loader


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_glob", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_explorer_aux", action="store_true")
    parser.add_argument("--enforce_explorer_quat_continuity", action="store_true")

    args = parser.parse_args()

    loader = build_decision_loader(
        input_glob=args.input_glob,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        use_explorer_aux=args.use_explorer_aux,
        enforce_explorer_quat_continuity=args.enforce_explorer_quat_continuity,
    )

    print("[INFO] Decision loader created.")
    print(f"[INFO] Number of batches: {len(loader)}")

    batch = next(iter(loader))
    print_batch_preview(batch)

    print("\n[DONE] Decision utility input-format check completed.")


if __name__ == "__main__":
    main()