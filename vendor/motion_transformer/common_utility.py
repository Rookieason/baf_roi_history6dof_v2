# sota_baseline/common/common_utility.py
from __future__ import annotations

import argparse
import glob
import json
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

AUX_COLS_DEFAULT = [
    "speed_mean",
    "head_rate_yaw_q90",
    "yaw_zero_cross_per_s",
    "head_rate_yaw_hi_prop",
]


@dataclass
class PoseWindowSample:
    pose_x: np.ndarray                  # [T, 7] = [x, y, z, qw, qx, qy, qz]
    target_translation: np.ndarray      # [3], normalized translation
    target_quaternion: np.ndarray       # [4], raw unit quaternion, wxyz
    current_translation: np.ndarray     # [3], normalized translation
    current_quaternion: np.ndarray      # [4], raw unit quaternion, wxyz
    prev_quaternion: np.ndarray         # [4], raw unit quaternion, wxyz

    aux_x: Optional[np.ndarray]         # [A], optional normalized aux at time t

    file_name: str
    window_start_index: int
    end_row_index: int
    target_row_index: int
    time_t: float
    time_t1: float
    label_t: str
    label_t1: str
    explorer_score_t: float

    trans_mean: np.ndarray              # [3], per-file mean
    trans_std: np.ndarray               # [3], per-file std
    aux_mean: Optional[np.ndarray]      # [A], per-file mean
    aux_std: Optional[np.ndarray]       # [A], per-file std

    # new extra info
    head_rate_yaw_q90_t: float
    yaw_zero_cross_per_s_t: float
    head_rate_yaw_hi_prop_t: float
    head_speed_mean_t: float
    score_valid_t: float
    scene_name: str
    split: str


def euler_array_deg_to_quaternion_wxyz(euler_deg: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles [N, 3] in degrees to unit quaternion [N, 4].
    Output quaternion format is [qw, qx, qy, qz].
    """
    euler_deg = np.asarray(euler_deg, dtype=np.float32)

    if euler_deg.ndim != 2 or euler_deg.shape[1] != 3:
        raise ValueError(f"Expected euler_deg with shape [N, 3], got {euler_deg.shape}")

    r = R.from_euler("xyz", euler_deg, degrees=True)
    q_xyzw = r.as_quat().astype(np.float32)
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1)
    q_wxyz = normalize_quaternion_np(q_wxyz)
    return q_wxyz.astype(np.float32)


def normalize_quaternion_np(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (norm + eps)


def normalize_quaternion_torch(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)


def quaternion_geodesic_distance_np(q_pred: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """
    Quaternion geodesic distance in degrees.
    Input format: [qw, qx, qy, qz].
    """
    q_pred = normalize_quaternion_np(q_pred)
    q_target = normalize_quaternion_np(q_target)

    inner = np.abs(np.sum(q_pred * q_target, axis=-1))
    inner = np.clip(inner, 0.0, 1.0)

    return np.rad2deg(2.0 * np.arccos(inner)).astype(np.float32)


def quaternion_geodesic_distance_torch(q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    """
    Quaternion geodesic distance in radians.
    Input format: [qw, qx, qy, qz].
    """
    q_pred = normalize_quaternion_torch(q_pred)
    q_target = normalize_quaternion_torch(q_target)

    inner = torch.abs(torch.sum(q_pred * q_target, dim=-1))
    inner = torch.clamp(inner, 0.0, 1.0)

    return 2.0 * torch.acos(inner)


def quaternion_cosine_loss(
    q_pred: torch.Tensor,
    q_target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Stable quaternion loss for training.

    loss = 1 - |<q_pred, q_target>|
    """
    q_pred = normalize_quaternion_torch(q_pred)
    q_target = normalize_quaternion_torch(q_target)

    inner = torch.abs(torch.sum(q_pred * q_target, dim=-1))
    loss = 1.0 - torch.clamp(inner, 0.0, 1.0)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    raise ValueError(f"Unsupported reduction: {reduction}")


def translation_l1_loss(
    pred_translation: torch.Tensor,
    target_translation: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    loss = torch.abs(pred_translation - target_translation)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    raise ValueError(f"Unsupported reduction: {reduction}")


def translation_error_m_np(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred_xyz - gt_xyz, axis=-1).astype(np.float32)


def denormalize_translation_np(
    x_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized translation back to meter-space.

    Supports:
        x_norm: [3] or [N, 3]
        mean:   [3] or [N, 3]
        std:    [3] or [N, 3]
    """
    x_norm = np.asarray(x_norm, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return (x_norm * std + mean).astype(np.float32)


def denormalize_translation_torch(
    x_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Convert normalized translation back to meter-space.

    Supports:
        x_norm: [B, 3]
        mean:   [B, 3]
        std:    [B, 3]
    """
    return x_norm * std + mean


def summarize_array(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def save_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class PoseWindowDataset(Dataset):
    """
    Unified motion-only next-step 6DoF pose prediction dataset.

    For each valid decision time t:

        input  = [p_{t-T+1}, ..., p_t]
        target = p_{t+1}

    pose_x uses:
        [normalized HeadX, normalized HeadY, normalized HeadZ, qw, qx, qy, qz]

    Translation is normalized per file by default.
    Rotation is converted from Euler angles to unit quaternion and is not mean/std normalized.
    """

    def __init__(
        self,
        input_glob: str,
        seq_len: int = 30,
        normalize_translation: bool = True,
        use_timer_sort: bool = True,
        timer_col: str = "Timer",
        label_col: str = "Label",
        explorer_score_col: str = "explorer_score",
        aux_cols: Optional[List[str]] = None,
        normalize_aux: bool = True,
        filter_label: Optional[str] = None,
        clean_transition_only: bool = False,
    ) -> None:
        self.input_glob = input_glob
        self.seq_len = int(seq_len)
        self.normalize_translation = bool(normalize_translation)
        self.use_timer_sort = bool(use_timer_sort)
        self.timer_col = timer_col
        self.label_col = label_col
        self.explorer_score_col = explorer_score_col
        self.aux_cols = aux_cols
        self.normalize_aux = bool(normalize_aux)
        self.filter_label = filter_label
        self.clean_transition_only = bool(clean_transition_only)

        self.samples: List[PoseWindowSample] = []

        file_paths = sorted(glob.glob(input_glob))
        if not file_paths:
            raise FileNotFoundError(f"No files matched pattern: {input_glob}")

        for file_path in file_paths:
            self.samples.extend(self._build_from_file(file_path))

    def _validate_columns(self, df: pd.DataFrame, file_path: str) -> None:
        required = [self.timer_col, *POSE_COLS]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{file_path} missing required columns: {missing}")

        if self.aux_cols is not None:
            missing_aux = [c for c in self.aux_cols if c not in df.columns]
            if missing_aux:
                raise ValueError(f"{file_path} missing aux columns: {missing_aux}")

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.use_timer_sort:
            df = df.sort_values(self.timer_col).reset_index(drop=True)

        drop_cols = [self.timer_col, *POSE_COLS]
        if self.aux_cols is not None:
            drop_cols += self.aux_cols

        df = df.dropna(subset=drop_cols).reset_index(drop=True)

        if self.label_col in df.columns:
            df[self.label_col] = df[self.label_col].astype(str)

        return df

    def _build_from_file(self, file_path: str) -> List[PoseWindowSample]:
        df = pd.read_csv(file_path)
        self._validate_columns(df, file_path)
        df = self._prepare_dataframe(df)

        file_name = os.path.basename(file_path)
        samples: List[PoseWindowSample] = []

        if len(df) <= self.seq_len:
            return samples

        trans_raw = df[TR_COLS].to_numpy(dtype=np.float32)
        euler_raw = df[ROT_COLS].to_numpy(dtype=np.float32)
        quat_raw = euler_array_deg_to_quaternion_wxyz(euler_raw)

        if self.normalize_translation:
            trans_mean = trans_raw.mean(axis=0).astype(np.float32)
            trans_std = (trans_raw.std(axis=0) + 1e-6).astype(np.float32)
            trans_out = ((trans_raw - trans_mean) / trans_std).astype(np.float32)
        else:
            trans_mean = np.zeros((3,), dtype=np.float32)
            trans_std = np.ones((3,), dtype=np.float32)
            trans_out = trans_raw.astype(np.float32)

        if self.aux_cols is not None:
            aux_raw = df[self.aux_cols].to_numpy(dtype=np.float32)

            if self.normalize_aux:
                aux_mean = aux_raw.mean(axis=0).astype(np.float32)
                aux_std = (aux_raw.std(axis=0) + 1e-6).astype(np.float32)
                aux_out = ((aux_raw - aux_mean) / aux_std).astype(np.float32)
            else:
                aux_mean = np.zeros((len(self.aux_cols),), dtype=np.float32)
                aux_std = np.ones((len(self.aux_cols),), dtype=np.float32)
                aux_out = aux_raw.astype(np.float32)
        else:
            aux_mean = None
            aux_std = None
            aux_out = None

        pose7 = np.concatenate([trans_out, quat_raw], axis=1).astype(np.float32)

        n = len(df)

        for end_idx in range(self.seq_len - 1, n - 1):
            start_idx = end_idx - self.seq_len + 1
            target_idx = end_idx + 1
            prev_idx = end_idx - 1

            if prev_idx < 0:
                continue

            row_t = df.iloc[end_idx]
            row_t1 = df.iloc[target_idx]

            label_t = str(row_t[self.label_col]) if self.label_col in df.columns else "Unlabeled"
            label_t1 = str(row_t1[self.label_col]) if self.label_col in df.columns else "Unlabeled"

            if self.filter_label is not None:
                if label_t != self.filter_label:
                    continue
                if self.clean_transition_only and label_t1 != self.filter_label:
                    continue

            pose_x = pose7[start_idx:end_idx + 1]
            if pose_x.shape != (self.seq_len, 7):
                continue

            explorer_score_t = (
                float(row_t[self.explorer_score_col])
                if self.explorer_score_col in df.columns
                else float("nan")
            )

            aux_x = aux_out[end_idx].astype(np.float32) if aux_out is not None else None

            sample = PoseWindowSample(
                pose_x=pose_x.astype(np.float32),
                target_translation=trans_out[target_idx].astype(np.float32),
                target_quaternion=quat_raw[target_idx].astype(np.float32),
                current_translation=trans_out[end_idx].astype(np.float32),
                current_quaternion=quat_raw[end_idx].astype(np.float32),
                prev_quaternion=quat_raw[prev_idx].astype(np.float32),

                aux_x=aux_x,

                file_name=file_name,
                window_start_index=int(start_idx),
                end_row_index=int(end_idx),
                target_row_index=int(target_idx),
                time_t=float(row_t[self.timer_col]),
                time_t1=float(row_t1[self.timer_col]),
                label_t=label_t,
                label_t1=label_t1,
                explorer_score_t=explorer_score_t,

                trans_mean=trans_mean.copy(),
                trans_std=trans_std.copy(),
                aux_mean=None if aux_mean is None else aux_mean.copy(),
                aux_std=None if aux_std is None else aux_std.copy(),

                head_rate_yaw_q90_t = float(row_t["head_rate_yaw_q90"]) if "head_rate_yaw_q90" in df.columns else float("nan"),
                yaw_zero_cross_per_s_t = float(row_t["yaw_zero_cross_per_s"]) if "yaw_zero_cross_per_s" in df.columns else float("nan"),
                head_rate_yaw_hi_prop_t = float(row_t["head_rate_yaw_hi_prop"]) if "head_rate_yaw_hi_prop" in df.columns else float("nan"),
                head_speed_mean_t = float(row_t["speed_mean"]) if "speed_mean" in df.columns else float("nan"),
                score_valid_t = float(row_t["score_valid"]) if "score_valid" in df.columns else float("nan"),
                scene_name = str(row_t["scene_name"]) if "scene_name" in df.columns else "",
                split = str(row_t["split"]) if "split" in df.columns else "",
            )

            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]

        item = {
            "pose_x": torch.tensor(s.pose_x, dtype=torch.float32),
            "target_translation": torch.tensor(s.target_translation, dtype=torch.float32),
            "target_quaternion": torch.tensor(s.target_quaternion, dtype=torch.float32),
            "current_translation": torch.tensor(s.current_translation, dtype=torch.float32),
            "current_quaternion": torch.tensor(s.current_quaternion, dtype=torch.float32),
            "prev_quaternion": torch.tensor(s.prev_quaternion, dtype=torch.float32),
            "meta": {
                "file_name": s.file_name,
                "window_start_index": s.window_start_index,
                "end_row_index": s.end_row_index,
                "target_row_index": s.target_row_index,
                "time_t": s.time_t,
                "time_t1": s.time_t1,
                "label_t": s.label_t,
                "label_t1": s.label_t1,
                "explorer_score_t": s.explorer_score_t,
                "trans_mean": s.trans_mean,
                "trans_std": s.trans_std,
                "aux_mean": s.aux_mean,
                "aux_std": s.aux_std,
                "head_rate_yaw_q90_t": s.head_rate_yaw_q90_t,
                "yaw_zero_cross_per_s_t": s.yaw_zero_cross_per_s_t,
                "head_rate_yaw_hi_prop_t": s.head_rate_yaw_hi_prop_t,
                "head_speed_mean_t": s.head_speed_mean_t,
                "score_valid_t": s.score_valid_t,
                "scene_name": s.scene_name,
                "split": s.split,
            },
        }

        if s.aux_x is not None:
            item["aux_x"] = torch.tensor(s.aux_x, dtype=torch.float32)
        else:
            item["aux_x"] = None

        return item


def pose_collate_fn(batch: List[Dict]) -> Dict:
    pose_x = torch.stack([b["pose_x"] for b in batch], dim=0)
    target_translation = torch.stack([b["target_translation"] for b in batch], dim=0)
    target_quaternion = torch.stack([b["target_quaternion"] for b in batch], dim=0)
    current_translation = torch.stack([b["current_translation"] for b in batch], dim=0)
    current_quaternion = torch.stack([b["current_quaternion"] for b in batch], dim=0)
    prev_quaternion = torch.stack([b["prev_quaternion"] for b in batch], dim=0)

    has_aux = batch[0]["aux_x"] is not None
    aux_x = torch.stack([b["aux_x"] for b in batch], dim=0) if has_aux else None

    metas = [b["meta"] for b in batch]

    return {
        "pose_x": pose_x,
        "aux_x": aux_x,
        "target_translation": target_translation,
        "target_quaternion": target_quaternion,
        "current_translation": current_translation,
        "current_quaternion": current_quaternion,
        "prev_quaternion": prev_quaternion,
        "meta": metas,
    }


def print_dataset_preview(dataset: PoseWindowDataset, num_preview: int = 3) -> None:
    print("\n=== Dataset Summary ===")
    print(f"num_samples: {len(dataset)}")

    if len(dataset) == 0:
        return

    label_counts: Dict[str, int] = {}
    for s in dataset.samples:
        label_counts[s.label_t] = label_counts.get(s.label_t, 0) + 1

    print("label_t counts:")
    for k, v in sorted(label_counts.items()):
        print(f"  {k}: {v}")

    print("\n=== Preview Samples ===")
    n = min(num_preview, len(dataset))

    for i in range(n):
        item = dataset[i]

        pose_x = item["pose_x"]
        target_q = item["target_quaternion"]
        current_q = item["current_quaternion"]
        prev_q = item["prev_quaternion"]
        meta = item["meta"]

        print(f"\n--- Sample {i} ---")
        print(f"file_name          : {meta['file_name']}")
        print(f"window_start_index : {meta['window_start_index']}")
        print(f"end_row_index      : {meta['end_row_index']}")
        print(f"target_row_index   : {meta['target_row_index']}")
        print(f"time_t             : {meta['time_t']:.6f}")
        print(f"time_t1            : {meta['time_t1']:.6f}")
        print(f"label_t            : {meta['label_t']}")
        print(f"label_t1           : {meta['label_t1']}")
        print(f"explorer_score_t   : {meta['explorer_score_t']}")
        print(f"pose_x.shape       : {tuple(pose_x.shape)}")
        print(f"target_translation : {item['target_translation'].numpy()}")
        print(f"target_quaternion  : {target_q.numpy()}")
        print(f"aux_x              : {None if item['aux_x'] is None else item['aux_x'].numpy()}")

        print("quaternion norms:")
        print(f"  prev    : {torch.linalg.norm(prev_q).item():.6f}")
        print(f"  current : {torch.linalg.norm(current_q).item():.6f}")
        print(f"  target  : {torch.linalg.norm(target_q).item():.6f}")

        print("translation mean/std:")
        print(f"  mean: {meta['trans_mean']}")
        print(f"  std : {meta['trans_std']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_preview", type=int, default=3)

    parser.add_argument("--no_normalize_translation", action="store_true")
    parser.add_argument("--use_aux", type=bool, default=False)
    parser.add_argument("--no_normalize_aux", action="store_true")

    parser.add_argument(
        "--filter_label",
        type=str,
        default=None,
        choices=[None, "Explorer", "Focused", "Unlabeled"],
        help="Optional label filter at decision time t.",
    )
    parser.add_argument(
        "--clean_transition_only",
        action="store_true",
        help="If filter_label is set, also require target label t+1 to match.",
    )

    args = parser.parse_args()

    aux_cols = AUX_COLS_DEFAULT if args.use_aux else None

    dataset = PoseWindowDataset(
        input_glob=args.input_glob,
        seq_len=args.seq_len,
        normalize_translation=not args.no_normalize_translation,
        aux_cols=aux_cols,
        normalize_aux=not args.no_normalize_aux,
        filter_label=args.filter_label,
        clean_transition_only=args.clean_transition_only,
    )

    print_dataset_preview(dataset, num_preview=args.num_preview)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pose_collate_fn,
    )

    batch = next(iter(loader))

    print("\n=== Batch Check ===")
    print(f"pose_x.shape              : {tuple(batch['pose_x'].shape)}")
    print(f"target_translation.shape  : {tuple(batch['target_translation'].shape)}")
    print(f"target_quaternion.shape   : {tuple(batch['target_quaternion'].shape)}")
    print(f"current_translation.shape : {tuple(batch['current_translation'].shape)}")
    print(f"current_quaternion.shape  : {tuple(batch['current_quaternion'].shape)}")
    print(f"prev_quaternion.shape     : {tuple(batch['prev_quaternion'].shape)}")
    print(f"aux_x                     : {None if batch['aux_x'] is None else tuple(batch['aux_x'].shape)}")

    q_norm = torch.linalg.norm(batch["target_quaternion"], dim=-1)
    print("\n=== Quaternion Norm Check ===")
    print(f"target quaternion norm mean: {q_norm.mean().item():.6f}")
    print(f"target quaternion norm min : {q_norm.min().item():.6f}")
    print(f"target quaternion norm max : {q_norm.max().item():.6f}")


if __name__ == "__main__":
    main()