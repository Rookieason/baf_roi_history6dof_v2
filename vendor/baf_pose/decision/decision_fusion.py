from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BAF_POSE_DIR = os.path.dirname(SCRIPT_DIR)
if BAF_POSE_DIR not in sys.path:
    sys.path.insert(0, BAF_POSE_DIR)

from utility import build_decision_loader
from confidence.model import build_confidence_estimator
from prediction.focused_branch.model import FocusedResidualLSTM, FocusedBranchConfig
from prediction.explorer_branch.model import build_explorer_branch


POSE_COLS = ["HeadX", "HeadY", "HeadZ", "HeadRX", "HeadRY", "HeadRZ"]
TR_COLS = ["HeadX", "HeadY", "HeadZ"]
ROT_COLS = ["HeadRX", "HeadRY", "HeadRZ"]


def circular_abs_diff_deg(pred_angle: np.ndarray, target_angle: np.ndarray) -> np.ndarray:
    diff = pred_angle - target_angle
    diff = (diff + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def translation_error_m(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred_xyz - gt_xyz, axis=1)


def rotation_mae_deg(pred_rot: np.ndarray, gt_rot: np.ndarray) -> np.ndarray:
    err = circular_abs_diff_deg(pred_rot, gt_rot)
    return err.mean(axis=1)


def quaternion_geo_deg_np(q_pred: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """
    Quaternion geodesic distance in degrees.
    Quaternion format: [qw, qx, qy, qz].
    q_pred and q_target are [N, 4].
    """
    q_pred = q_pred.astype(np.float32)
    q_target = q_target.astype(np.float32)

    q_pred = q_pred / (np.linalg.norm(q_pred, axis=1, keepdims=True) + 1e-8)
    q_target = q_target / (np.linalg.norm(q_target, axis=1, keepdims=True) + 1e-8)

    inner = np.abs(np.sum(q_pred * q_target, axis=1))
    inner = np.clip(inner, 0.0, 1.0)

    return np.rad2deg(2.0 * np.arccos(inner))


def euler_deg_to_quaternion_wxyz(euler_deg: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles [N, 3] in degrees to quaternion [N, 4].
    Output format: [qw, qx, qy, qz].
    """
    r = R.from_euler("xyz", euler_deg, degrees=True)
    q_xyzw = r.as_quat().astype(np.float32)
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1)
    q_wxyz = q_wxyz / (np.linalg.norm(q_wxyz, axis=1, keepdims=True) + 1e-8)
    return q_wxyz.astype(np.float32)


def quaternion_wxyz_to_euler_deg(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [N, 4] in [qw, qx, qy, qz] to Euler angles [N, 3] in degrees.
    This Euler output is mainly for CSV readability and circular-MAE diagnostics.
    """
    q_wxyz = q_wxyz.astype(np.float32)
    q_wxyz = q_wxyz / (np.linalg.norm(q_wxyz, axis=1, keepdims=True) + 1e-8)

    q_xyzw = np.stack(
        [
            q_wxyz[:, 1],
            q_wxyz[:, 2],
            q_wxyz[:, 3],
            q_wxyz[:, 0],
        ],
        axis=1,
    ).astype(np.float32)

    return R.from_quat(q_xyzw).as_euler("xyz", degrees=True).astype(np.float32)


def slerp_quaternion_wxyz(q0: np.ndarray, q1: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Batch quaternion SLERP.

    q0: [N, 4], Focused quaternion
    q1: [N, 4], Explorer quaternion
    weight: [N], interpolation weight toward q1

    Output:
        q: [N, 4], normalized fused quaternion
    """
    q0 = q0.astype(np.float32)
    q1 = q1.astype(np.float32)
    weight = weight.astype(np.float32).reshape(-1, 1)

    q0 = q0 / (np.linalg.norm(q0, axis=1, keepdims=True) + 1e-8)
    q1 = q1 / (np.linalg.norm(q1, axis=1, keepdims=True) + 1e-8)

    dot = np.sum(q0 * q1, axis=1, keepdims=True)

    # Use the shortest path on the quaternion sphere.
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    dot = np.clip(dot, 0.0, 1.0)

    # If the two quaternions are very close, normalized lerp is more stable.
    close = dot > 0.9995

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * weight
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / (sin_theta_0 + 1e-8)
    s1 = sin_theta / (sin_theta_0 + 1e-8)

    q_slerp = s0 * q0 + s1 * q1
    q_lerp = (1.0 - weight) * q0 + weight * q1

    q = np.where(close, q_lerp, q_slerp)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)

    return q.astype(np.float32)


def calibrate_alpha(alpha_t: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Conservative alpha calibration for fusion.

    alpha <= low  -> 0
    alpha >= high -> 1
    otherwise     -> linear interpolation
    """
    return np.clip((alpha_t - low) / (high - low), 0.0, 1.0).astype(np.float32)


def compute_alpha_fusion(
        alpha_t: np.ndarray,
        mode: str,
        alpha_low: float,
        alpha_high: float,
        fixed_alpha: float,
        meta=None,
    ) -> np.ndarray:
    """
    Compute the fusion coefficient used by the decision module.

    mode:
        direct:
            lambda_t = alpha_t

        calibrated:
            lambda_t = clip((alpha_t - alpha_low) / (alpha_high - alpha_low), 0, 1)

        fixed:
            lambda_t = fixed_alpha

        oracle_label:
            lambda_t = 0 for Focused
            lambda_t = 1 for Explorer
            lambda_t = fixed_alpha for Unlabeled / others

    Note:
        oracle_label is for diagnostic ablation only.
        It must not be described as an online method.
    """
    alpha_t = np.asarray(alpha_t, dtype=np.float32).reshape(-1)

    if mode == "direct":
        return np.clip(alpha_t, 0.0, 1.0).astype(np.float32)

    if mode == "calibrated":
        return calibrate_alpha(
            alpha_t,
            low=alpha_low,
            high=alpha_high,
        )

    if mode == "fixed":
        return np.full_like(alpha_t, fill_value=fixed_alpha, dtype=np.float32)

    if mode == "oracle_label":
        if meta is None:
            raise ValueError("oracle_label alpha mode requires meta.")

        out = np.full_like(alpha_t, fill_value=fixed_alpha, dtype=np.float32)

        for i, m in enumerate(meta):
            label = str(m.get("label_t", ""))

            if label == "Focused":
                out[i] = 0.0
            elif label == "Explorer":
                out[i] = 1.0
            else:
                out[i] = fixed_alpha

        return out.astype(np.float32)

    raise ValueError(f"Unknown alpha mode: {mode}")


def compute_temporal_optimized_alpha(
    alpha_t: np.ndarray,
    meta,
    temporal_state: Dict[str, float],
    mu: float,
    init_mode: str = "alpha",
    init_value: float = 0.5,
) -> np.ndarray:
    """
    Thesis-aligned temporal alpha optimization.

    Closed-form solution:
        w_t = mu * alpha_t + (1 - mu) * w_{t-1}

    The previous weight is maintained independently for each file.

    Args:
        alpha_t:
            Explorer confidence from confidence estimator, shape [B].
        meta:
            Batch metadata. Each sample must contain file_name.
        temporal_state:
            Dictionary storing previous optimized weight for each file.
        mu:
            Update coefficient in [0, 1].
        init_mode:
            alpha: initialize w_{t-1} by the first alpha of each file.
            fixed: initialize w_{t-1} by init_value.
        init_value:
            Initial value used when init_mode == fixed.

    Returns:
        optimized weight array, shape [B].
    """
    alpha_t = np.asarray(alpha_t, dtype=np.float32).reshape(-1)
    out = np.zeros_like(alpha_t, dtype=np.float32)

    mu = float(np.clip(mu, 0.0, 1.0))

    for i, m in enumerate(meta):
        file_name = str(m["file_name"])

        if file_name not in temporal_state:
            if init_mode == "alpha":
                w_prev = float(alpha_t[i])
            elif init_mode == "fixed":
                w_prev = float(init_value)
            else:
                raise ValueError(f"Unknown temporal init mode: {init_mode}")
        else:
            w_prev = float(temporal_state[file_name])

        w_t = mu * float(alpha_t[i]) + (1.0 - mu) * w_prev
        w_t = float(np.clip(w_t, 0.0, 1.0))

        temporal_state[file_name] = w_t
        out[i] = w_t

    return out


def apply_decision_fusion(
        fusion_mode: str,
        alpha_fusion: np.ndarray,
        alpha_t: np.ndarray,
        focused_tr: np.ndarray,
        explorer_tr: np.ndarray,
        focused_q: np.ndarray,
        explorer_q: np.ndarray,
        hard_threshold: float,
        fixed_alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply decision-level ablation variants.

    Returns:
        final_tr:
            [B, 3] final translation prediction.

        final_q:
            [B, 4] final quaternion prediction.

        effective_alpha:
            [B] actual fusion coefficient used by this fusion mode.
    """
    batch_size = focused_tr.shape[0]

    if fusion_mode == "focused_only":
        effective_alpha = np.zeros((batch_size,), dtype=np.float32)
        final_tr = focused_tr
        final_q = focused_q
        return final_tr, final_q, effective_alpha

    if fusion_mode == "explorer_only":
        effective_alpha = np.ones((batch_size,), dtype=np.float32)
        final_tr = explorer_tr
        final_q = explorer_q
        return final_tr, final_q, effective_alpha

    if fusion_mode == "fixed_average":
        effective_alpha = np.full((batch_size,), fill_value=fixed_alpha, dtype=np.float32)

        final_tr = (
            (1.0 - effective_alpha[:, None]) * focused_tr
            + effective_alpha[:, None] * explorer_tr
        )

        final_q = slerp_quaternion_wxyz(
            q0=focused_q,
            q1=explorer_q,
            weight=effective_alpha,
        )

        return final_tr, final_q, effective_alpha

    if fusion_mode == "hard_routing":
        effective_alpha = (alpha_t >= hard_threshold).astype(np.float32)

        final_tr = (
            (1.0 - effective_alpha[:, None]) * focused_tr
            + effective_alpha[:, None] * explorer_tr
        )

        final_q = slerp_quaternion_wxyz(
            q0=focused_q,
            q1=explorer_q,
            weight=effective_alpha,
        )

        return final_tr, final_q, effective_alpha

    if fusion_mode == "translation_focused_rotation":
        effective_alpha = alpha_fusion.astype(np.float32)

        final_tr = (
            (1.0 - effective_alpha[:, None]) * focused_tr
            + effective_alpha[:, None] * explorer_tr
        )

        final_q = focused_q

        return final_tr, final_q, effective_alpha

    if fusion_mode == "full_slerp":
        effective_alpha = alpha_fusion.astype(np.float32)

        final_tr = (
            (1.0 - effective_alpha[:, None]) * focused_tr
            + effective_alpha[:, None] * explorer_tr
        )

        final_q = slerp_quaternion_wxyz(
            q0=focused_q,
            q1=explorer_q,
            weight=effective_alpha,
        )

        return final_tr, final_q, effective_alpha

    raise ValueError(f"Unknown fusion mode: {fusion_mode}")


def load_confidence_model(ckpt_path: str, device: torch.device, fallback_seq_len: int):
    ckpt = torch.load(ckpt_path, map_location=device)
    seq_len = int(ckpt.get("seq_len", fallback_seq_len))

    model = build_confidence_estimator(
        input_dim=6,
        seq_len=seq_len,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


def load_focused_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})

    # config = FocusedBranchConfig(
    #     input_dim=6,
    #     hidden_dim=int(args.get("hidden_dim", 128)),
    #     num_layers=int(args.get("num_layers", 2)),
    #     dropout=float(args.get("dropout", 0.1)),
    #     use_layer_norm=True,
    # )
    config = FocusedBranchConfig(
        input_dim=int(args.get("input_dim", 7)),
        hidden_dim=int(args.get("hidden_dim", 128)),
        num_layers=int(args.get("num_layers", 2)),
        dropout=float(args.get("dropout", 0.1)),
        use_layer_norm=True,
    )

    model = FocusedResidualLSTM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


def load_explorer_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
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


def stack_meta_array(meta, key: str) -> np.ndarray:
    return np.stack([m[key] for m in meta], axis=0).astype(np.float32)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_glob", type=str, required=True)

    parser.add_argument("--confidence_checkpoint", type=str, required=True)
    parser.add_argument("--focused_checkpoint", type=str, required=True)
    parser.add_argument("--explorer_checkpoint", type=str, required=True)

    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Path for saving the by-behavior summary CSV.",
    )
    parser.add_argument("--method_name", type=str, default="decision_fusion")

    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)

    # parser.add_argument(
    #     "--alpha_mode",
    #     type=str,
    #     default="calibrated",
    #     choices=["direct", "calibrated"],
    # )
    # parser.add_argument("--alpha_low", type=float, default=0.3)
    # parser.add_argument("--alpha_high", type=float, default=0.9)
    parser.add_argument(
        "--alpha_mode",
        type=str,
        default="calibrated",
        choices=["direct", "calibrated", "fixed", "oracle_label", "temporal_optimized"],
        help="How to convert confidence alpha_t into a fusion coefficient.",
    )

    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="full_slerp",
        choices=[
            "focused_only",
            "explorer_only",
            "fixed_average",
            "hard_routing",
            "translation_focused_rotation",
            "full_slerp",
        ],
        help="Decision fusion ablation mode.",
    )

    parser.add_argument("--alpha_low", type=float, default=0.3)
    parser.add_argument("--alpha_high", type=float, default=0.9)
    parser.add_argument("--fixed_alpha", type=float, default=0.5)
    parser.add_argument("--hard_threshold", type=float, default=0.5)

    parser.add_argument(
        "--temporal_mu",
        type=float,
        default=0.6,
        help="Update coefficient for temporal optimized alpha: w_t = mu * alpha_t + (1 - mu) * w_{t-1}.",
    )

    parser.add_argument(
        "--temporal_init_mode",
        type=str,
        default="alpha",
        choices=["alpha", "fixed"],
        help="Initialization mode for temporal optimized alpha.",
    )

    parser.add_argument(
        "--temporal_init_value",
        type=float,
        default=0.5,
        help="Initial w_prev when temporal_init_mode is fixed.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    confidence_model = load_confidence_model(
        ckpt_path=args.confidence_checkpoint,
        device=device,
        fallback_seq_len=args.seq_len,
    )

    focused_model = load_focused_model(
        ckpt_path=args.focused_checkpoint,
        device=device,
    )

    explorer_model, explorer_use_aux = load_explorer_model(
        ckpt_path=args.explorer_checkpoint,
        device=device,
    )

    print(f"[INFO] Explorer checkpoint use_aux: {explorer_use_aux}")

    loader = build_decision_loader(
        input_glob=args.input_glob,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        use_explorer_aux=explorer_use_aux,
        enforce_explorer_quat_continuity=False,
    )

    rows = []
    temporal_alpha_state: Dict[str, float] = {}

    for batch_idx, batch in enumerate(loader):
        x_conf = batch["x_conf"].to(device)
        x_focused = batch["x_focused"].to(device)
        x_explorer = batch["x_explorer"].to(device)
        aux_explorer = batch["aux_explorer"]
        gt_pose = batch["gt_pose"].cpu().numpy()
        meta = batch["meta"]

        if aux_explorer is not None:
            aux_explorer = aux_explorer.to(device)

        confidence_out = confidence_model(x_conf)
        focused_out = focused_model(x_focused)
        explorer_out = explorer_model(x_explorer, aux_explorer)

        alpha_t = confidence_out["confidence"].detach().cpu().numpy().reshape(-1)

        # if args.alpha_mode == "direct":
        #     alpha_fusion = np.clip(alpha_t, 0.0, 1.0).astype(np.float32)
        # else:
        #     alpha_fusion = calibrate_alpha(
        #         alpha_t,
        #         low=args.alpha_low,
        #         high=args.alpha_high,
        #     )
        # alpha_fusion = compute_alpha_fusion(
        #     alpha_t=alpha_t,
        #     mode=args.alpha_mode,
        #     alpha_low=args.alpha_low,
        #     alpha_high=args.alpha_high,
        #     fixed_alpha=args.fixed_alpha,
        #     meta=meta,
        # )
        if args.alpha_mode == "temporal_optimized":
            alpha_fusion = compute_temporal_optimized_alpha(
                alpha_t=alpha_t,
                meta=meta,
                temporal_state=temporal_alpha_state,
                mu=args.temporal_mu,
                init_mode=args.temporal_init_mode,
                init_value=args.temporal_init_value,
            )
        else:
            alpha_fusion = compute_alpha_fusion(
                alpha_t=alpha_t,
                mode=args.alpha_mode,
                alpha_low=args.alpha_low,
                alpha_high=args.alpha_high,
                fixed_alpha=args.fixed_alpha,
                meta=meta,
            )

        # focused_pred_norm = focused_out["y_hat"].detach().cpu().numpy()
        explorer_pred_tr_norm = explorer_out["p_hat_tr"].detach().cpu().numpy()
        explorer_pred_q = explorer_out["q_hat_norm"].detach().cpu().numpy()

        # focused_pose_mean = stack_meta_array(meta, "focused_pose_mean")
        # focused_pose_std = stack_meta_array(meta, "focused_pose_std")

        explorer_trans_mean = stack_meta_array(meta, "explorer_trans_mean")
        explorer_trans_std = stack_meta_array(meta, "explorer_trans_std")

        # focused_pred_raw = focused_pred_norm * focused_pose_std + focused_pose_mean
        # focused_tr = focused_pred_raw[:, :3]
        # focused_euler = focused_pred_raw[:, 3:]

        # focused_q = euler_deg_to_quaternion_wxyz(focused_euler)
        focused_pred_tr_norm = focused_out["p_hat_tr"].detach().cpu().numpy()
        focused_pred_q = focused_out["q_hat_norm"].detach().cpu().numpy()

        focused_trans_mean = stack_meta_array(meta, "focused_trans_mean")
        focused_trans_std = stack_meta_array(meta, "focused_trans_std")

        focused_tr = focused_pred_tr_norm * focused_trans_std + focused_trans_mean
        focused_q = focused_pred_q / (
            np.linalg.norm(focused_pred_q, axis=1, keepdims=True) + 1e-8
        )

        focused_euler = quaternion_wxyz_to_euler_deg(focused_q)

        explorer_tr = explorer_pred_tr_norm * explorer_trans_std + explorer_trans_mean
        explorer_q = explorer_pred_q / (
            np.linalg.norm(explorer_pred_q, axis=1, keepdims=True) + 1e-8
        )

        # Decision fusion.
        # final_tr = (
        #     (1.0 - alpha_fusion[:, None]) * focused_tr
        #     + alpha_fusion[:, None] * explorer_tr
        # )

        # final_q = slerp_quaternion_wxyz(
        #     q0=focused_q,
        #     q1=explorer_q,
        #     weight=alpha_fusion,
        # )
        # Decision fusion / ablation.
        final_tr, final_q, effective_alpha = apply_decision_fusion(
            fusion_mode=args.fusion_mode,
            alpha_fusion=alpha_fusion,
            alpha_t=alpha_t,
            focused_tr=focused_tr,
            explorer_tr=explorer_tr,
            focused_q=focused_q,
            explorer_q=explorer_q,
            hard_threshold=args.hard_threshold,
            fixed_alpha=args.fixed_alpha,
        )

        final_euler = quaternion_wxyz_to_euler_deg(final_q)

        final_pose = np.concatenate([final_tr, final_euler], axis=1).astype(np.float32)

        gt_q = stack_meta_array(meta, "target_quaternion")

        final_translation_err = translation_error_m(final_tr, gt_pose[:, :3])
        final_rotation_geo = quaternion_geo_deg_np(final_q, gt_q)
        final_rotation_euler_mae = rotation_mae_deg(final_euler, gt_pose[:, 3:])

        focused_translation_err = translation_error_m(focused_tr, gt_pose[:, :3])
        focused_rotation_geo = quaternion_geo_deg_np(focused_q, gt_q)

        explorer_translation_err = translation_error_m(explorer_tr, gt_pose[:, :3])
        explorer_rotation_geo = quaternion_geo_deg_np(explorer_q, gt_q)

        batch_size = len(gt_pose)

        for i in range(batch_size):
            row = {
                "method_name": args.method_name,
                "file_name": meta[i]["file_name"],
                "window_start_index": meta[i]["window_start_index"],
                "end_row_index": meta[i]["end_row_index"],
                "target_row_index": meta[i]["target_row_index"],
                "time_t": meta[i]["time_t"],
                "time_t1": meta[i]["time_t1"],
                "Label_t": meta[i]["label_t"],
                "Label_t1": meta[i]["label_t1"],
                "explorer_score_t": meta[i]["explorer_score_t"],

                "head_rate_yaw_q90_t": (
                    float(meta[i]["head_rate_yaw_q90_t"])
                    if meta[i].get("head_rate_yaw_q90_t") is not None else np.nan
                ),
                "yaw_zero_cross_per_s_t": (
                    float(meta[i]["yaw_zero_cross_per_s_t"])
                    if meta[i].get("yaw_zero_cross_per_s_t") is not None else np.nan
                ),
                "head_rate_yaw_hi_prop_t": (
                    float(meta[i]["head_rate_yaw_hi_prop_t"])
                    if meta[i].get("head_rate_yaw_hi_prop_t") is not None else np.nan
                ),
                "head_speed_mean_t": (
                    float(meta[i]["head_speed_mean_t"])
                    if meta[i].get("head_speed_mean_t") is not None else np.nan
                ),
                "score_valid_t": (
                    float(meta[i]["score_valid_t"])
                    if meta[i].get("score_valid_t") is not None else np.nan
                ),
                "scene_name": meta[i].get("scene_name", ""),
                "split": meta[i].get("split", ""),

                "alpha_t": float(alpha_t[i]),
                "alpha_fusion": float(alpha_fusion[i]),
                "effective_alpha": float(effective_alpha[i]),
                "alpha_mode": args.alpha_mode,
                "fusion_mode": args.fusion_mode,
                "temporal_mu": args.temporal_mu,
            }

            for j, col in enumerate(TR_COLS):
                row[f"focused_pred_{col}"] = float(focused_tr[i, j])
                row[f"explorer_pred_{col}"] = float(explorer_tr[i, j])
                row[f"final_pred_{col}"] = float(final_tr[i, j])
                row[f"gt_{col}"] = float(gt_pose[i, j])

            for j, col in enumerate(ROT_COLS):
                row[f"focused_pred_{col}"] = float(focused_euler[i, j])
                row[f"explorer_pred_{col}"] = float(quaternion_wxyz_to_euler_deg(explorer_q[i:i+1])[0, j])
                row[f"final_pred_{col}"] = float(final_euler[i, j])
                row[f"gt_{col}"] = float(gt_pose[i, j + 3])

            for j, name in enumerate(["qw", "qx", "qy", "qz"]):
                row[f"focused_pred_{name}"] = float(focused_q[i, j])
                row[f"explorer_pred_{name}"] = float(explorer_q[i, j])
                row[f"final_pred_{name}"] = float(final_q[i, j])
                row[f"gt_{name}"] = float(gt_q[i, j])

            row["focused_translation_error_m"] = float(focused_translation_err[i])
            row["focused_rotation_geo_deg"] = float(focused_rotation_geo[i])

            row["explorer_translation_error_m"] = float(explorer_translation_err[i])
            row["explorer_rotation_geo_deg"] = float(explorer_rotation_geo[i])

            row["final_translation_error_m"] = float(final_translation_err[i])
            row["final_rotation_geo_deg"] = float(final_rotation_geo[i])
            row["final_rotation_euler_mae_deg"] = float(final_rotation_euler_mae[i])

            rows.append(row)

        if (batch_idx + 1) % 20 == 0:
            print(f"[INFO] Processed batches: {batch_idx + 1}/{len(loader)}")

    out_df = pd.DataFrame(rows)

    output_dir = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(output_dir, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"[INFO] Method: {args.method_name}")
    print(f"[INFO] Saved decision fusion output to: {args.output_csv}")
    print(f"[INFO] Number of rows: {len(out_df)}")
    print(f"[INFO] Alpha mode: {args.alpha_mode}")
    print(f"[INFO] Fusion mode: {args.fusion_mode}")
    print(f"[INFO] alpha_low={args.alpha_low}, alpha_high={args.alpha_high}")
    print(f"[INFO] fixed_alpha={args.fixed_alpha}, hard_threshold={args.hard_threshold}")
    print(
        f"[INFO] temporal_mu={args.temporal_mu}, "
        f"temporal_init_mode={args.temporal_init_mode}, "
        f"temporal_init_value={args.temporal_init_value}"
    )

    print("\n[INFO] Overall errors:")
    print(f"Focused translation mean : {out_df['focused_translation_error_m'].mean():.6f} m")
    print(f"Explorer translation mean: {out_df['explorer_translation_error_m'].mean():.6f} m")
    print(f"Final translation mean   : {out_df['final_translation_error_m'].mean():.6f} m")

    print(f"Focused rotation geo mean : {out_df['focused_rotation_geo_deg'].mean():.6f} deg")
    print(f"Explorer rotation geo mean: {out_df['explorer_rotation_geo_deg'].mean():.6f} deg")
    print(f"Final rotation geo mean   : {out_df['final_rotation_geo_deg'].mean():.6f} deg")

    # if "Label_t" in out_df.columns:
    #     print("\n[INFO] Final errors by Label_t:")
    #     grouped = out_df.groupby("Label_t")[
    #         [
    #             "final_translation_error_m",
    #             "final_rotation_geo_deg",
    #             "focused_translation_error_m",
    #             "explorer_translation_error_m",
    #             "focused_rotation_geo_deg",
    #             "explorer_rotation_geo_deg",
    #         ]
    #     ].agg(["count", "mean", "median"])
    #     print(grouped)
    if "Label_t" in out_df.columns:
        print("\n[INFO] Final errors by Label_t:")

        def q90(x: pd.Series) -> float:
            return x.quantile(0.90)

        def q95(x: pd.Series) -> float:
            return x.quantile(0.95)

        grouped = out_df.groupby("Label_t")[
            [
                "final_translation_error_m",
                "final_rotation_geo_deg",
                "focused_translation_error_m",
                "explorer_translation_error_m",
                "focused_rotation_geo_deg",
                "explorer_rotation_geo_deg",
            ]
        ].agg(["count", "mean", "median", q90, q95])

        grouped_flat = grouped.copy()
        grouped_flat.columns = [
            f"{metric}_{stat}" for metric, stat in grouped_flat.columns
        ]
        grouped_flat = grouped_flat.reset_index()

        with pd.option_context(
            "display.max_columns", None,
            "display.width", 260,
            "display.max_colwidth", None,
        ):
            print(grouped_flat.to_string(index=False))

        if args.summary_csv is None:
            summary_csv = args.output_csv.replace(".csv", "_by_label_summary.csv")
        else:
            summary_csv = args.summary_csv

        summary_dir = os.path.dirname(os.path.abspath(summary_csv))
        os.makedirs(summary_dir, exist_ok=True)

        grouped_flat.to_csv(summary_csv, index=False)
        print(f"[INFO] Saved by-label summary to: {summary_csv}")

    print("\n[DONE] Decision fusion completed.")


if __name__ == "__main__":
    main()
