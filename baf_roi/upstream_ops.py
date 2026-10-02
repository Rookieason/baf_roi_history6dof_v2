"""Verbatim upstream functions. See docs/LOGIC_CHANGES.md and provenance.json."""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict

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

def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    focused_weight: float = 1.0,
    explorer_weight: float = 1.0,
) -> torch.Tensor:
    """
    Explorer = 1, Focused = 0

    To make the model care more about Focused samples,
    set focused_weight > explorer_weight.
    """
    targets = targets.float()
    base_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(
        targets == 0.0,
        torch.full_like(targets, focused_weight),
        torch.full_like(targets, explorer_weight),
    )
    return (base_loss * weights).mean()
