from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FocusedBranchConfig:
    """
    Focused-specialized prediction branch.

    Updated design:
      - Input pose window x_t: [B, T, 7]
        where each timestep is [x, y, z, qw, qx, qy, qz]
      - Translation is predicted as a residual in normalized translation space.
      - Rotation is predicted as a relative quaternion residual.
      - Final quaternion is reconstructed by:
            q_hat = delta_q_hat ⊗ q_current
    """
    input_dim: int = 7
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    use_layer_norm: bool = True


def quaternion_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(q, p=2, dim=-1, eps=eps)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Quaternion conjugate for q = [qw, qx, qy, qz].
    """
    qw = q[..., :1]
    qxyz = q[..., 1:]
    return torch.cat([qw, -qxyz], dim=-1)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product for quaternions in [qw, qx, qy, qz] format.

    q = q1 ⊗ q2
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_inverse_unit(q: torch.Tensor) -> torch.Tensor:
    """
    Inverse of a unit quaternion.
    For normalized quaternions, inverse equals conjugate.
    """
    q = quaternion_normalize(q)
    return quaternion_conjugate(q)


def quaternion_align_sign(q_pred: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
    """
    Align q_pred sign to q_ref for output readability.
    Geodesic loss is sign-invariant, but this avoids unnecessary q / -q flips.
    """
    dot = torch.sum(q_pred * q_ref, dim=-1, keepdim=True)
    return torch.where(dot < 0.0, -q_pred, q_pred)


def quaternion_geodesic_distance(
    q_pred: torch.Tensor,
    q_target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    d_geo(q_pred, q_target) = 2 arccos(|<q_pred, q_target>|)
    Output unit: radians.
    """
    if q_pred.shape != q_target.shape:
        raise ValueError(
            f"q_pred and q_target must have the same shape, got "
            f"{tuple(q_pred.shape)} vs {tuple(q_target.shape)}"
        )

    q_pred = quaternion_normalize(q_pred)
    q_target = quaternion_normalize(q_target)

    inner = torch.sum(q_pred * q_target, dim=-1).abs()
    inner = torch.clamp(inner, min=0.0, max=1.0 - eps)

    return 2.0 * torch.arccos(inner)


class FocusedResidualLSTM(nn.Module):
    """
    Focused-specialized prediction branch.

    Input:
        x: [B, T, 7] = [x, y, z, qw, qx, qy, qz]

    Output:
        {
            "y_hat": [B, 7],
            "p_hat_tr": [B, 3],
            "delta_hat_tr": [B, 3],
            "delta_q_hat_raw": [B, 4],
            "delta_q_hat_norm": [B, 4],
            "q_hat_norm": [B, 4],
            "current_translation": [B, 3],
            "prev_translation": [B, 3],
            "current_quaternion": [B, 4],
            "prev_quaternion": [B, 4],
            "latent": [B, H],
        }
    """

    def __init__(self, config: Optional[FocusedBranchConfig] = None) -> None:
        super().__init__()
        self.config = config or FocusedBranchConfig()

        lstm_dropout = self.config.dropout if self.config.num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.use_layer_norm = self.config.use_layer_norm
        if self.use_layer_norm:
            self.latent_norm = nn.LayerNorm(self.config.hidden_dim)

        self.translation_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 3),
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, T, D], got {tuple(x.shape)}")

        _, seq_len, input_dim = x.shape
        if input_dim != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, got {input_dim}"
            )
        if seq_len < 2:
            raise ValueError(
                "Focused branch requires at least 2 timesteps to compute smoothness loss."
            )

        current_translation = x[:, -1, :3]
        prev_translation = x[:, -2, :3]

        current_quaternion = quaternion_normalize(x[:, -1, 3:7])
        prev_quaternion = quaternion_normalize(x[:, -2, 3:7])

        _, (h_n, _) = self.encoder(x)
        latent = h_n[-1]

        if self.use_layer_norm:
            latent = self.latent_norm(latent)

        delta_hat_tr = self.translation_head(latent)
        p_hat_tr = current_translation + delta_hat_tr

        delta_q_hat_raw = self.rotation_head(latent)
        delta_q_hat_norm = quaternion_normalize(delta_q_hat_raw)

        q_hat_norm = quaternion_multiply(delta_q_hat_norm, current_quaternion)
        q_hat_norm = quaternion_normalize(q_hat_norm)
        q_hat_norm = quaternion_align_sign(q_hat_norm, current_quaternion)

        y_hat = torch.cat([p_hat_tr, q_hat_norm], dim=-1)

        return {
            "y_hat": y_hat,
            "p_hat_tr": p_hat_tr,
            "delta_hat_tr": delta_hat_tr,
            "delta_q_hat_raw": delta_q_hat_raw,
            "delta_q_hat_norm": delta_q_hat_norm,
            "q_hat_raw": delta_q_hat_raw,
            "q_hat_norm": q_hat_norm,
            "current_translation": current_translation,
            "prev_translation": prev_translation,
            "current_quaternion": current_quaternion,
            "prev_quaternion": prev_quaternion,
            "latent": latent,
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["y_hat"]


def focused_translation_residual_loss(
    delta_hat_tr: torch.Tensor,
    current_translation: torch.Tensor,
    target_translation: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Translation residual loss:
        target_delta = p_(t+1) - p_t
    """
    if not (delta_hat_tr.shape == current_translation.shape == target_translation.shape):
        raise ValueError("Translation shapes must match.")

    target_delta_tr = target_translation - current_translation
    loss = F.l1_loss(delta_hat_tr, target_delta_tr, reduction="none")
    loss_per_sample = loss.mean(dim=-1)

    if reduction == "mean":
        return loss_per_sample.mean()
    if reduction == "sum":
        return loss_per_sample.sum()
    if reduction == "none":
        return loss_per_sample

    raise ValueError(f"Unsupported reduction: {reduction}")


def focused_rotation_geodesic_loss(
    q_hat_norm: torch.Tensor,
    target_quaternion: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    d = quaternion_geodesic_distance(q_hat_norm, target_quaternion)

    if reduction == "mean":
        return d.mean()
    if reduction == "sum":
        return d.sum()
    if reduction == "none":
        return d

    raise ValueError(f"Unsupported reduction: {reduction}")


def focused_smoothness_loss(
    delta_hat_tr: torch.Tensor,
    current_translation: torch.Tensor,
    prev_translation: torch.Tensor,
    q_hat_norm: torch.Tensor,
    current_quaternion: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focused smoothness regularization.

    Translation:
        predicted delta should be close to the current velocity.

    Rotation:
        predicted quaternion should not move too far away from q_t.
    """
    current_vel_tr = current_translation - prev_translation
    loss_tr = F.l1_loss(delta_hat_tr, current_vel_tr, reduction="none").mean(dim=-1)

    loss_rot = quaternion_geodesic_distance(q_hat_norm, current_quaternion)

    loss = loss_tr + loss_rot

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    raise ValueError(f"Unsupported reduction: {reduction}")


def focused_total_loss(
    delta_hat_tr: torch.Tensor,
    current_translation: torch.Tensor,
    prev_translation: torch.Tensor,
    target_translation: torch.Tensor,
    q_hat_norm: torch.Tensor,
    current_quaternion: torch.Tensor,
    target_quaternion: torch.Tensor,
    lambda_smooth: float = 0.1,
    lambda_tr: float = 1.0,
    lambda_rot: float = 0.3,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """
    Final Focused loss:
        L_F = lambda_tr * L_tr + lambda_rot * L_rot + lambda_smooth * L_smooth
    """
    loss_tr = focused_translation_residual_loss(
        delta_hat_tr=delta_hat_tr,
        current_translation=current_translation,
        target_translation=target_translation,
        reduction="none" if reduction != "none" else "none",
    )

    loss_rot = focused_rotation_geodesic_loss(
        q_hat_norm=q_hat_norm,
        target_quaternion=target_quaternion,
        reduction="none" if reduction != "none" else "none",
    )

    loss_smooth = focused_smoothness_loss(
        delta_hat_tr=delta_hat_tr,
        current_translation=current_translation,
        prev_translation=prev_translation,
        q_hat_norm=q_hat_norm,
        current_quaternion=current_quaternion,
        reduction="none" if reduction != "none" else "none",
    )

    loss_total = lambda_tr * loss_tr + lambda_rot * loss_rot + lambda_smooth * loss_smooth

    if reduction == "mean":
        return {
            "loss_total": loss_total.mean(),
            "loss_tr": loss_tr.mean(),
            "loss_rot": loss_rot.mean(),
            "loss_smooth": loss_smooth.mean(),
        }
    if reduction == "sum":
        return {
            "loss_total": loss_total.sum(),
            "loss_tr": loss_tr.sum(),
            "loss_rot": loss_rot.sum(),
            "loss_smooth": loss_smooth.sum(),
        }
    if reduction == "none":
        return {
            "loss_total": loss_total,
            "loss_tr": loss_tr,
            "loss_rot": loss_rot,
            "loss_smooth": loss_smooth,
        }

    raise ValueError(f"Unsupported reduction: {reduction}")


if __name__ == "__main__":
    model = FocusedResidualLSTM()

    x = torch.randn(8, 30, 7)
    x[:, :, 3:7] = F.normalize(x[:, :, 3:7], p=2, dim=-1)

    target_translation = torch.randn(8, 3)
    target_quaternion = F.normalize(torch.randn(8, 4), p=2, dim=-1)

    out = model(x)

    losses = focused_total_loss(
        delta_hat_tr=out["delta_hat_tr"],
        current_translation=out["current_translation"],
        prev_translation=out["prev_translation"],
        target_translation=target_translation,
        q_hat_norm=out["q_hat_norm"],
        current_quaternion=out["current_quaternion"],
        target_quaternion=target_quaternion,
        lambda_smooth=0.1,
        lambda_tr=1.0,
        lambda_rot=0.3,
        reduction="mean",
    )

    print("y_hat shape:", out["y_hat"].shape)
    print("p_hat_tr shape:", out["p_hat_tr"].shape)
    print("q_hat_norm shape:", out["q_hat_norm"].shape)
    print("loss_total:", float(losses["loss_total"]))
    print("loss_tr:", float(losses["loss_tr"]))
    print("loss_rot:", float(losses["loss_rot"]))
    print("loss_smooth:", float(losses["loss_smooth"]))