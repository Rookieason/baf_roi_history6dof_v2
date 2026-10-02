from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExplorerBranchConfig:
    """
    Explorer-specialized prediction branch.

    Thesis-aligned design:
      - Input pose window x_t: [B, T, 7]
        where each timestep is [x, y, z, qw, qx, qy, qz]
      - Optional auxiliary motion features c_t
      - Shared TCN encoder
      - Decoupled translation and rotation heads
      - Translation residual reconstruction
      - Quaternion normalization
    """
    pose_dim: int = 7
    aux_dim: int = 4
    hidden_dim: int = 128
    num_tcn_blocks: int = 3
    kernel_size: int = 3
    dropout: float = 0.1

    # More conservative defaults for first-stage training
    gamma_rot: float = 0.3
    eta_tail: float = 0.1


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
    Align q_pred sign to q_ref to avoid q / -q discontinuity in auxiliary losses or logging.
    Geodesic loss itself is sign-invariant, but this helps keep outputs interpretable.
    """
    dot = torch.sum(q_pred * q_ref, dim=-1, keepdim=True)
    return torch.where(dot < 0.0, -q_pred, q_pred)


def relative_quaternion_target(
    current_quaternion: torch.Tensor,
    target_quaternion: torch.Tensor,
) -> torch.Tensor:
    """
    Compute target relative quaternion:

        delta_q = q_target ⊗ inverse(q_current)

    Quaternion format: [qw, qx, qy, qz].
    """
    q_current = quaternion_normalize(current_quaternion)
    q_target = quaternion_normalize(target_quaternion)

    delta_q = quaternion_multiply(
        q_target,
        quaternion_inverse_unit(q_current),
    )

    delta_q = quaternion_normalize(delta_q)

    # Prefer the short-arc representation with positive scalar part.
    # This is not required for geodesic loss, but stabilizes the residual target convention.
    delta_q = torch.where(delta_q[..., :1] < 0.0, -delta_q, delta_q)

    return delta_q


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=padding
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _chomp(x: torch.Tensor, chomp_size: int) -> torch.Tensor:
        if chomp_size == 0:
            return x
        return x[:, :, :-chomp_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self._chomp(out, self.conv1.padding[0])
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self._chomp(out, self.conv2.padding[0])
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class ExplorerTCNBranch(nn.Module):
    """
    Inputs:
        pose_x: [B, T, 7] = [x, y, z, qw, qx, qy, qz]
        aux_x:  [B, T, A] or [B, A] or None if aux_dim == 0

    Outputs:
        {
            "y_hat": [B, 7],          # [x, y, z, qw, qx, qy, qz]
            "p_hat_tr": [B, 3],
            "delta_hat_tr": [B, 3],
            "q_hat_raw": [B, 4],
            "q_hat_norm": [B, 4],
            "current_translation": [B, 3],
            "latent": [B, H],
        }
    """

    def __init__(self, config: Optional[ExplorerBranchConfig] = None) -> None:
        super().__init__()
        self.config = config or ExplorerBranchConfig()

        total_input_dim = self.config.pose_dim + self.config.aux_dim

        self.input_proj = nn.Conv1d(
            in_channels=total_input_dim,
            out_channels=self.config.hidden_dim,
            kernel_size=1,
        )

        blocks = []
        for i in range(self.config.num_tcn_blocks):
            blocks.append(
                TemporalConvBlock(
                    channels=self.config.hidden_dim,
                    kernel_size=self.config.kernel_size,
                    dilation=2 ** i,
                    dropout=self.config.dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)

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

    def forward(
        self,
        pose_x: torch.Tensor,
        aux_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if pose_x.ndim != 3:
            raise ValueError(f"pose_x must have shape [B, T, {self.config.pose_dim}], got {tuple(pose_x.shape)}")
        if pose_x.shape[-1] != self.config.pose_dim:
            raise ValueError(
                f"Expected pose_x last dim={self.config.pose_dim}, got {pose_x.shape[-1]}"
            )

        batch_size, seq_len, _ = pose_x.shape

        if self.config.aux_dim > 0:
            if aux_x is None:
                raise ValueError(f"aux_x is required because aux_dim={self.config.aux_dim}")

            if aux_x.ndim == 2:
                if aux_x.shape[0] != batch_size or aux_x.shape[1] != self.config.aux_dim:
                    raise ValueError(
                        f"Expected aux_x shape [B, {self.config.aux_dim}], got {tuple(aux_x.shape)}"
                    )
                aux_x = aux_x.unsqueeze(1).expand(-1, seq_len, -1)

            elif aux_x.ndim == 3:
                if aux_x.shape[:2] != pose_x.shape[:2]:
                    raise ValueError("pose_x and aux_x must share the same [B, T] dimensions.")
                if aux_x.shape[-1] != self.config.aux_dim:
                    raise ValueError(
                        f"Expected aux_x last dim={self.config.aux_dim}, got {aux_x.shape[-1]}"
                    )
            else:
                raise ValueError(
                    f"aux_x must have shape [B, A] or [B, T, A], got {tuple(aux_x.shape)}"
                )

            x = torch.cat([pose_x, aux_x], dim=-1)
        else:
            x = pose_x

        current_translation = pose_x[:, -1, :3]
        current_quaternion = pose_x[:, -1, 3:7]
        current_quaternion = quaternion_normalize(current_quaternion)

        x = x.transpose(1, 2)  # [B, D, T]
        x = self.input_proj(x)
        x = self.tcn(x)

        latent = x[:, :, -1]

        delta_hat_tr = self.translation_head(latent)
        p_hat_tr = current_translation + delta_hat_tr

        # q_hat_raw = self.rotation_head(latent)
        # q_hat_norm = F.normalize(q_hat_raw, p=2, dim=-1)

        # y_hat = torch.cat([p_hat_tr, q_hat_norm], dim=-1)

        delta_q_hat_raw = self.rotation_head(latent)
        delta_q_hat_norm = quaternion_normalize(delta_q_hat_raw)

        q_hat_norm = quaternion_multiply(
            delta_q_hat_norm,
            current_quaternion,
        )
        q_hat_norm = quaternion_normalize(q_hat_norm)

        # Optional: align predicted quaternion sign to current quaternion for readability.
        q_hat_norm = quaternion_align_sign(q_hat_norm, current_quaternion)

        y_hat = torch.cat([p_hat_tr, q_hat_norm], dim=-1)

        # return {
        #     "y_hat": y_hat,
        #     "p_hat_tr": p_hat_tr,
        #     "delta_hat_tr": delta_hat_tr,
        #     "q_hat_raw": q_hat_raw,
        #     "q_hat_norm": q_hat_norm,
        #     "current_translation": current_translation,
        #     "latent": latent,
        # }
        return {
            "y_hat": y_hat,
            "p_hat_tr": p_hat_tr,
            "delta_hat_tr": delta_hat_tr,
            "delta_q_hat_raw": delta_q_hat_raw,
            "delta_q_hat_norm": delta_q_hat_norm,
            "q_hat_raw": delta_q_hat_raw,      # optional compatibility alias
            "q_hat_norm": q_hat_norm,
            "current_translation": current_translation,
            "current_quaternion": current_quaternion,
            "latent": latent,
        }

    def predict(
        self,
        pose_x: torch.Tensor,
        aux_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(pose_x, aux_x)["y_hat"]


def explorer_translation_residual_loss(
    delta_hat_tr: torch.Tensor,
    current_translation: torch.Tensor,
    target_translation: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    if not (delta_hat_tr.shape == current_translation.shape == target_translation.shape):
        raise ValueError("Shapes must match.")

    delta_target_tr = target_translation - current_translation
    loss = F.l1_loss(delta_hat_tr, delta_target_tr, reduction="none")
    loss_per_sample = loss.mean(dim=-1)

    if reduction == "mean":
        return loss_per_sample.mean()
    if reduction == "sum":
        return loss_per_sample.sum()
    if reduction == "none":
        return loss_per_sample
    raise ValueError(f"Unsupported reduction: {reduction}")


def quaternion_geodesic_distance(
    q_pred: torch.Tensor,
    q_target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    d_geo(q_pred, q_target) = 2 arccos(|<q_pred, q_target>|)
    """
    if q_pred.shape != q_target.shape:
        raise ValueError(
            f"q_pred and q_target must have same shape, got {tuple(q_pred.shape)} vs {tuple(q_target.shape)}"
        )

    q_pred = F.normalize(q_pred, p=2, dim=-1)
    q_target = F.normalize(q_target, p=2, dim=-1)

    inner = torch.sum(q_pred * q_target, dim=-1).abs()
    inner = torch.clamp(inner, min=0.0, max=1.0 - eps)
    return 2.0 * torch.arccos(inner)


def explorer_rotation_geodesic_loss(
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


def explorer_tail_weight(
    current_quaternion: torch.Tensor,
    prev_quaternion: torch.Tensor,
    eta_tail: float = 0.1,
) -> torch.Tensor:
    d_geo = quaternion_geodesic_distance(current_quaternion, prev_quaternion)
    return 1.0 + eta_tail * d_geo


def explorer_total_loss(
    delta_hat_tr: torch.Tensor,
    current_translation: torch.Tensor,
    target_translation: torch.Tensor,
    q_hat_norm: torch.Tensor,
    target_quaternion: torch.Tensor,
    current_quaternion: torch.Tensor,
    prev_quaternion: torch.Tensor,
    gamma_rot: float = 0.3,
    eta_tail: float = 0.1,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """
    L_E = ω_t^(E) * (L_tr^(E) + γ_E L_rot^(E))
    """
    loss_tr = explorer_translation_residual_loss(
        delta_hat_tr=delta_hat_tr,
        current_translation=current_translation,
        target_translation=target_translation,
        reduction="none" if reduction != "none" else "none",
    )

    loss_rot = explorer_rotation_geodesic_loss(
        q_hat_norm=q_hat_norm,
        target_quaternion=target_quaternion,
        reduction="none" if reduction != "none" else "none",
    )

    weight = explorer_tail_weight(
        current_quaternion=current_quaternion,
        prev_quaternion=prev_quaternion,
        eta_tail=eta_tail,
    )

    loss_total = weight * (loss_tr + gamma_rot * loss_rot)

    if reduction == "mean":
        return {
            "loss_total": loss_total.mean(),
            "loss_tr": loss_tr.mean(),
            "loss_rot": loss_rot.mean(),
            "tail_weight": weight.mean(),
        }
    if reduction == "sum":
        return {
            "loss_total": loss_total.sum(),
            "loss_tr": loss_tr.sum(),
            "loss_rot": loss_rot.sum(),
            "tail_weight": weight.sum(),
        }
    if reduction == "none":
        return {
            "loss_total": loss_total,
            "loss_tr": loss_tr,
            "loss_rot": loss_rot,
            "tail_weight": weight,
        }

    raise ValueError(f"Unsupported reduction: {reduction}")


def build_explorer_branch(
    pose_dim: int = 7,
    aux_dim: int = 4,
    hidden_dim: int = 128,
    num_tcn_blocks: int = 3,
    kernel_size: int = 3,
    dropout: float = 0.1,
    gamma_rot: float = 0.3,
    eta_tail: float = 0.1,
) -> ExplorerTCNBranch:
    config = ExplorerBranchConfig(
        pose_dim=pose_dim,
        aux_dim=aux_dim,
        hidden_dim=hidden_dim,
        num_tcn_blocks=num_tcn_blocks,
        kernel_size=kernel_size,
        dropout=dropout,
        gamma_rot=gamma_rot,
        eta_tail=eta_tail,
    )
    return ExplorerTCNBranch(config)


if __name__ == "__main__":
    model = build_explorer_branch(
        pose_dim=7,
        aux_dim=0,
        hidden_dim=128,
        num_tcn_blocks=3,
        kernel_size=3,
        dropout=0.1,
        gamma_rot=0.3,
        eta_tail=0.1,
    )

    pose_x = torch.randn(8, 30, 7)
    aux_x = None

    target_translation = torch.randn(8, 3)
    target_quaternion = F.normalize(torch.randn(8, 4), p=2, dim=-1)
    current_quaternion = F.normalize(torch.randn(8, 4), p=2, dim=-1)
    prev_quaternion = F.normalize(torch.randn(8, 4), p=2, dim=-1)

    out = model(pose_x, aux_x)

    losses = explorer_total_loss(
        delta_hat_tr=out["delta_hat_tr"],
        current_translation=out["current_translation"],
        target_translation=target_translation,
        q_hat_norm=out["q_hat_norm"],
        target_quaternion=target_quaternion,
        current_quaternion=current_quaternion,
        prev_quaternion=prev_quaternion,
        gamma_rot=0.3,
        eta_tail=0.1,
        reduction="mean",
    )

    print("y_hat shape:", out["y_hat"].shape)
    print("loss_total:", float(losses["loss_total"]))
    print("loss_tr:", float(losses["loss_tr"]))
    print("loss_rot:", float(losses["loss_rot"]))
    print("tail_weight:", float(losses["tail_weight"]))