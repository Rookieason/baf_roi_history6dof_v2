# sota_baseline/motion_transformer/model.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import math
import torch
import torch.nn as nn


@dataclass
class MotionTransformerConfig:
    """
    Motion-only Transformer baseline for next-step 6DoF pose prediction.

    Input:
        pose_x: [B, T, 7]
            [normalized x, normalized y, normalized z, qw, qx, qy, qz]

    Output:
        delta_hat_tr: [B, 3]
            predicted normalized translation residual.

        p_hat_tr: [B, 3]
            reconstructed next-step normalized translation.

        q_delta_norm: [B, 4]
            predicted relative quaternion residual in [qw, qx, qy, qz].

        q_hat_norm: [B, 4]
            reconstructed next-step unit quaternion in [qw, qx, qy, qz].
    """

    input_dim: int = 7
    seq_len: int = 30
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    latent_dim: int = 128
    use_last_token: bool = True
    prediction_mode: str = "residual_translation_relative_quaternion"


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Shape:
        input:  [B, T, d_model]
        output: [B, T, d_model]
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [B, T, C], got {tuple(x.shape)}")

        seq_len = x.shape[1]
        if seq_len > self.pe.shape[1]:
            raise ValueError(
                f"Input seq_len={seq_len} exceeds max positional length={self.pe.shape[1]}"
            )

        x = x + self.pe[:, :seq_len, :].to(dtype=x.dtype, device=x.device)
        return self.dropout(x)


def normalize_quaternion_torch(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)


def quaternion_multiply_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product for quaternions in [qw, qx, qy, qz] format.

    q = q1 ⊗ q2
    """
    q1 = normalize_quaternion_torch(q1)
    q2 = normalize_quaternion_torch(q2)

    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    q = torch.stack([w, x, y, z], dim=-1)
    return normalize_quaternion_torch(q)


class MotionTransformerPredictor(nn.Module):
    """
    Motion-only Transformer encoder baseline.

    This model is an adapted baseline inspired by Transformer-based viewport
    prediction, but restricted to the unified motion-only 6DoF pose prediction
    protocol.

    Compared with the first version, this version predicts:
        1. translation residual relative to current translation;
        2. relative quaternion residual composed with current quaternion.

    It does not use gaze, saliency, video content, behavior labels, or decision
    fusion.
    """

    def __init__(self, config: Optional[MotionTransformerConfig] = None) -> None:
        super().__init__()

        if config is None:
            config = MotionTransformerConfig()

        self.config = config

        if config.d_model % config.num_heads != 0:
            raise ValueError(
                f"d_model={config.d_model} must be divisible by num_heads={config.num_heads}"
            )

        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
        )

        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_len=max(config.seq_len, 512),
            dropout=config.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
        )

        self.latent_proj = nn.Sequential(
            nn.Linear(config.d_model, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.translation_residual_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, 3),
        )

        self.quaternion_residual_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, 4),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        final_q_layer = self.quaternion_residual_head[-1]
        if isinstance(final_q_layer, nn.Linear):
            nn.init.zeros_(final_q_layer.weight)
            nn.init.zeros_(final_q_layer.bias)
            final_q_layer.bias.data[0] = 1.0

    def forward(self, pose_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pose_x:
                Tensor with shape [B, T, 7].

        Returns:
            Dictionary with:
                delta_hat_tr:
                    [B, 3], predicted normalized translation residual.

                p_hat_tr:
                    [B, 3], reconstructed normalized translation.

                q_delta_raw:
                    [B, 4], raw relative quaternion head output.

                q_delta_norm:
                    [B, 4], normalized relative quaternion.

                q_hat_norm:
                    [B, 4], reconstructed next-step quaternion.

                current_translation:
                    [B, 3], current normalized translation.

                current_quaternion:
                    [B, 4], current unit quaternion.

                latent:
                    [B, latent_dim], final sequence representation.
        """
        if pose_x.ndim != 3:
            raise ValueError(
                f"Expected pose_x with shape [B, T, C], got {tuple(pose_x.shape)}"
            )

        if pose_x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, got {pose_x.shape[-1]}"
            )

        if pose_x.shape[1] > self.config.seq_len:
            raise ValueError(
                f"Expected seq_len <= {self.config.seq_len}, got {pose_x.shape[1]}"
            )

        current_translation = pose_x[:, -1, :3]
        current_quaternion = normalize_quaternion_torch(pose_x[:, -1, 3:7])

        h = self.input_proj(pose_x)
        h = self.pos_encoder(h)
        h = self.encoder(h)

        if self.config.use_last_token:
            seq_latent = h[:, -1, :]
        else:
            seq_latent = h.mean(dim=1)

        latent = self.latent_proj(seq_latent)

        delta_hat_tr = self.translation_residual_head(latent)
        p_hat_tr = current_translation + delta_hat_tr

        q_delta_raw = self.quaternion_residual_head(latent)
        q_delta_norm = normalize_quaternion_torch(q_delta_raw)

        q_hat_norm = quaternion_multiply_wxyz(
            q_delta_norm,
            current_quaternion,
        )

        return {
            "delta_hat_tr": delta_hat_tr,
            "p_hat_tr": p_hat_tr,
            "q_delta_raw": q_delta_raw,
            "q_delta_norm": q_delta_norm,
            "q_hat_raw": q_hat_norm,
            "q_hat_norm": q_hat_norm,
            "current_translation": current_translation,
            "current_quaternion": current_quaternion,
            "latent": latent,
        }


def build_motion_transformer(
    input_dim: int = 7,
    seq_len: int = 30,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    activation: str = "gelu",
    latent_dim: int = 128,
    use_last_token: bool = True,
) -> MotionTransformerPredictor:
    config = MotionTransformerConfig(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        latent_dim=latent_dim,
        use_last_token=use_last_token,
    )

    return MotionTransformerPredictor(config)


def build_motion_transformer_from_checkpoint_config(
    config_dict: Dict,
) -> MotionTransformerPredictor:
    return build_motion_transformer(
        input_dim=int(config_dict.get("input_dim", 7)),
        seq_len=int(config_dict.get("seq_len", 30)),
        d_model=int(config_dict.get("d_model", 128)),
        num_heads=int(config_dict.get("num_heads", 4)),
        num_layers=int(config_dict.get("num_layers", 2)),
        dim_feedforward=int(config_dict.get("dim_feedforward", 256)),
        dropout=float(config_dict.get("dropout", 0.1)),
        activation=str(config_dict.get("activation", "gelu")),
        latent_dim=int(config_dict.get("latent_dim", 128)),
        use_last_token=bool(config_dict.get("use_last_token", True)),
    )


def config_to_dict(config: MotionTransformerConfig) -> Dict:
    return asdict(config)


def _debug_forward() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_motion_transformer(
        input_dim=7,
        seq_len=30,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ).to(device)

    pose_x = torch.randn(8, 30, 7, device=device)

    # Make the quaternion channels valid enough for debug.
    pose_x[:, :, 3:7] = normalize_quaternion_torch(pose_x[:, :, 3:7])

    out = model(pose_x)

    print("=== MotionTransformer Debug Forward ===")
    print(f"device          : {device}")
    print(f"pose_x          : {tuple(pose_x.shape)}")
    print(f"delta_hat_tr    : {tuple(out['delta_hat_tr'].shape)}")
    print(f"p_hat_tr        : {tuple(out['p_hat_tr'].shape)}")
    print(f"q_delta_raw     : {tuple(out['q_delta_raw'].shape)}")
    print(f"q_delta_norm    : {tuple(out['q_delta_norm'].shape)}")
    print(f"q_hat_norm      : {tuple(out['q_hat_norm'].shape)}")
    print(f"latent          : {tuple(out['latent'].shape)}")
    print(f"q_delta norm    : {torch.linalg.norm(out['q_delta_norm'], dim=-1).mean().item():.6f}")
    print(f"q_hat norm      : {torch.linalg.norm(out['q_hat_norm'], dim=-1).mean().item():.6f}")


if __name__ == "__main__":
    _debug_forward()