# model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConfidenceEstimatorConfig:
    """
    Configuration for the behavior confidence estimator.

    Assumed input shape:
        x: [batch, seq_len, input_dim]

    Default input_dim = 6 corresponds to:
        [x, y, z, yaw, pitch, roll]
    """
    input_dim: int = 6
    seq_len: int = 30

    # 1D-CNN backbone
    conv_channels: tuple[int, ...] = (32, 64, 128)
    kernel_sizes: tuple[int, ...] = (5, 5, 3)
    dropout: float = 0.1
    use_batch_norm: bool = True

    # MLP head
    hidden_dim: int = 64

    # Optional output behavior
    return_logits: bool = True


class ConvBlock1D(nn.Module):
    """
    A simple temporal convolution block:
    Conv1d -> BatchNorm1d (optional) -> ReLU -> Dropout
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        # "same-length" temporal padding for odd kernel sizes
        padding = kernel_size // 2

        layers = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not use_batch_norm,
            )
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))

        layers.extend([
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BehavioralConfidenceEstimator(nn.Module):
    """
    1D-CNN-based behavioral confidence estimator.

    Input:
        x: [B, T, D]
            B: batch size
            T: sequence length
            D: pose feature dimension (default 6)

    Output dict:
        {
            "logits": [B],
            "confidence": [B],   # sigmoid(logits), interpreted as Explorer confidence
            "embedding": [B, C], # pooled temporal representation
        }

    Design alignment with thesis:
        - causal 6DoF pose window as input
        - 1D-CNN for local temporal motion features
        - global average pooling for window-level representation
        - fully connected layers for scalar Explorer confidence
    """
    def __init__(self, config: Optional[ConfidenceEstimatorConfig] = None) -> None:
        super().__init__()
        self.config = config or ConfidenceEstimatorConfig()

        channels = [self.config.input_dim, *self.config.conv_channels]
        kernel_sizes = self.config.kernel_sizes

        if len(self.config.conv_channels) != len(kernel_sizes):
            raise ValueError(
                "Length of conv_channels must match length of kernel_sizes. "
                f"Got {len(self.config.conv_channels)} vs {len(kernel_sizes)}."
            )

        conv_blocks = []
        for i in range(len(kernel_sizes)):
            conv_blocks.append(
                ConvBlock1D(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    dropout=self.config.dropout,
                    use_batch_norm=self.config.use_batch_norm,
                )
            )
        self.feature_extractor = nn.Sequential(*conv_blocks)

        final_channels = self.config.conv_channels[-1]

        self.head = nn.Sequential(
            nn.Linear(final_channels, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, D]

        Returns:
            dict with:
                logits: [B]
                confidence: [B]
                embedding: [B, C]
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input x to have shape [B, T, D], got shape {tuple(x.shape)}"
            )

        batch_size, seq_len, input_dim = x.shape

        if input_dim != self.config.input_dim:
            raise ValueError(
                f"Expected input feature dim {self.config.input_dim}, got {input_dim}"
            )

        # Conv1d expects [B, C, T]
        x = x.transpose(1, 2)  # [B, D, T]

        features = self.feature_extractor(x)   # [B, C, T]
        embedding = features.mean(dim=-1)     # global average pooling -> [B, C]

        logits = self.head(embedding).squeeze(-1)      # [B]
        confidence = torch.sigmoid(logits)             # [B]

        return {
            "logits": logits,
            "confidence": confidence,
            "embedding": embedding,
        }

    def predict_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience inference API.
        Returns Explorer confidence in [0, 1].
        """
        return self.forward(x)["confidence"]


def confidence_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy loss using logits.

    Args:
        logits:  [B]
        targets: [B], values in {0, 1} or float labels in [0, 1]

    Returns:
        scalar loss
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have the same shape, "
            f"got {tuple(logits.shape)} vs {tuple(targets.shape)}"
        )

    targets = targets.float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def build_confidence_estimator(
    input_dim: int = 6,
    seq_len: int = 30,
) -> BehavioralConfidenceEstimator:
    """
    Factory function with thesis-aligned defaults.
    """
    config = ConfidenceEstimatorConfig(
        input_dim=input_dim,
        seq_len=seq_len,
        conv_channels=(32, 64, 128),
        kernel_sizes=(5, 5, 3),
        hidden_dim=64,
        dropout=0.1,
        use_batch_norm=True,
    )
    return BehavioralConfidenceEstimator(config)


if __name__ == "__main__":
    # Minimal sanity check
    model = build_confidence_estimator(input_dim=6, seq_len=30)

    dummy_x = torch.randn(8, 30, 6)   # [batch, seq_len, 6DoF]
    out = model(dummy_x)

    print("logits shape:", out["logits"].shape)
    print("confidence shape:", out["confidence"].shape)
    print("embedding shape:", out["embedding"].shape)

    dummy_y = torch.randint(0, 2, (8,)).float()
    loss = confidence_bce_loss(out["logits"], dummy_y)
    print("loss:", float(loss))