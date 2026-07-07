"""Feed-forward block used by legacy Transformer ablation modules."""

from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    """Two-layer Transformer feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["FeedForward"]
