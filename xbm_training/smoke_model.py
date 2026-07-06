from __future__ import annotations

import torch
import torch.nn as nn


class SmokeMIL(nn.Module):
    """Tiny model used only to check that the training package runs end to end."""

    def __init__(self, in_channels: int = 12, num_classes: int = 1):
        super().__init__()
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))
        return self.head(pooled)
