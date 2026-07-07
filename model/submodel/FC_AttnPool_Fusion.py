from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_AttnPool_Fusion(nn.Module):
    """Learnable attention pooling over the 21 same-FOV multiscale views."""

    def __init__(self, dim: int = 1536):
        super().__init__()
        self.dim = int(dim)
        self.scorer = nn.Linear(self.dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != self.dim or x.size(-1) != 21:
            raise ValueError(
                f"expected input shape (B,{self.dim},M,21), got {tuple(x.shape)}"
            )
        t = x.permute(0, 2, 3, 1).contiguous()  # (B,M,21,C)
        weights = F.softmax(self.scorer(t), dim=2)  # (B,M,21,1)
        fused = (weights * t).sum(dim=2)  # (B,M,C)
        return fused.permute(0, 2, 1).unsqueeze(-1).contiguous()
