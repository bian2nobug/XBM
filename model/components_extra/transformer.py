"""Standard Transformer encoder for encoder-component ablations."""

from __future__ import annotations

import torch
from torch import nn

from .attention import Attention
from .feedforward import FeedForward


class Transformer_Encoder(nn.Module):
    """Pre-normalized Transformer encoder used as a standard-attention baseline."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, *, return_attention: bool = False):
        last_attn = None
        for attn, ff in self.layers:
            attn_out, last_attn = attn(x)
            x = x + attn_out
            x = x + ff(x)
        x = self.norm(x)
        return (x, last_attn) if return_attention else x


TransformerEncoder = Transformer_Encoder

__all__ = ["Transformer_Encoder", "TransformerEncoder"]
