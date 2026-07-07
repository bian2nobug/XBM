"""Standard multi-head self-attention used by Transformer ablation modules."""

from __future__ import annotations

import torch
from torch import nn


class Attention(nn.Module):
    """Multi-head self-attention with optional attention-weight exposure."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim_head) * int(heads)
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.scale = self.dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.multi_score = None

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.heads, self.dim_head).transpose(1, 2) for t in qkv]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
        out = self.to_out(out)
        self.multi_score = attn
        return out, attn


__all__ = ["Attention"]
