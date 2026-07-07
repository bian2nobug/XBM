"""Cross-modal multi-head attention blocks used by XBM."""

from __future__ import annotations

import torch
from torch import nn


class CrossModalMultiHeadAttention(nn.Module):
    """Multi-head attention where query tokens attend to key/value tokens."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.embed_dim = int(embed_dim)
        self.head_dim = self.embed_dim // self.num_heads
        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        batch_size = query.size(0)
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        output = self.out_proj(attended)
        output = self.layer_norm(output + attended)
        return output, attention_weights


class MultiLayerCrossModalAttention(nn.Module):
    """Stacked cross-modal attention layers."""

    def __init__(self, num_layers: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossModalMultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        attention_weights = []
        for layer in self.layers:
            query, attn = layer(query, key, value)
            attention_weights.append(attn)
        return query, attention_weights
