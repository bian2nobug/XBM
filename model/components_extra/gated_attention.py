"""Gated attention block for MIL-style auxiliary models."""

from __future__ import annotations

import torch
from torch import nn


class Gated_attention(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, output_dim: int = 1,
                 dropout: bool = False, p_dropout_atn: float = 0.25):
        super().__init__()
        a_layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        b_layers = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        if dropout:
            a_layers.append(nn.Dropout(p_dropout_atn))
            b_layers.append(nn.Dropout(p_dropout_atn))
        self.attention_a = nn.Sequential(*a_layers)
        self.attention_b = nn.Sequential(*b_layers)
        self.attention_c = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention_c(self.attention_a(x) * self.attention_b(x))


GatedAttention = Gated_attention

__all__ = ["Gated_attention", "GatedAttention"]
