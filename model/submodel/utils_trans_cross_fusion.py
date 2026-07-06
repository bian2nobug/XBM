import torch
from torch import nn


class AttnPool1D(nn.Module):
    """Attention pooling over a sequence dimension.

    Input:  x (B, L, D)
    Output: out (B, D)
    """
    def __init__(self, dim, hidden=128, dropout=0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        s = self.score(x).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask, -1e9)
        w = torch.softmax(s, dim=-1)
        w = self.drop(w)
        return (w.unsqueeze(-1) * x).sum(dim=1)
