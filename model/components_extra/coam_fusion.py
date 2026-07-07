"""COAM/Pyramid-style same-FOV multiscale fusion used in ablation models."""

from __future__ import annotations

import torch
import torch.nn as nn


class MHSA(nn.Module):
    """Multi-head self-attention block used inside the pyramid fusion module."""

    def __init__(self, dim: int = 1536, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(inner, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm_head = nn.LayerNorm(dim_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        h = self.heads
        q = q.view(B, T, h, -1).transpose(1, 2)
        k = k.view(B, T, h, -1).transpose(1, 2)
        v = v.view(B, T, h, -1).transpose(1, 2)
        q = self.norm_head(q)
        k = self.norm_head(k)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj_drop(self.proj(out))


class FFN(nn.Module):
    """Feed-forward sublayer for pyramid fusion blocks."""

    def __init__(self, dim: int = 1536, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-normalized attention and feed-forward encoder block."""

    def __init__(self, dim: int = 1536, heads: int = 8, dim_head: int = 64, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class IntraScaleSA_KeepTokens(nn.Module):
    """Intra-scale self-attention that returns a scale-level CLS token and the original token set."""

    def __init__(self, dim: int = 1536, N: int = 16, depth: int = 1, heads: int = 8, dim_head: int = 64, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.N = int(N)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, N + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, heads, dim_head, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, M, N = x.shape
        if C != self.dim or N != self.N:
            raise ValueError(f"expected input shape (B,{self.dim},M,{self.N}), got {tuple(x.shape)}")
        x = x.permute(0, 2, 3, 1).contiguous().view(B * M, N, C)
        cls = self.cls.expand(B * M, 1, C)
        seq = torch.cat([cls, x], dim=1) + self.pos
        for block in self.blocks:
            seq = block(seq)
        cls_out = seq[:, 0].view(B, M, C).permute(0, 2, 1)
        tok_out = seq[:, 1:].view(B, M, N, C).permute(0, 3, 1, 2).contiguous()
        return cls_out, tok_out


class InterScaleUpdate(nn.Module):
    """Update lower-resolution tokens using grouped higher-resolution tokens."""

    def __init__(self, dim: int = 1536, depth: int = 1, heads: int = 8, dim_head: int = 64, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, heads, dim_head, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        B, C, M, Nl = low.shape
        Nh = high.size(-1)
        if Nh % Nl != 0:
            raise ValueError(f"high-token count Nh={Nh} must be divisible by low-token count Nl={Nl}")
        group = Nh // Nl
        low_seq = low.permute(0, 2, 3, 1).contiguous()
        high_seq = high.permute(0, 2, 3, 1).contiguous().view(B, M, Nl, group, C)
        seq = torch.cat([low_seq.unsqueeze(3), high_seq], dim=3).view(B * M * Nl, 1 + group, C)
        for block in self.blocks:
            seq = block(seq)
        updated = seq[:, 0]
        return updated.view(B, M, Nl, C).permute(0, 3, 1, 2).contiguous()


class CSMIL_PyramidProgressive(nn.Module):
    """Progressive 20x-to-10x-to-5x same-FOV pyramid fusion.

    Input shape: ``(B, C, M, 21)`` with 21 views ordered as 1 five-magnification
    view, 4 ten-magnification views, and 16 twenty-magnification views.
    Output shape: ``(B, C, M, 1)``.
    """

    def __init__(
        self,
        dim: int = 1536,
        depth_intra: int = 1,
        heads_intra: int = 8,
        dim_head_intra: int = 64,
        mlp_ratio_intra: float = 4.0,
        dropout_intra: float = 0.0,
        depth_inter_20_10: int = 1,
        depth_inter_10_5: int = 1,
        heads_inter: int = 8,
        dim_head_inter: int = 64,
        mlp_ratio_inter: float = 4.0,
        dropout_inter: float = 0.0,
        learnable_fuse: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.intra20 = IntraScaleSA_KeepTokens(dim, N=16, depth=depth_intra, heads=heads_intra, dim_head=dim_head_intra, mlp_ratio=mlp_ratio_intra, dropout=dropout_intra)
        self.intra10 = IntraScaleSA_KeepTokens(dim, N=4, depth=depth_intra, heads=heads_intra, dim_head=dim_head_intra, mlp_ratio=mlp_ratio_intra, dropout=dropout_intra)
        self.intra5 = IntraScaleSA_KeepTokens(dim, N=1, depth=depth_intra, heads=heads_intra, dim_head=dim_head_intra, mlp_ratio=mlp_ratio_intra, dropout=dropout_intra)
        self.inter_20_10 = InterScaleUpdate(dim, depth=depth_inter_20_10, heads=heads_inter, dim_head=dim_head_inter, mlp_ratio=mlp_ratio_inter, dropout=dropout_inter)
        self.inter_10_5 = InterScaleUpdate(dim, depth=depth_inter_10_5, heads=heads_inter, dim_head=dim_head_inter, mlp_ratio=mlp_ratio_inter, dropout=dropout_inter)
        self.learnable_fuse = bool(learnable_fuse)
        if self.learnable_fuse:
            self.w = nn.Parameter(torch.randn(3, 1))
        else:
            self.register_buffer("w_fixed", torch.tensor([[1.0], [1.0], [1.0]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.dim or x.size(-1) != 21:
            raise ValueError(f"expected input shape (B,{self.dim},M,21), got {tuple(x.shape)}")
        f5, f10, f20 = x[..., 0:1], x[..., 1:5], x[..., 5:21]

        cls20, tok20 = self.intra20(f20)
        tok10 = self.inter_20_10(f10, tok20)
        cls10, tok10 = self.intra10(tok10)
        tok5 = self.inter_10_5(f5, tok10)
        cls5, _ = self.intra5(tok5)

        if self.learnable_fuse:
            weights = torch.softmax(self.w, dim=0)
        else:
            weights = self.w_fixed / self.w_fixed.sum()
        fused = weights[0] * cls5 + weights[1] * cls10 + weights[2] * cls20
        return fused.unsqueeze(-1)


__all__ = [
    "MHSA",
    "FFN",
    "EncoderBlock",
    "IntraScaleSA_KeepTokens",
    "InterScaleUpdate",
    "CSMIL_PyramidProgressive",
]
