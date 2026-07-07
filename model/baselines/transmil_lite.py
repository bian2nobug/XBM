"""TransMIL-Lite baseline with Low-Rank QKV, MQA, and GEGLU blocks.

This module is provided as an architectural baseline component. It does not
include task-specific training scripts or pretrained weights.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

# =========================================================
# 1) 你要保留的三点：Low-rank QKV + MQA + GEGLU-FFN
#    ——做成一个可变长序列的 TransformerLite（用于 TransMIL 聚合）
# =========================================================

def posemb_sincos_1d(n: int, dim: int, temperature: int = 10000, dtype=torch.float32, device=None):
    assert dim % 2 == 0
    pos = torch.arange(n, device=device)[:, None]
    i = torch.arange(dim // 2, device=device)[None, :]
    omega = 1.0 / (temperature ** (i / (dim // 2 - 1)))
    angles = pos * omega
    pe = torch.cat([angles.sin(), angles.cos()], dim=1)
    return pe.to(dtype=dtype)

class LowRankQKV(nn.Module):
    """低秩参数化 QKV，配合 MQA：K/V 共享（每个头共用同一 K、V）"""
    def __init__(self, dim, heads, dim_head, rank=128, bias=False):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.down = nn.Linear(dim, rank, bias=bias)
        self.Bq   = nn.Linear(rank, heads * dim_head, bias=bias)
        self.Bk   = nn.Linear(rank, dim_head, bias=bias)
        self.Bv   = nn.Linear(rank, dim_head, bias=bias)

    def forward(self, x):
        z = self.down(x)     # [B,N,r]
        q = self.Bq(z)       # [B,N,h*dh]
        k = self.Bk(z)       # [B,N,dh]
        v = self.Bv(z)       # [B,N,dh]
        return q, k, v

class MultiHeadAttention_MQA_LR(nn.Module):
    """MQA + Low-rank QKV（训练默认不返回注意力，省显存）"""
    def __init__(self, dim, heads=4, dim_head=64, rank=128, dropout=0.1, lowrank_out=True):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.scale = dim_head ** -0.5
        self.qkv = LowRankQKV(dim, heads, dim_head, rank=rank, bias=False)

        inner_q = heads * dim_head
        self.use_lowrank_out = lowrank_out
        if lowrank_out:
            r = min(128, max(32, inner_q // 2))
            self.out_A = nn.Linear(inner_q, r, bias=False)
            self.out_B = nn.Linear(r, dim, bias=False)
        else:
            self.to_out = nn.Linear(inner_q, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    @torch.cuda.amp.autocast(enabled=False)
    def _softmax_stable(self, scores):
        scores = scores.float()
        scores = scores - scores.max(dim=-1, keepdim=True).values
        return torch.softmax(scores, dim=-1)

    def forward(self, x, need_weights: bool=False):
        B, N, _ = x.shape
        h, dh = self.heads, self.dim_head

        q, k, v = self.qkv(x)                              # q:[B,N,h*dh], k/v:[B,N,dh]
        q = q.view(B, N, h, dh).transpose(1, 2)            # [B,h,N,dh]
        k = k.unsqueeze(1).expand(B, h, N, dh)             # 共享K
        v = v.unsqueeze(1).expand(B, h, N, dh)             # 共享V

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,h,N,N]
        attn = self._softmax_stable(scores)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, h * dh)  # [B,N,h*dh]
        out = self.out_B(self.out_A(out)) if self.use_lowrank_out else self.to_out(out)

        return out, (attn if need_weights else None)

class FeedForward_GEGLU(nn.Module):
    """GEGLU-FFN（用 ff_mult 控制宽度）"""
    def __init__(self, dim, ff_mult=0.75, dropout=0.2):
        super().__init__()
        hid = max(64, int(dim * ff_mult))
        self.wg = nn.Linear(dim, hid, bias=False)
        self.wu = nn.Linear(dim, hid, bias=False)
        self.proj = nn.Linear(hid, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        g = F.gelu(self.wg(x))
        u = self.wu(x)
        x = g * u
        x = self.proj(self.dropout(x))
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class TransformerLite(nn.Module):
    """轻量 Transformer：Low-rank QKV + MQA + GEGLU"""
    def __init__(self, dim, depth, heads, dim_head, rank=128, ff_mult=0.75, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            attn = MultiHeadAttention_MQA_LR(dim, heads=heads, dim_head=dim_head, rank=rank, dropout=dropout, lowrank_out=True)
            ffn  = FeedForward_GEGLU(dim, ff_mult=ff_mult, dropout=dropout)
            self.blocks.append(nn.ModuleList([PreNorm(dim, attn), PreNorm(dim, ffn)]))

    def forward(self, x, return_attention: bool=False):
        attn_last = None
        for li, (attn, ffn) in enumerate(self.blocks):
            x_attn, a = attn(x, need_weights=(return_attention and li == len(self.blocks) - 1))
            x = x + x_attn
            x = x + ffn(x)
            if a is not None:
                attn_last = a
        return x, attn_last


# =========================================================
# 2) 把上面的 TransformerLite 塞进 TransMIL 的 CLS 聚合器里
#    ——替换你原来的 TransformerAggregator 即可
# =========================================================

class TransformerAggregatorLite(nn.Module):
    """
    tokens: (B, L, D) -> (B, D)
    用你的 TransformerLite（Low-rank+MQA+GEGLU）替换 nn.TransformerEncoder
    """
    def __init__(self, d_model=512, depth=2, heads=8, dim_head=64,
                 rank=128, ff_mult=0.75, dropout=0.2, add_pos_emb=True):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.add_pos_emb = add_pos_emb
        self.encoder = TransformerLite(
            dim=d_model, depth=depth, heads=heads, dim_head=dim_head,
            rank=rank, ff_mult=ff_mult, dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)
        self.last_attn = None

    def forward(self, tokens, key_padding_mask=None, return_attention: bool=False):
        # 这里先不处理 padding mask（你当前数据一般是固定 M，不需要 pad）
        B, L, D = tokens.shape
        cls = self.cls.expand(B, -1, -1)             # (B,1,D)
        x = torch.cat([cls, tokens], dim=1)          # (B,1+L,D)

        if self.add_pos_emb:
            pe = posemb_sincos_1d(x.size(1), D, dtype=x.dtype, device=x.device)
            x = x + pe.unsqueeze(0)

        x, attn = self.encoder(x, return_attention=return_attention)
        self.last_attn = attn
        out = self.norm(x[:, 0])                     # CLS -> (B,D)
        return out


# =========================================================
# 3) 你的 TransMIL 主模型：只改一行，把 agg 换成 AggregatorLite
# =========================================================

class FC_AttnPool_Fusion(nn.Module):
    """x: (B,C,M,N) -> (B,C,M,1)"""
    def __init__(self, dim):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)
    def forward(self, x):
        B, C, M, N = x.shape
        t = x.permute(0, 2, 3, 1).contiguous()   # (B,M,N,C)
        a = self.scorer(t)                       # (B,M,N,1)
        w = F.softmax(a, dim=2)                  # (B,M,N,1)
        fused = (w * t).sum(dim=2)               # (B,M,C)
        return fused.permute(0, 2, 1).unsqueeze(-1).contiguous()  # (B,C,M,1)

class MultiScaleMultiModalTransMIL_Lite(nn.Module):
    """
    x: (B, 1536+aux_dim, M, N)
    - 多尺度：N->1（你原来的 attention pooling）
    - 多模态：aux -> token 拼进去
    - 聚合：CLS + TransformerLite(低秩MQA+GEGLU)
    """
    def __init__(
        self,
        n_feats=1536,
        n_out=1,
        aux_dim=0,
        use_multiscale=True,
        use_multimodal=True,
        add_token=True,
        view_index=0,
        d_model=512,
        depth=2,
        heads=8,
        dim_head=64,
        rank=128,
        ff_mult=0.75,
        dropout=0.2
    ):
        super().__init__()
        self.n_feats = n_feats
        self.aux_dim = aux_dim
        self.use_multiscale = use_multiscale
        self.use_multimodal = use_multimodal and (aux_dim > 0)
        self.add_token = add_token and self.use_multimodal
        self.view_index = view_index

        self.scale_fusion = FC_AttnPool_Fusion(dim=n_feats)

        self.inst_proj = nn.Sequential(
            nn.Linear(n_feats, d_model),
            nn.LayerNorm(d_model)
        )

        if self.use_multimodal:
            # 建议 aux 先 LN 再投影，稳定
            self.aux_proj = nn.Sequential(
                nn.LayerNorm(aux_dim),
                nn.Linear(aux_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )

        # >>> 关键替换：把原 agg 换成你这套 TransformerLite 版本
        self.agg = TransformerAggregatorLite(
            d_model=d_model, depth=depth, heads=heads, dim_head=dim_head,
            rank=rank, ff_mult=ff_mult, dropout=dropout, add_pos_emb=True
        )

        self.head = nn.Linear(d_model, n_out)

    def split_modalities(self, x):
        img = x[:, :self.n_feats, :, :]
        aux = None
        if self.use_multimodal:
            aux = x[:, self.n_feats:, :, :][..., 0, 0]  # (B, aux_dim)
        return img, aux

    def forward(self, x, return_attention: bool=False):
        img, aux = self.split_modalities(x) if (self.aux_dim > 0) else (x, None)

        if self.use_multiscale:
            img = self.scale_fusion(img)  # (B,1536,M,1)
        else:
            img = img[..., self.view_index:self.view_index+1]

        inst = img.squeeze(-1).permute(0, 2, 1).contiguous()  # (B,M,1536)
        tokens = self.inst_proj(inst)                          # (B,M,D)

        if self.add_token:
            aux_tok = self.aux_proj(aux).unsqueeze(1)         # (B,1,D)
            tokens = torch.cat([aux_tok, tokens], dim=1)      # (B,1+M,D)

        slide_emb = self.agg(tokens, return_attention=return_attention)  # (B,D)
        logits = self.head(slide_emb)                                     # (B,n_out)
        return logits


TransMILLite = MultiScaleMultiModalTransMIL_Lite

__all__ = ["MultiScaleMultiModalTransMIL_Lite", "TransMILLite", "TransformerLite", "TransformerAggregatorLite"]
