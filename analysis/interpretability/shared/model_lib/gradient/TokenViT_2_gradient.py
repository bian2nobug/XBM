# TokenViT_2_gradient.py
# TokenViT module with gradient tracking support
# Used for Attention x Gradient interpretability analysis

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


# ---------------- positional encoding ----------------
def posemb_sincos_1d(n, dim, temperature: int = 10000, dtype=torch.float32):
    assert dim % 2 == 0
    pos = torch.arange(n)[:, None]
    i = torch.arange(dim // 2)[None, :]
    omega = 1.0 / (temperature ** (i / (dim // 2 - 1)))
    angles = pos * omega
    pe = torch.cat([angles.sin(), angles.cos()], dim=1)
    return pe.type(dtype)


# ---------------- low-rank QKV + MQA ----------------
class LowRankQKV(nn.Module):
    def __init__(self, dim, heads, dim_head, rank=128, bias=False):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.down = nn.Linear(dim, rank, bias=bias)
        self.Bq = nn.Linear(rank, heads * dim_head, bias=bias)
        self.Bk = nn.Linear(rank, dim_head, bias=bias)
        self.Bv = nn.Linear(rank, dim_head, bias=bias)

    def forward(self, x):
        z = self.down(x)
        q = self.Bq(z)
        k = self.Bk(z)
        v = self.Bv(z)
        return q, k, v


class MultiHeadAttention_MQA_LR(nn.Module):
    """MQA attention module with gradient tracking support"""
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

    def _maybe_pack_attn(self, attn, need_weights, topk, cpu_offload, for_gradient=False):
        if not need_weights:
            return None
        if for_gradient:
            A = attn
            A.retain_grad()
            return A
        else:
            with torch.no_grad():
                A = attn.detach()
                if topk is not None and topk > 0 and topk < A.size(-1):
                    v, i = torch.topk(A, k=topk, dim=-1)
                    if cpu_offload:
                        return {"indices": i.cpu(), "values": v.cpu()}
                    return {"indices": i, "values": v}
                else:
                    if cpu_offload:
                        return A.cpu()
                    return A

    def forward(self, x, need_weights=False, topk=None, cpu_offload=True, for_gradient=False):
        B, N, _ = x.shape
        h, dh = self.heads, self.dim_head

        q, k, v = self.qkv(x)
        q = q.view(B, N, h, dh).transpose(1, 2)
        k = k.unsqueeze(1).expand(B, h, N, dh)
        v = v.unsqueeze(1).expand(B, h, N, dh)

        scores = torch.matmul(q, k.transpose(-1, -2)) * (dh ** -0.5)
        attn = self._softmax_stable(scores)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, h * dh)
        if self.use_lowrank_out:
            out = self.out_B(self.out_A(out))
        else:
            out = self.to_out(out)

        packed = self._maybe_pack_attn(attn, need_weights, topk, cpu_offload, for_gradient)
        return out, packed


# ---------------- low-rank QKV, standard multi-head MHA ----------------
class LowRankQKV_FullMHA(nn.Module):
    def __init__(self, dim, heads, dim_head, rank=128, bias=False):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.down = nn.Linear(dim, rank, bias=bias)
        self.Bq = nn.Linear(rank, heads * dim_head, bias=bias)
        self.Bk = nn.Linear(rank, heads * dim_head, bias=bias)
        self.Bv = nn.Linear(rank, heads * dim_head, bias=bias)

    def forward(self, x):
        z = self.down(x)
        q = self.Bq(z)
        k = self.Bk(z)
        v = self.Bv(z)
        return q, k, v


class MultiHeadAttention_LR_MHA(nn.Module):
    """Standard MHA attention module with gradient tracking support"""
    def __init__(self, dim, heads=4, dim_head=64, rank=128, dropout=0.1, lowrank_out=True):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.scale = dim_head ** -0.5
        self.qkv = LowRankQKV_FullMHA(dim, heads, dim_head, rank=rank, bias=False)

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

    def _maybe_pack_attn(self, attn, need_weights, topk, cpu_offload, for_gradient=False):
        if not need_weights:
            return None
        if for_gradient:
            A = attn
            A.retain_grad()
            return A
        else:
            with torch.no_grad():
                A = attn.detach()
                if topk is not None and topk > 0 and topk < A.size(-1):
                    v, i = torch.topk(A, k=topk, dim=-1)
                    if cpu_offload:
                        return {"indices": i.cpu(), "values": v.cpu()}
                    return {"indices": i, "values": v}
                else:
                    if cpu_offload:
                        return A.cpu()
                    return A

    def forward(self, x, need_weights=False, topk=None, cpu_offload=True, for_gradient=False):
        B, N, _ = x.shape
        h, dh = self.heads, self.dim_head

        q, k, v = self.qkv(x)
        q = q.view(B, N, h, dh).transpose(1, 2)
        k = k.view(B, N, h, dh).transpose(1, 2)
        v = v.view(B, N, h, dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self._softmax_stable(scores)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, h * dh)

        if self.use_lowrank_out:
            out = self.out_B(self.out_A(out))
        else:
            out = self.to_out(out)

        packed = self._maybe_pack_attn(attn, need_weights, topk, cpu_offload, for_gradient)
        return out, packed


# ---------------- GEGLU-FFN ----------------
class FeedForward_GEGLU(nn.Module):
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


# ---------------- traditional FFN ----------------
class FeedForward(nn.Module):
    def __init__(self, dim, ff_mult=4.0, dropout=0.2):
        super().__init__()
        hid = max(64, int(dim * ff_mult))
        self.fc1 = nn.Linear(dim, hid, bias=False)
        self.fc2 = nn.Linear(hid, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ---------------- pre-norm wrapper ----------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# ---------------- lightweight Transformer encoder ----------------
class TransformerLite(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=None,
                 rank=128, ff_mult=0.75, dropout=0.2, share_block=False,
                 use_geglu=True, use_MQA=True):
        super().__init__()
        self.depth = depth
        self.share_block = share_block
        self.use_geglu = use_geglu
        self.use_MQA = use_MQA

        def make_block():
            if self.use_MQA:
                attn = MultiHeadAttention_MQA_LR(
                    dim, heads, dim_head, rank=rank, dropout=dropout, lowrank_out=True)
            else:
                attn = MultiHeadAttention_LR_MHA(
                    dim, heads, dim_head, rank=rank, dropout=dropout, lowrank_out=True)

            if self.use_geglu:
                ffn = FeedForward_GEGLU(dim, ff_mult=ff_mult, dropout=dropout)
            else:
                ffn = FeedForward(dim, ff_mult=ff_mult, dropout=dropout)

            return nn.ModuleList([PreNorm(dim, attn), PreNorm(dim, ffn)])

        if share_block:
            self.block = make_block()
        else:
            self.blocks = nn.ModuleList([make_block() for _ in range(depth)])

    def forward(self, x, return_attention=False, last_only=True,
                topk=None, cpu_offload=True, for_gradient=False):
        attns = []

        def run_block(x, block, need_weights: bool):
            attn_mod, ffn_mod = block
            x_attn, attn_w = attn_mod(x, need_weights=need_weights, topk=topk,
                                       cpu_offload=cpu_offload, for_gradient=for_gradient)
            x = x + x_attn
            x = x + ffn_mod(x)
            return x, attn_w

        if self.share_block:
            for li in range(self.depth):
                need_w = return_attention and ((not last_only) or (li == self.depth - 1))
                x, a = run_block(x, self.block, need_w)
                if return_attention and a is not None:
                    attns.append(a)
        else:
            for li, blk in enumerate(self.blocks):
                need_w = return_attention and ((not last_only) or (li == self.depth - 1))
                x, a = run_block(x, blk, need_w)
                if return_attention and a is not None:
                    attns.append(a)

        if not return_attention or len(attns) == 0:
            attn_all = None
        else:
            attn_all = attns[0] if len(attns) == 1 else attns
        return x, attn_all


# ---------------- TokenViT ----------------
class TokenViT(nn.Module):
    def __init__(self, in_dim=1536, num_tokens=20000, dim=512, depth=2, heads=4,
                 dim_head=64, mlp_dim=1024, use_geglu=True, use_MQA=True,
                 num_classes=3, pool_stride=1, for_gradient=False):
        super().__init__()
        assert pool_stride == 1, "pool_stride can only be 1"
        self.num_tokens = num_tokens
        self.for_gradient = for_gradient

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim, bias=False),
            nn.LayerNorm(dim),
        )
        self.register_buffer("pos_emb_1d", posemb_sincos_1d(num_tokens, dim), persistent=False)

        self.transformer = TransformerLite(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
            rank=128, ff_mult=0.75, dropout=0.2, share_block=False,
            use_geglu=use_geglu, use_MQA=use_MQA
        )
        self.to_latent = nn.Identity()
        self.head = nn.Linear(dim, num_classes, bias=False)
        self.multi_score = None

    def forward(self, x, return_attention: bool = False, last_only: bool = True,
                topk: Optional[int] = None, cpu_offload: bool = True,
                for_gradient: bool = False):
        B, N, _ = x.shape
        assert N == self.num_tokens
        x = self.input_proj(x)
        x = x + self.pos_emb_1d[:N].to(x.device, dtype=x.dtype)

        # use for_gradient from the instance attribute or the parameter
        use_for_gradient = for_gradient or self.for_gradient

        x, attn = self.transformer(
            x,
            return_attention=return_attention,
            last_only=last_only,
            topk=topk,
            cpu_offload=cpu_offload,
            for_gradient=use_for_gradient
        )
        x = self.to_latent(x)
        self.multi_score = attn
        return x, attn

