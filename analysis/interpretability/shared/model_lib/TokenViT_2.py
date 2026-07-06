import os
import torch
from torch import nn
import torch.nn.functional as F
# optional-typing compatible with Python 3.8/3.9
from typing import Optional

# ---------------- positional encoding ----------------
def posemb_sincos_1d(n, dim, temperature: int = 10000, dtype=torch.float32):
    assert dim % 2 == 0
    pos = torch.arange(n)[:, None]
    i   = torch.arange(dim // 2)[None, :]
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
        self.Bq   = nn.Linear(rank, heads * dim_head, bias=bias)
        self.Bk   = nn.Linear(rank, dim_head, bias=bias)
        self.Bv   = nn.Linear(rank, dim_head, bias=bias)

    def forward(self, x):
        z = self.down(x)                     # [B,N,r]
        q = self.Bq(z)                       # [B,N,h*dh]
        k = self.Bk(z)                       # [B,N,dh]
        v = self.Bv(z)                       # [B,N,dh]
        return q, k, v

class MultiHeadAttention_MQA_LR(nn.Module):
    """
    Adds need_weights / topk / cpu_offload:
      - need_weights=False: do not return attention (training default)
      - topk: return only the top-k weights per row (indices, values), avoiding the N x N huge matrix
      - cpu_offload: move attention (or its topk) to CPU to free GPU memory
    """
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

    @torch.cuda.amp.autocast(enabled=False)   # softmax is more stable in float32
    def _softmax_stable(self, scores):
        scores = scores.float()
        scores = scores - scores.max(dim=-1, keepdim=True).values
        return torch.softmax(scores, dim=-1)

    def _maybe_pack_attn(self, attn, need_weights, topk, cpu_offload):
        if not need_weights:
            return None
        with torch.no_grad():
            A = attn.detach()  # detach gradient, purely for visualization/analysis
            if topk is not None and topk > 0 and topk < A.size(-1):
                v, i = torch.topk(A, k=topk, dim=-1)  # [B,H,N,topk]
                if cpu_offload:
                    return {"indices": i.cpu(), "values": v.cpu()}
                return {"indices": i, "values": v}
            else:
                if cpu_offload:
                    return A.cpu()
                return A

    def forward(self, x, need_weights=False, topk=None, cpu_offload=True):
        B, N, _ = x.shape
        h, dh = self.heads, self.dim_head

        q, k, v = self.qkv(x)                        # q:[B,N,h*dh], k/v:[B,N,dh]
        q = q.view(B, N, h, dh).transpose(1, 2)      # [B,h,N,dh]
        k = k.unsqueeze(1).expand(B, h, N, dh)       # shared K
        v = v.unsqueeze(1).expand(B, h, N, dh)       # shared V

        scores = torch.matmul(q, k.transpose(-1, -2)) * (dh ** -0.5)  # [B,h,N,N]
        attn = self._softmax_stable(scores)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, h*dh)  # [B,N,h*dh]
        if self.use_lowrank_out:
            out = self.out_B(self.out_A(out))
        else:
            out = self.to_out(out)

        packed = self._maybe_pack_attn(attn, need_weights, topk, cpu_offload)
        return out, packed

# ---------------- low-rank QKV, standard multi-head MHA (does not share K,V) ----------------
class LowRankQKV_FullMHA(nn.Module):
    """
    Low-rank parameterized Q,K,V:
      x -> Linear(dim -> rank)
        -> Bq: rank -> (heads * dim_head)
        -> Bk: rank -> (heads * dim_head)
        -> Bv: rank -> (heads * dim_head)
    This yields standard multi-head Q/K/V, not MQA.
    """
    def __init__(self, dim, heads, dim_head, rank=128, bias=False):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.down = nn.Linear(dim, rank, bias=bias)
        self.Bq   = nn.Linear(rank, heads * dim_head, bias=bias)
        self.Bk   = nn.Linear(rank, heads * dim_head, bias=bias)
        self.Bv   = nn.Linear(rank, heads * dim_head, bias=bias)

    def forward(self, x):
        # x: [B, N, dim]
        z = self.down(x)                     # [B, N, r]
        q = self.Bq(z)                       # [B, N, h*dh]
        k = self.Bk(z)                       # [B, N, h*dh]
        v = self.Bv(z)                       # [B, N, h*dh]
        return q, k, v


class MultiHeadAttention_LR_MHA(nn.Module):
    """
    Uses only low-rank (Q/K/V projection matrices are low-rank factorized); attention is standard multi-head MHA:
      - K,V are no longer shared (no longer MQA)
      - still keeps need_weights / topk / cpu_offload / lowrank_out
    """
    def __init__(self, dim, heads=4, dim_head=64,
                 rank=128, dropout=0.1, lowrank_out=True):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.scale = dim_head ** -0.5

        # low-rank Q,K,V (standard multi-head)
        self.qkv = LowRankQKV_FullMHA(dim, heads, dim_head, rank=rank, bias=False)

        inner_q = heads * dim_head
        self.use_lowrank_out = lowrank_out
        if lowrank_out:
            # optionally low-rank factorize the output too
            r = min(128, max(32, inner_q // 2))
            self.out_A = nn.Linear(inner_q, r, bias=False)
            self.out_B = nn.Linear(r, dim, bias=False)
        else:
            self.to_out = nn.Linear(inner_q, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    @torch.cuda.amp.autocast(enabled=False)   # softmax is more stable in float32
    def _softmax_stable(self, scores):
        scores = scores.float()
        scores = scores - scores.max(dim=-1, keepdim=True).values
        return torch.softmax(scores, dim=-1)

    def _maybe_pack_attn(self, attn, need_weights, topk, cpu_offload):
        if not need_weights:
            return None
        with torch.no_grad():
            A = attn.detach()  # analysis/visualization only, detach gradient
            if topk is not None and topk > 0 and topk < A.size(-1):
                v, i = torch.topk(A, k=topk, dim=-1)  # [B,H,N,topk]
                if cpu_offload:
                    return {"indices": i.cpu(), "values": v.cpu()}
                return {"indices": i, "values": v}
            else:
                if cpu_offload:
                    return A.cpu()
                return A

    def forward(self, x, need_weights=False, topk=None, cpu_offload=True):
        """
        x: [B, N, dim]
        Returns:
          out:   [B, N, dim]
          packed: None or
                  - full attention [B, H, N, N] (may be on CPU)
                  - or {'indices': [B,H,N,topk], 'values': [B,H,N,topk]}
        """
        B, N, _ = x.shape
        h, dh = self.heads, self.dim_head

        # ---- low-rank Q,K,V ----
        q, k, v = self.qkv(x)                        # [B, N, h*dh] * 3
        q = q.view(B, N, h, dh).transpose(1, 2)      # [B, h, N, dh]
        k = k.view(B, N, h, dh).transpose(1, 2)      # [B, h, N, dh]
        v = v.view(B, N, h, dh).transpose(1, 2)      # [B, h, N, dh]

        # ---- attention ----
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, h, N, N]
        attn = self._softmax_stable(scores)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                  # [B, h, N, dh]
        out = out.transpose(1, 2).contiguous().view(B, N, h * dh)  # [B, N, h*dh]

        # ---- output projection (optional low-rank) ----
        if self.use_lowrank_out:
            out = self.out_B(self.out_A(out))
        else:
            out = self.to_out(out)

        packed = self._maybe_pack_attn(attn, need_weights, topk, cpu_offload)
        return out, packed

# ---------------- GEGLU-FFN (scalable width) ----------------
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
        x = F.relu(x)        # use ReLU here (the most standard Transformer form)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
# ---------------- pre-norm wrapper (passes through kwargs) ----------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# ---------------- lightweight Transformer encoder ----------------
class TransformerLite(nn.Module):
    """
    Adds forward(..., return_attention=False, last_only=True, topk=None, cpu_offload=True)
    - return_attention=False: do not return attention (training default)
    - last_only=True: keep only the last layer's attention
    - topk: return only the top-k per row, avoiding N x N
    - cpu_offload: move attention or its topk to CPU
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=None,
                 rank=128, ff_mult=0.75, dropout=0.2, share_block=False,use_geglu=True,use_MQA = True):
        super().__init__()

        self.depth = depth
        self.share_block = share_block
        self.use_geglu = use_geglu
        self.use_MQA = use_MQA

        def make_block():
            if self.use_MQA:
                attn = MultiHeadAttention_MQA_LR(
                    dim, heads, dim_head,
                    rank=rank, dropout=dropout, lowrank_out=True
                )
            else:
                attn = MultiHeadAttention_LR_MHA(
                    dim, heads, dim_head,
                    rank=rank, dropout=dropout, lowrank_out=True
                )

            if self.use_geglu:
                ffn = FeedForward_GEGLU(dim, ff_mult=ff_mult, dropout=dropout)
            else:
                # standard FFN, also uses ff_mult to control hidden dim
                ffn = FeedForward(dim, ff_mult=ff_mult, dropout=dropout)

            return nn.ModuleList([PreNorm(dim, attn), PreNorm(dim, ffn)])
        

        if share_block:
            self.block = make_block()
        else:
            self.blocks = nn.ModuleList([make_block() for _ in range(depth)])

    def forward(self, x, return_attention=False, last_only=True,
                topk=None, cpu_offload=True):
        attns = []

        def run_block(x, block, need_weights: bool):
            attn_mod, ffn_mod = block
            x_attn, attn_w = attn_mod(x, need_weights=need_weights, topk=topk, cpu_offload=cpu_offload)
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
            # when last_only=True, attns has a single element; keep the return type consistent
            attn_all = attns[0] if len(attns) == 1 else attns
        return x, attn_all

# ---------------- TokenViT (adds the return_attention interface) ----------------
class TokenViT(nn.Module):
    def __init__(self, in_dim=1536, num_tokens=20000, dim=512, depth=2, heads=4,
                 dim_head=64, mlp_dim=1024,use_geglu = True,use_MQA = True,num_classes=3, pool_stride=1):
        super().__init__()
        assert pool_stride == 1, "pool_stride can only be 1"
        self.num_tokens = num_tokens

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim, bias=False),
            nn.LayerNorm(dim),
        )
        self.register_buffer("pos_emb_1d", posemb_sincos_1d(num_tokens, dim), persistent=False)

        self.transformer = TransformerLite(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
            rank=128, ff_mult=0.75, dropout=0.2, share_block=False,use_geglu=use_geglu,use_MQA = use_MQA
        )
        self.to_latent = nn.Identity()
        self.head = nn.Linear(dim, num_classes, bias=False)
        self.multi_score = None

    def forward(self, x,
                return_attention: bool=False,   # training default False; set True for visualization
                last_only: bool=True,           # return only the last layer's attention
                topk: Optional[int] = None,     # 3.8/3.9 compatible: int | None -> Optional[int]
                cpu_offload: bool=True):        # move attention to CPU to save GPU memory
        B, N, _ = x.shape
        assert N == self.num_tokens
        x = self.input_proj(x)
        x = x + self.pos_emb_1d[:N].to(x.device, dtype=x.dtype)

        x, attn = self.transformer(
            x,
            return_attention=return_attention,
            last_only=last_only,
            topk=topk,
            cpu_offload=cpu_offload
        )
        x = self.to_latent(x)
        self.multi_score = attn
        return x, attn



'''
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "model_lib"))
import torch
from TokenViT_2 import TokenViT


B, N, D = 1, 20000, 1536                 # default input shape (large memory usage, B=1)

model = TokenViT(in_dim=D, num_tokens=N,
                     dim=512, depth=2, heads=2, dim_head=256, mlp_dim=1024,
                     num_classes=3, pool_stride=1)
# if memory is tight, use half precision: x = torch.randn(B,N,D, dtype=torch.float16).to(torch.float32)
x = torch.randn(B, N, D)

seq, attn = model(x,
                return_attention=True,   # training default False; set True for visualization
                last_only=True,           # return only the last layer's attention
                topk=None,          # return only top-k per row (None=return full)
                cpu_offload=True)


import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "model_lib"))
from model_summary import model_summary
model_summary(model, input_size=(1, 20000, 1536))

'''


'''
# ---------------- positional encoding ----------------
def posemb_sincos_1d(n, dim, temperature: int = 10000, dtype=torch.float32):
    assert dim % 2 == 0
    pos = torch.arange(n)[:, None]
    i   = torch.arange(dim // 2)[None, :]
    omega = 1.0 / (temperature ** (i / (dim // 2 - 1)))
    angles = pos * omega
    pe = torch.cat([angles.sin(), angles.cos()], dim=1)
    return pe.type(dtype)

# ---------------- low-rank QKV + MQA ----------------
class LowRankQKV(nn.Module):
    def __init__(self, dim, heads, dim_head, rank=128, bias=False):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        self.down = nn.Linear(dim, rank, bias=bias)                 # dim -> r
        self.Bq   = nn.Linear(rank, heads * dim_head, bias=bias)    # r -> h*dh
        self.Bk   = nn.Linear(rank, dim_head, bias=bias)            # r -> dh
        self.Bv   = nn.Linear(rank, dim_head, bias=bias)            # r -> dh

    def forward(self, x):
        z = self.down(x)                     # [B,N,r]
        q = self.Bq(z)                       # [B,N,h*dh]
        k = self.Bk(z)                       # [B,N,dh]
        v = self.Bv(z)                       # [B,N,dh]
        return q, k, v

class MultiHeadAttention_MQA_LR(nn.Module):
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

    def forward(self, x):
        B, N, _ = x.shape
        h, dh = self.heads, self.dim_head

        q, k, v = self.qkv(x)                        # q:[B,N,h*dh], k/v:[B,N,dh]
        q = q.view(B, N, h, dh).transpose(1, 2)      # [B,h,N,dh]
        k = k.unsqueeze(1).expand(B, h, N, dh)       # shared K
        v = v.unsqueeze(1).expand(B, h, N, dh)       # shared V

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,h,N,N]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, h*dh)  # [B,N,h*dh]
        if self.use_lowrank_out:
            out = self.out_B(self.out_A(out))
        else:
            out = self.to_out(out)
        return out, attn

# ---------------- GEGLU-FFN (scalable width) ----------------
class FeedForward_GEGLU(nn.Module):
    def __init__(self, dim, ff_mult=0.75, dropout=0.1):
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

# ---------------- pre-norm wrapper ----------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))

# ---------------- lightweight Transformer encoder ----------------
class TransformerLite(nn.Module):
    """
    Equivalent interface to the original Transformer(dim, depth, heads, dim_head, mlp_dim);
    extra params: rank, ff_mult, dropout, share_block
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=None,
                 rank=128, ff_mult=0.75, dropout=0.1, share_block=False):
        super().__init__()
        self.depth = depth
        self.share_block = share_block

        def make_block():
            attn = MultiHeadAttention_MQA_LR(dim, heads, dim_head, rank=rank, dropout=dropout, lowrank_out=True)
            ffn  = FeedForward_GEGLU(dim, ff_mult=ff_mult, dropout=dropout)
            return nn.ModuleList([PreNorm(dim, attn), PreNorm(dim, ffn)])

        if share_block:
            self.block = make_block()
        else:
            self.blocks = nn.ModuleList([make_block() for _ in range(depth)])

    def forward(self, x, return_attention=False):
        attns = []

        def run_block(x, block):
            attn_mod, ffn_mod = block
            x_attn, attn_w = attn_mod(x)    # attn_w: [B,h,N,N]
            x = x + x_attn
            x = x + ffn_mod(x)
            return x, attn_w

        if self.share_block:
            for _ in range(self.depth):
                x, a = run_block(x, self.block)
                if return_attention:
                    attns.append(a.unsqueeze(1))
        else:
            for blk in self.blocks:
                x, a = run_block(x, blk)
                if return_attention:
                    attns.append(a.unsqueeze(1))

        attn_all = torch.cat(attns, dim=1) if return_attention and attns else None  # [B,depth,h,N,N]
        return x, attn_all

# ---------------- TokenViT (only replaces the internal Transformer) ----------------
class TokenViT(nn.Module):
    def __init__(self, in_dim=1536, num_tokens=20000, dim=512, depth=2, heads=4,
                 dim_head=64, mlp_dim=1024, num_classes=3, pool_stride=1):
        super().__init__()
        assert pool_stride == 1, "pool_stride can only be 1"
        self.num_tokens = num_tokens

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim, bias=False),
            nn.LayerNorm(dim),
        )
        self.register_buffer("pos_emb_1d", posemb_sincos_1d(num_tokens, dim), persistent=False)

        # replace the original Transformer with the lightweight version; interface and calls stay the same
        self.transformer = TransformerLite(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
            rank=128,        # low-rank bottleneck
            ff_mult=0.75,    # shrink FFN width
            dropout=0.1,
            share_block=False # set True for more aggressive parameter reduction
        )
        self.to_latent = nn.Identity()
        self.head = nn.Linear(dim, num_classes, bias=False)
        self.multi_score = None

    def forward(self, x):         # x: [B,N,in_dim]
        B, N, _ = x.shape
        assert N == self.num_tokens
        x = self.input_proj(x)    # [B,N,dim]
        x = x + self.pos_emb_1d[:N].to(x.device, dtype=x.dtype)

        x, self.multi_score = self.transformer(x, return_attention=True)   # [B,N,dim], [B,depth,h,N,N]
        x = self.to_latent(x)
        return x, self.multi_score
'''

'''
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "model_lib"))
import torch
from TokenViT_2 import TokenViT


B, N, D = 1, 20000, 1536                 # default input shape (large memory usage, B=1)

model = TokenViT(in_dim=D, num_tokens=N,
                     dim=512, depth=2, heads=2, dim_head=256, mlp_dim=1024,use_geglu =False,
                     num_classes=3, pool_stride=1)
# if memory is tight, use half precision: x = torch.randn(B,N,D, dtype=torch.float16).to(torch.float32)
x = torch.randn(B, N, D)

seq, attn = model(x)


import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "model_lib"))
from model_summary import model_summary
model_summary(model, input_size=(1, 20000, 1536))

'''