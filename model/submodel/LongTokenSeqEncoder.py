# LongTokenSeqEncoder.py
import torch
from torch import nn

def posemb_sincos_1d(n, dim, temperature: int = 10000, dtype=torch.float32, device=None):
    assert dim % 2 == 0
    pos = torch.arange(n, device=device, dtype=dtype)[:, None]
    i   = torch.arange(dim // 2, device=device, dtype=dtype)[None, :]
    omega = 1.0 / (temperature ** (i / (dim // 2 - 1)))
    angles = pos * omega
    pe = torch.cat([angles.sin(), angles.cos()], dim=1)  # [n, dim]
    return pe

class FFN(nn.Module):
    def __init__(self, dim, mult=4.0, drop=0.1):
        super().__init__()
        hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
    def forward(self, x): return self.net(x)

class PerceiverIOSequenceEncoder(nn.Module):
    """
    输入:  x (B, N, in_dim)  —— 例如 (B, 20000, 1536)
    输出:  y (B, N, dim)     —— 例如 (B, 20000, 256)

    每层结构：
      1) latents <- tokens  (cross-attn)
      2) latents <-> latents (self-attn + FFN)
      3) tokens  <- latents  (cross-attn)  # 回写到逐 token 表征
      4) tokens  FFN
    """
    def __init__(
        self,
        in_dim: int,
        dim: int = 256,
        depth: int = 2,
        heads: int = 8,
        num_latents: int = 128,
        dropout: float = 0.1,
        ffn_mult: float = 4.0,
        add_pos_emb: bool = True,
        track_attn: bool = True,   # 若需可视化，可置 True 存最后一层注意力
    ):
        super().__init__()
        self.dim = dim
        self.add_pos = add_pos_emb
        self.track_attn = track_attn

        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.LayerNorm(dim),
        )

        # 可学习 latent queries: (1, L, D)
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)

        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict(dict(
                # latents <- tokens
                ln_lat_x1 = nn.LayerNorm(dim),
                ln_tok_x1 = nn.LayerNorm(dim),
                cross1    = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                drop1     = nn.Dropout(dropout),

                # latent self
                ln_lat_sa = nn.LayerNorm(dim),
                self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                drop_sa   = nn.Dropout(dropout),
                ffn_lat   = FFN(dim, mult=ffn_mult, drop=dropout),

                # tokens <- latents
                ln_tok_x2 = nn.LayerNorm(dim),
                ln_lat_x2 = nn.LayerNorm(dim),
                cross2    = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                drop2     = nn.Dropout(dropout),
                ffn_tok   = FFN(dim, mult=ffn_mult, drop=dropout),
            )))

        self.out_norm = nn.LayerNorm(dim)

        # 可选：暴露最后一次 tokens<-latents 的 attn 权重 (B, heads, N, L)
        self.last_attn_tokens_from_latents = None

    def forward(self, x, key_padding_mask=None):  # x: (B, N, in_dim)
        
        B, N, _ = x.shape
        x = self.in_proj(x)  # -> (B, N, D)

        if self.add_pos:
            pos = posemb_sincos_1d(N, self.dim, dtype=x.dtype, device=x.device)
            x = x + pos.unsqueeze(0)

        lat = self.latents.expand(B, -1, -1).contiguous()  # (B, L, D)

        attn_keep = self.track_attn
        self.last_attn_tokens_from_latents = None

        for blk in self.blocks:
            # 1) latents <- tokens (cross-attn)
            q = blk.ln_lat_x1(lat)
            kv = blk.ln_tok_x1(x)
            lat_upd, _ = blk.cross1(q, kv, kv,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=False)
            lat = lat + blk.drop1(lat_upd)

            # 2) latent self-attn + FFN
            ql = blk.ln_lat_sa(lat)
            sa_out, _ = blk.self_attn(ql, ql, ql, need_weights=False)
            lat = lat + blk.drop_sa(sa_out)
            lat = lat + blk.ffn_lat(lat)

            # 3) tokens <- latents (cross-attn)   (保持 N 不变)
            qt = blk.ln_tok_x2(x)
            kl = blk.ln_lat_x2(lat)
            tok_upd, attn = blk.cross2(qt, kl, kl,
                                       need_weights=attn_keep,
                                       average_attn_weights=False)
            x = x + blk.drop2(tok_upd)

            self.last_attn_tokens_from_latents = attn


            # 4) token FFN
            x = x + blk.ffn_tok(x)

        return self.out_norm(x),self.last_attn_tokens_from_latents  # (B, N, D)
