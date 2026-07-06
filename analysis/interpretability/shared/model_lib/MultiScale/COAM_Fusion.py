import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== Basic block: MHSA + FFN (PreNorm, ROAM-style) ==========
class MHSA(nn.Module):
    def __init__(self, dim=1536, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads                              # number of heads
        self.scale = dim_head ** -0.5                   # scaling
        inner = heads * dim_head                        # inner dim
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(inner, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm_head = nn.LayerNorm(dim_head)         # per-head normalization

    def forward(self, x):                               # x:(B,T,C)
        B, T, C = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        h = self.heads
        q = q.view(B, T, h, -1).transpose(1, 2)         # (B,h,T,d)
        k = k.view(B, T, h, -1).transpose(1, 2)
        v = v.view(B, T, h, -1).transpose(1, 2)
        q = self.norm_head(q); k = self.norm_head(k)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,h,T,T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)                     # (B,h,T,d)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B,T,C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class FFN(nn.Module):
    def __init__(self, dim=1536, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid, dim), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, dim=1536, heads=8, dim_head=64, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MHSA(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = FFN(dim, mlp_ratio, dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ========== Intra-scale SA: output CLS and "tokens with CLS removed" (for the next cross-scale step) ==========
class IntraScaleSA_KeepTokens(nn.Module):
    """
    Input:  x (B,C,M,N)
    Output: cls:(B,C,M) , tokens:(B,C,M,N)  # tokens are the N tokens with CLS removed (shape unchanged)
    """
    def __init__(self, dim=1536, N=16, depth=1, heads=8, dim_head=64, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.N = N
        self.cls = nn.Parameter(torch.randn(1, 1, dim))     # CLS
        self.pos = nn.Parameter(torch.zeros(1, N + 1, dim)) # positional encoding
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, heads, dim_head, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):                                   # (B,C,M,N)
        B, C, M, N = x.shape
        assert C == self.dim and N == self.N
        x = x.permute(0, 2, 3, 1).contiguous()              # (B,M,N,C)
        x = x.view(B * M, N, C)                             # (BM,N,C)
        cls = self.cls.expand(B * M, 1, C)                  # (BM,1,C)
        seq = torch.cat([cls, x], dim=1) + self.pos         # (BM,N+1,C)
        for blk in self.blocks:
            seq = blk(seq)                                  # (BM,N+1,C)
        cls_out = seq[:, 0]                                 # (BM,C)
        tok_out = seq[:, 1:]                                # (BM,N,C)
        cls_out = cls_out.view(B, M, C).permute(0, 2, 1)    # (B,C,M)
        tok_out = tok_out.view(B, M, N, C).permute(0, 3, 1, 2).contiguous()  # (B,C,M,N)
        return cls_out, tok_out

# ========== Cross-scale SA: update low-scale tokens using grouped high-scale tokens ==========
class InterScaleUpdate(nn.Module):
    """
    Input:  low:(B,C,M,Nl) , high:(B,C,M,Nh) , Nh must be divisible by Nl
    Process: for each low-scale token, concatenate it with its corresponding g=Nh/Nl high-scale tokens, run SA, take position 0 as the updated low-scale token
    Output: low_new:(B,C,M,Nl)
    """
    def __init__(self, dim=1536, depth=1, heads=8, dim_head=64, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, heads, dim_head, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, low, high):                           # low:(B,C,M,Nl) high:(B,C,M,Nh)
        B, C, M, Nl = low.shape
        Nh = high.size(-1)
        assert Nh % Nl == 0, "Nh must be a multiple of Nl"
        g = Nh // Nl                                        # number of high tokens per low token
        # arrange into sequence [low_i, high_i^g]
        lt = low.permute(0, 2, 3, 1).contiguous()           # (B,M,Nl,C)
        ht = high.permute(0, 2, 3, 1).contiguous()          # (B,M,Nh,C)
        ht = ht.view(B, M, Nl, g, C)                        # (B,M,Nl,g,C)
        seq = torch.cat([lt.unsqueeze(3), ht], dim=3)       # (B,M,Nl,1+g,C)
        seq = seq.view(B * M * Nl, 1 + g, C)                # (BMNl,1+g,C)
        for blk in self.blocks:
            seq = blk(seq)                                  # (BMNl,1+g,C)
        upd = seq[:, 0]                                     # (BMNl,C)
        upd = upd.view(B, M, Nl, C).permute(0, 3, 1, 2).contiguous()  # (B,C,M,Nl)
        return upd

# ========== Progressive pyramid fusion ==========
class CSMIL_PyramidProgressive(nn.Module):
    """
    Input:  x (B,1536,M,21)  where 21=1(5x)+4(10x)+16(20x)
    Output: y (B,1536,M,1)
    """
    def __init__(
        self, dim=1536,
        depth_intra=1, heads_intra=8, dim_head_intra=64, mlp_ratio_intra=4.0, dropout_intra=0.0,
        depth_inter_20_10=1, depth_inter_10_5=1, heads_inter=8, dim_head_inter=64, mlp_ratio_inter=4.0, dropout_inter=0.0,
        learnable_fuse=True
    ):
        super().__init__()
        # intra-scale
        self.intra20 = IntraScaleSA_KeepTokens(dim, N=16, depth=depth_intra, heads=heads_intra, dim_head=dim_head_intra, mlp_ratio=mlp_ratio_intra, dropout=dropout_intra)
        self.intra10 = IntraScaleSA_KeepTokens(dim, N=4,  depth=depth_intra, heads=heads_intra, dim_head=dim_head_intra, mlp_ratio=mlp_ratio_intra, dropout=dropout_intra)
        self.intra5  = IntraScaleSA_KeepTokens(dim, N=1,  depth=depth_intra, heads=heads_intra, dim_head=dim_head_intra, mlp_ratio=mlp_ratio_intra, dropout=dropout_intra)
        # cross-scale
        self.inter_20_10 = InterScaleUpdate(dim, depth=depth_inter_20_10, heads=heads_inter, dim_head=dim_head_inter, mlp_ratio=mlp_ratio_inter, dropout=dropout_inter)
        self.inter_10_5  = InterScaleUpdate(dim, depth=depth_inter_10_5,  heads=heads_inter, dim_head=dim_head_inter, mlp_ratio=mlp_ratio_inter, dropout=dropout_inter)
        # fusion weights
        self.learnable_fuse = learnable_fuse
        if learnable_fuse:
            self.w = nn.Parameter(torch.randn(3, 1))        # three-scale weights
        else:
            self.register_buffer("w_fixed", torch.tensor([[1.0],[1.0],[1.0]]))  # fixed equal weights

    def forward(self, x):                                    # x:(B,1536,M,21)
        assert x.size(1) == 1536 and x.size(-1) == 21
        B, C, M, _ = x.shape
        f5, f10, f20 = x[..., 0:1], x[..., 1:5], x[..., 5:21]    # split by 5/10/20x

        # (1) 20x intra-scale SA (get cls20 and tokens20)
        cls20, tok20 = self.intra20(f20)                     # cls20:(B,C,M) tok20:(B,C,M,16)

        # (2) 20->10 cross-scale update of 10x tokens
        tok10_init = f10                                     # initial 10x tokens
        tok10_upd  = self.inter_20_10(tok10_init, tok20)     # (B,C,M,4)

        # (3) 10x intra-scale SA (get cls10 and tokens10)
        cls10, tok10 = self.intra10(tok10_upd)               # cls10:(B,C,M) tok10:(B,C,M,4)

        # (4) 10->5 cross-scale update of 5x tokens
        tok5_init = f5                                       # initial 5x token
        tok5_upd  = self.inter_10_5(tok5_init, tok10)        # (B,C,M,1)

        # (5) 5x intra-scale SA (get cls5)
        cls5, _ = self.intra5(tok5_upd)                      # cls5:(B,C,M)

        # (6) fuse the three-scale CLS into the ROI representation
        if self.learnable_fuse:
            w = torch.softmax(self.w, dim=0)                 # (3,1)
        else:
            w = self.w_fixed / self.w_fixed.sum()
        y = w[0] * cls5 + w[1] * cls10 + w[2] * cls20        # (B,C,M)
        return y.unsqueeze(-1)                                # (B,1536,M,1)

# ----------------- simple self-test -----------------
if __name__ == "__main__":
    B, F, M = 2, 1536, 6
    x = torch.randn(B, F, M, 21)
    net = CSMIL_PyramidProgressive(
        depth_intra=1, heads_intra=8, dim_head_intra=64, dropout_intra=0.1,
        depth_inter_20_10=1, depth_inter_10_5=1, heads_inter=4, dim_head_inter=64, dropout_inter=0.1,
        learnable_fuse=True
    )
    y = net(x)
    print(y.shape)  # (B,1536,M,1)


'''
import torch
import sys
sys.path.append('')
from COAM_Fusion import CSMIL_PyramidProgressive


B, F, M = 2, 1536, 6
x = torch.randn(B, F, M, 21)
net = CSMIL_PyramidProgressive(
        depth_intra=1, heads_intra=8, dim_head_intra=64, dropout_intra=0.1,
        depth_inter_20_10=1, depth_inter_10_5=1, heads_inter=4, dim_head_inter=64, dropout_inter=0.1,
        learnable_fuse=True
    )
y = net(x)
print(y.shape)  # (B,1536,M,1)

'''