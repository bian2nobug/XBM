import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- helper: move 1536 to the last dim for linear, then move back -----------------
def to_last1536(x):      # (B,1536,M,N) -> (B,M,N,1536)
    return x.permute(0, 2, 3, 1).contiguous()

def back_from_last1536(x):  # (B,M,N,1536) -> (B,1536,M,N)
    return x.permute(0, 3, 1, 2).contiguous()

# ----------------- intra-scale aggregation: attention-weighted average over N in {1,4,16} -----------------
class IntraScaleAggregator(nn.Module):
    """
    Input:  (B,1536,M,N)
    Process: score each token (1536-d) with 1536->1, softmax over N to get weights, then weighted sum
    Output: (B,1536,M)   # N is aggregated into a single 1536 vector
    """
    def __init__(self):
        super().__init__()
        self.scorer = nn.Linear(1536, 1)  # linear only on the 1536 dim

    def forward(self, x):
        t = to_last1536(x)                 # (B,M,N,1536)
        a = self.scorer(t)                 # (B,M,N,1)
        w = F.softmax(a, dim=2)            # normalize over N
        fused = (w * t).sum(dim=2)         # (B,M,1536)
        return fused.permute(0, 2, 1).contiguous()  # (B,1536,M)

# ----------------- inter-scale fusion: score z5/z10/z20 with 1536->1, then weighted sum -----------------
class InterScaleFusion(nn.Module):
    """
    Input:  three per-scale aggregated tensors (B,1536,M)
    Output: (B,1536,M)
    """
    def __init__(self):
        super().__init__()
        self.scorer = nn.Linear(1536, 1)  # linear only on the 1536 dim

    def forward(self, z_list):  # len=3: [z5,z10,z20]
        # per-scale (B,M,1) scores
        scores = []
        for z in z_list:
            t = z.permute(0, 2, 1)        # (B,M,1536)
            s = self.scorer(t)            # (B,M,1)
            scores.append(s)
        S = torch.cat(scores, dim=2)      # (B,M,3)
        W = F.softmax(S, dim=2)           # (B,M,3)

        # weighted sum (reshape (B,M,1) to (B,1,M) to broadcast with (B,1536,M))
        z = 0
        for i, zi in enumerate(z_list):
            wi = W[:, :, i:i+1].permute(0, 2, 1)  # (B,1,M)
            z = z + zi * wi                       # (B,1536,M)
        return z

# ----------------- full model: CS-MIL-style multi-scale fusion -----------------
class CSMILStyleMultiscaleFusion(nn.Module):
    """
    Input:  x (B,1536,M,21)   where 21=1+4+16
    Output: y (B,1536,M,1)
    """
    def __init__(self):
        super().__init__()
        self.agg5  = IntraScaleAggregator()
        self.agg10 = IntraScaleAggregator()
        self.agg20 = IntraScaleAggregator()
        self.inter = InterScaleFusion()

    def forward(self, x):
        assert x.size(1) == 1536 and x.size(-1) == 21, "expected input shape (B,1536,M,21)"
        f5  = x[..., 0:1]    # (B,1536,M,1)
        f10 = x[..., 1:5]    # (B,1536,M,4)
        f20 = x[..., 5:21]   # (B,1536,M,16)

        z5  = self.agg5(f5)      # (B,1536,M)
        z10 = self.agg10(f10)    # (B,1536,M)
        z20 = self.agg20(f20)    # (B,1536,M)

        z   = self.inter([z5, z10, z20])  # (B,1536,M)
        return z.unsqueeze(-1)            # (B,1536,M,1)

# =========================================================
# 1) Mixture of experts: CSMIL + MoE (all linears on the 1536 dim)
# =========================================================
class CSMIL_MoE_Multiscale(nn.Module):
    """
    Input:  x (B,1536,M,21)  where 21=1+4+16
    Output: y (B,1536,M,1)
    Flow:   intra-scale aggregation -> inter-scale fusion -> MoE (1536-dim expert transform) -> output
    """
    def __init__(self, num_experts=4, hidden=1536, dropout=0.1):
        super().__init__()
        self.agg5  = IntraScaleAggregator()
        self.agg10 = IntraScaleAggregator()
        self.agg20 = IntraScaleAggregator()
        self.inter = InterScaleFusion()
        # experts: linear only on the 1536 dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1536, hidden), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1536)
            ) for _ in range(num_experts)
        ])
        # gating: per-position expert weights from the 1536-dim feature
        gate_hidden = max(32, hidden // 2)
        self.gate = nn.Sequential(
            nn.Linear(1536, gate_hidden), nn.GELU(),
            nn.Linear(gate_hidden, num_experts)
        )

    def forward(self, x):
        assert x.size(1) == 1536 and x.size(-1) == 21
        f5, f10, f20 = x[...,0:1], x[...,1:5], x[...,5:21]
        z5  = self.agg5(f5)                     # (B,1536,M)
        z10 = self.agg10(f10)                   # (B,1536,M)
        z20 = self.agg20(f20)                   # (B,1536,M)
        zf  = self.inter([z5, z10, z20])        # (B,1536,M)

        # gating on 1536
        g = self.gate(zf.permute(0,2,1))        # (B,M,K)
        g = F.softmax(g, dim=-1)

        # expert transform and mix by expert weights
        y = 0
        for e, expert in enumerate(self.experts):
            ye = expert(zf.permute(0,2,1))      # (B,M,1536)
            ye = ye.permute(0,2,1)              # (B,1536,M)
            ge = g[:,:,e:e+1].permute(0,2,1)    # (B,1,M)
            y  = y + ye * ge                    # (B,1536,M)

        return y.unsqueeze(-1)                  # (B,1536,M,1)
    
# ----------------- simple self-test -----------------
'''
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "shared", "model_lib", "MultiScale"))
from CSMIL import CSMILStyleMultiscaleFusion,CSMIL_MoE_Multiscale

B, F, M = 2, 1536, 32
x = torch.randn(B, F, M, 21)
model = CSMILStyleMultiscaleFusion()
y = model(x)
print(y.shape)   # expected: (B,1536,M,1)



moe = CSMIL_MoE_Multiscale(num_experts=4)
y_moe = moe(x)
print("MoE:", y_moe.shape)  # (B,1536,M,1)

'''

