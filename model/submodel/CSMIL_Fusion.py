import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- 小工具：把 1536 移到最后做线性，再移回 -----------------
def to_last1536(x):      # (B,1536,M,N) -> (B,M,N,1536)
    return x.permute(0, 2, 3, 1).contiguous()

def back_from_last1536(x):  # (B,M,N,1536) -> (B,1536,M,N)
    return x.permute(0, 3, 1, 2).contiguous()

# ----------------- 尺度内聚合：沿 N∈{1,4,16} 做注意力加权平均 -----------------
class IntraScaleAggregator(nn.Module):
    """
    输入:  (B,1536,M,N)
    过程:  对每个 token(1536-d) 用 1536->1 打分，softmax 沿 N 得权重，做加权和
    输出:  (B,1536,M)   # 已把 N 聚合成 1 个 1536 向量
    """
    def __init__(self):
        super().__init__()
        self.scorer = nn.Linear(1536, 1)  # 只在1536维上线性

    def forward(self, x):
        t = to_last1536(x)                 # (B,M,N,1536)
        a = self.scorer(t)                 # (B,M,N,1)
        w = F.softmax(a, dim=2)            # 沿 N 归一化
        fused = (w * t).sum(dim=2)         # (B,M,1536)
        return fused.permute(0, 2, 1).contiguous()  # (B,1536,M)

# ----------------- 尺度间融合：对 z5/z10/z20 用 1536->1 求权重后加权和 -----------------
class InterScaleFusion(nn.Module):
    """
    输入:  三个尺度聚合后的 (B,1536,M)
    输出:  (B,1536,M)
    """
    def __init__(self):
        super().__init__()
        self.scorer = nn.Linear(1536, 1)  # 只在1536维上线性
    
    def forward(self, z_list):  # len=3: [z5,z10,z20]
        # 逐尺度得到 (B,M,1) 打分
        scores = []
        for z in z_list:
            t = z.permute(0, 2, 1)        # (B,M,1536)
            s = self.scorer(t)            # (B,M,1)
            scores.append(s)
        S = torch.cat(scores, dim=2)      # (B,M,3)
        W = F.softmax(S, dim=2)           # (B,M,3)

        # 加权和（把 (B,M,1) 变成 (B,1,M) 以便与 (B,1536,M) 广播相乘）
        z = 0
        for i, zi in enumerate(z_list):
            wi = W[:, :, i:i+1].permute(0, 2, 1)  # (B,1,M)
            z = z + zi * wi                       # (B,1536,M)
        return z

# ----------------- 总模型：CS-MIL 风格多尺度融合 -----------------
class CSMILStyleMultiscaleFusion(nn.Module):
    """
    输入:  x (B,1536,M,21)   其中 21=1+4+16
    输出:  y (B,1536,M,1)
    """
    def __init__(self):
        super().__init__()
        self.agg5  = IntraScaleAggregator()
        self.agg10 = IntraScaleAggregator()
        self.agg20 = IntraScaleAggregator()
        self.inter = InterScaleFusion()

    def forward(self, x):
        assert x.size(1) == 1536 and x.size(-1) == 21, "期望输入形状 (B,1536,M,21)"
        f5  = x[..., 0:1]    # (B,1536,M,1)
        f10 = x[..., 1:5]    # (B,1536,M,4)
        f20 = x[..., 5:21]   # (B,1536,M,16)

        z5  = self.agg5(f5)      # (B,1536,M)
        z10 = self.agg10(f10)    # (B,1536,M)
        z20 = self.agg20(f20)    # (B,1536,M)

        z   = self.inter([z5, z10, z20])  # (B,1536,M)
        return z.unsqueeze(-1)            # (B,1536,M,1)

# =========================================================
# 1) 专家混合：CSMIL + MoE（所有线性在1536维）
# =========================================================
class CSMIL_MoE_Multiscale(nn.Module):
    """
    输入:  x (B,1536,M,21)  其中 21=1+4+16
    输出:  y (B,1536,M,1)
    流程:  尺度内聚合 -> 尺度间融合 -> MoE(1536维专家变换) -> 输出
    """
    def __init__(self, num_experts=4, hidden=1536, dropout=0.1):
        super().__init__()
        self.agg5  = IntraScaleAggregator()
        self.agg10 = IntraScaleAggregator()
        self.agg20 = IntraScaleAggregator()
        self.inter = InterScaleFusion()
        # 专家们：只在1536维上线性
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1536, hidden), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1536)
            ) for _ in range(num_experts)
        ])
        # gating：基于1536维特征得到每位置的专家权重
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

        # 专家变换并按专家权重混合
        y = 0
        for e, expert in enumerate(self.experts):
            ye = expert(zf.permute(0,2,1))      # (B,M,1536)
            ye = ye.permute(0,2,1)              # (B,1536,M)
            ge = g[:,:,e:e+1].permute(0,2,1)    # (B,1,M)
            y  = y + ye * ge                    # (B,1536,M)

        return y.unsqueeze(-1)                  # (B,1536,M,1)
    
# ----------------- 简单自测 -----------------
'''
import torch
import sys
sys.path.append('/WorkSpace/liudongbo/General_Model_Architecture/model/MultiScale')
from CSMIL_Fusion import CSMILStyleMultiscaleFusion,CSMIL_MoE_Multiscale
device = 'cuda:2'
B, F, M = 10, 1536, 1200
x = torch.randn(B, F, M, 21).to(device)
model = CSMILStyleMultiscaleFusion().to(device)
y = model(x)
print(y.shape)   # 期望: (B,1536,M,1)



moe = CSMIL_MoE_Multiscale(num_experts=4)
y_moe = moe(x)
print("MoE:", y_moe.shape)  # (B,1536,M,1)

'''

