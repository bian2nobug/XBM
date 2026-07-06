import torch
import torch.nn as nn
import torch.nn.functional as F

class FC_AttnPool_Fusion(nn.Module):
    """
    普通可学习融合基线：对 N=21 视图做一次 attention pooling
    - 输入:  x (B,1536,M,21)
    - 输出:  y (B,1536,M,1)

    思路：
      对每个 token(1536-d) 用一个全连接 1536->1 打分，
      在 N 维做 softmax 得权重，再做加权和压成 1 个视图。
    """
    def __init__(self, dim=1536):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.size(1) == 1536 and x.size(-1) == 21, \
            "期望输入形状 (B,1536,M,21)"

        # (B,1536,M,21) -> (B,M,21,1536)
        t = x.permute(0, 2, 3, 1).contiguous()

        # 打分: (B,M,21,1)
        a = self.scorer(t)

        # 权重: (B,M,21,1)
        w = F.softmax(a, dim=2)

        # 加权融合: (B,M,1536)
        fused = (w * t).sum(dim=2)

        # -> (B,1536,M,1)
        return fused.permute(0, 2, 1).unsqueeze(-1).contiguous()


'''
import torch
import sys
sys.path.append('/WorkSpace/liudongbo/General_Model_Architecture/model/MultiScale')

from FC_AttnPool_Fusion import FC_AttnPool_Fusion
# 1) 初始化
fusion = FC_AttnPool_Fusion(dim=1536)

# 2) 构造一个示例输入：x (B,1536,M,21)
B, M = 2, 512
x = torch.randn(B, 1536, M, 21)

# 3) 调用
y = fusion(x)

print("x:", x.shape)  # (2, 1536, 512, 21)
print("y:", y.shape)  # (2, 1536, 512, 1)

'''