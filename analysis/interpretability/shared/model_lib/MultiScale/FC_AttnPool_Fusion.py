import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class FC_AttnPool_Fusion(nn.Module):
    """
    Simple learnable fusion baseline: perform one attention pooling over the N=21 views
    - Input:  x (B,1536,M,21)
    - Output: y (B,1536,M,1)

    Idea:
      score each token (1536-d) with a fully connected layer 1536->1,
      softmax over the N dimension to get weights, then weighted-sum into a single view.
    """
    def __init__(self, dim=1536):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.size(1) == 1536 and x.size(-1) == 21, \
            "expected input shape (B,1536,M,21)"

        # (B,1536,M,21) -> (B,M,21,1536)
        t = x.permute(0, 2, 3, 1).contiguous()

        # score: (B,M,21,1)
        a = self.scorer(t)

        # weights: (B,M,21,1)
        w = F.softmax(a, dim=2)

        # weighted fusion: (B,M,1536)
        fused = (w * t).sum(dim=2)

        # -> (B,1536,M,1)
        return fused.permute(0, 2, 1).unsqueeze(-1).contiguous()


'''
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "shared", "model_lib", "MultiScale"))

from FC_AttnPool_Fusion import FC_AttnPool_Fusion
# 1) initialize
fusion = FC_AttnPool_Fusion(dim=1536)

# 2) build an example input: x (B,1536,M,21)
B, M = 2, 512
x = torch.randn(B, 1536, M, 21)

# 3) call
y = fusion(x)

print("x:", x.shape)  # (2, 1536, 512, 21)
print("y:", y.shape)  # (2, 1536, 512, 1)

'''