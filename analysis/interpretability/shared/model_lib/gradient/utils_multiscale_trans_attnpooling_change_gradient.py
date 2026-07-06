# utils_multiscale_trans_attnpooling_change_gradient.py
# Based on utils_multiscale_trans_attnpooling_change.py, with gradient tracking support
# Main change: attention weights keep gradients for Attention x Gradient analysis

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import sys
from math import ceil

# -------------------------
# external modules
# -------------------------
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "shared", "model_lib", "MultiScale"))
from COAM_Fusion import CSMIL_PyramidProgressive
from CSMIL_Fusion import CSMILStyleMultiscaleFusion, CSMIL_MoE_Multiscale
from FC_AttnPool_Fusion import FC_AttnPool_Fusion

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "shared", "model_lib"))
from CrossModalMultiHeadAttention import MultiLayerCrossModalAttention
from utils_trans_cross_fusion import AttnPool1D
from ClassificationHead_layernorm import ClassificationHead_layernorm
from regression_head import regression_head
from LongTokenSeqEncoder import PerceiverIOSequenceEncoder

# use the local gradient version of TokenViT
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from TokenViT_2_gradient import TokenViT


def split_tensor(
    x: torch.Tensor,
    split_dims: int = 1536,
    dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split the 4D tensor x into two parts along dim at the split_dims position.
    """
    assert x.ndim == 4, f"Expected a 4D tensor, got {x.ndim}D, shape={tuple(x.shape)}"
    C = x.size(dim)
    if not (1 <= split_dims < C):
        raise ValueError(f"split_dims must be in 1..{C-1}, current split_dims={split_dims}, C={C}")

    left, right = torch.split(x, [split_dims, C - split_dims], dim=dim)
    return left.contiguous(), right[..., 0, 0].contiguous()


class utils_multiScale_model_trans(nn.Module):
    """
    Backbone: fusion of the main modality (pathology multi-scale features) and clinical/auxiliary modalities
    Design changes (gradient version):
      1) clinical goes from "41 scalar tokens" -> "1 clinical token"
      2) add a slide token as a global aggregation channel
      3) after cross, add 1 joint self-attn layer
      4) [new] attention weights keep gradients, supporting AxG analysis
    """
    def __init__(
        self,
        split_dims: int = 1536,
        clin_dim: int = 41,
        Cross_num_layers: int = 2,
        Cross_embed_dim: int = 256,
        Cross_num_heads: int = 2,
        Pooling=AttnPool1D,
        Classify_dim: int = 1,
        Fusion_PyramidProgressive: bool = True,
        trans_perciever: bool = True,
        regression: bool = False,
        use_geglu: bool = True,
        use_MQA: bool = True,
        enc_return_attention: bool = True,
        enc_last_only: bool = True,
        enc_topk=None,
        enc_cpu_offload: bool = True,
        joint_heads: int = 4,
        for_gradient: bool = False  # [new] gradient-mode switch
    ):
        super().__init__()

        self.for_gradient = for_gradient  # store the gradient-mode flag

        # multi-scale fusion
        if Fusion_PyramidProgressive:
            self.Scale_Fusion = FC_AttnPool_Fusion(dim=1536)
        else:
            self.Scale_Fusion = CSMIL_MoE_Multiscale(num_experts=4)

        # clinical: clin_dim -> 1 token (B,1,256)
        self.clin_dim = clin_dim
        self.linear_C = nn.Sequential(
            nn.LayerNorm(self.clin_dim),
            nn.Linear(self.clin_dim, Cross_embed_dim),
            nn.GELU(),
            nn.LayerNorm(Cross_embed_dim)
        )

        # pathology: 1536 -> 256
        self.linear_H = nn.Linear(1536, Cross_embed_dim)

        # cross-modal attention block
        self.CrossBlock = MultiLayerCrossModalAttention(
            num_layers=Cross_num_layers,
            embed_dim=Cross_embed_dim,
            num_heads=Cross_num_heads
        )

        self.pool = Pooling(Cross_embed_dim, hidden=Cross_embed_dim * 2)
        self.split_dims = split_dims
        self.trans_perciever = trans_perciever

        # pathology sequence encoder
        if trans_perciever:
            self.trans = PerceiverIOSequenceEncoder(
                in_dim=Cross_embed_dim,
                dim=Cross_embed_dim,
                depth=3,
                heads=4,
                num_latents=int(ceil(Cross_embed_dim / 4)),
                dropout=0.2,
                ffn_mult=4.0,
                add_pos_emb=True,
                track_attn=True
            )
            self.enc_return_attention = True
            self.enc_last_only = True
            self.enc_topk = None
            self.enc_cpu_offload = True
        else:
            self.trans = TokenViT(
                in_dim=Cross_embed_dim,
                num_tokens=1500,
                dim=Cross_embed_dim,
                depth=1,
                heads=4,
                dim_head=int(ceil(Cross_embed_dim / 4)),
                mlp_dim=Cross_embed_dim * 2,
                use_geglu=use_geglu,
                use_MQA=use_MQA
            )
            self.enc_return_attention = enc_return_attention
            self.enc_last_only = enc_last_only
            self.enc_topk = enc_topk
            self.enc_cpu_offload = enc_cpu_offload

        # slide token (global aggregation channel)
        self.slide_token = nn.Parameter(torch.randn(1, 1, Cross_embed_dim) * 0.02)

        # joint self-attn
        if Cross_embed_dim % joint_heads != 0:
            raise ValueError(f"Cross_embed_dim={Cross_embed_dim} must be divisible by joint_heads={joint_heads}")

        joint_layer = nn.TransformerEncoderLayer(
            d_model=Cross_embed_dim,
            nhead=joint_heads,
            dim_feedforward=Cross_embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        self.joint_encoder = nn.TransformerEncoder(joint_layer, num_layers=1)
        self.joint_norm = nn.LayerNorm(Cross_embed_dim)

        # head
        if regression:
            self.head = regression_head(input_dim=Cross_embed_dim, output_dim=Classify_dim)
        else:
            self.head = ClassificationHead_layernorm(input_dim=Cross_embed_dim, output_dim=Classify_dim)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 1536+clin_dim, M, N)
        """
        feat1536, clin = split_tensor(x, split_dims=self.split_dims, dim=1)

        # 1) clinical token
        if clin.dim() != 2 or clin.size(1) != self.clin_dim:
            raise ValueError(f"clinical dimension mismatch: clin.shape={tuple(clin.shape)}, expected (B,{self.clin_dim})")
        clin_tok = self.linear_C(clin).unsqueeze(1)

        # 2) multi-scale fusion
        feat_fused = self.Scale_Fusion(feat1536).squeeze(-1)
        feat_proj = self.linear_H(feat_fused.transpose(1, 2))

        # 3) pathology sequence encoding
        if self.trans_perciever:
            feat_enc, enc_attn = self.trans(feat_proj)
        else:
            feat_enc, enc_attn = self.trans(
                feat_proj,
                return_attention=self.enc_return_attention,
                last_only=self.enc_last_only,
                topk=self.enc_topk,
                cpu_offload=self.enc_cpu_offload,
                for_gradient=self.for_gradient  # [pass through] gradient mode
            )

        # 4) cross-attn: query = [slide, clin]
        B = feat_enc.size(0)
        slide = self.slide_token.expand(B, -1, -1)
        query = torch.cat([slide, clin_tok], dim=1)

        cross_out, attn_weights = self.CrossBlock(
            query=query,
            key=feat_enc,
            value=feat_enc
        )

        # [gradient mode] keep the gradient of cross_attn
        if self.for_gradient and isinstance(attn_weights, list):
            # attn_weights is a list; take the last layer and keep its gradient
            attn_weights_grad = attn_weights[-1]
            if not attn_weights_grad.requires_grad:
                attn_weights_grad = attn_weights_grad.detach().requires_grad_(True)
            self.attn_weights = attn_weights  # keep the full list
        elif self.for_gradient:
            if not attn_weights.requires_grad:
                attn_weights = attn_weights.detach().requires_grad_(True)
            self.attn_weights = attn_weights
        else:
            self.attn_weights = attn_weights

        # [gradient mode] keep the gradient of enc_attn
        if self.for_gradient and enc_attn is not None:
            if not enc_attn.requires_grad:
                enc_attn = enc_attn.detach().requires_grad_(True)
        self.enc_attn = enc_attn

        # 5) joint self-attn
        joint_out = self.joint_encoder(cross_out)
        slide_emb = self.joint_norm(joint_out[:, 0])

        # 6) head
        output = self.head(slide_emb).to(dtype=torch.float32)

        # record attention
        self.multi_score = {
            "enc_attn": enc_attn,
            "cross_subtype": attn_weights,
        }

        return output
