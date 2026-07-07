from __future__ import annotations

from math import ceil
from typing import Tuple

import torch
import torch.nn as nn

try:
    from .submodel.FC_AttnPool_Fusion import FC_AttnPool_Fusion
    from .submodel.CSMIL_Fusion import CSMIL_MoE_Multiscale
    from .submodel.CrossModalMultiHeadAttention import MultiLayerCrossModalAttention
    from .submodel.utils_trans_cross_fusion import AttnPool1D
    from .submodel.ClassificationHead_layernorm import ClassificationHead_layernorm
    from .submodel.regression_head import regression_head
    from .submodel.LongTokenSeqEncoder import PerceiverIOSequenceEncoder
    from .submodel.TokenViT_2 import TokenViT
except ImportError:
    from submodel.FC_AttnPool_Fusion import FC_AttnPool_Fusion
    from submodel.CSMIL_Fusion import CSMIL_MoE_Multiscale
    from submodel.CrossModalMultiHeadAttention import MultiLayerCrossModalAttention
    from submodel.utils_trans_cross_fusion import AttnPool1D
    from submodel.ClassificationHead_layernorm import ClassificationHead_layernorm
    from submodel.regression_head import regression_head
    from submodel.LongTokenSeqEncoder import PerceiverIOSequenceEncoder
    from submodel.TokenViT_2 import TokenViT


def split_tensor(x: torch.Tensor, split_dims: int = 1536, dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a 4D tensor into pathology features and auxiliary features.

    Input shape is `(B, split_dims + aux_dim, M, N)`. The auxiliary channels are
    constant over the spatial/token dimensions and are read from `[..., 0, 0]`.
    `aux_dim=0` is supported for pathology-only XBM runs.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape={tuple(x.shape)}")
    C = x.size(dim)
    if not (1 <= split_dims <= C):
        raise ValueError(f"split_dims must be in [1, {C}], got split_dims={split_dims}, C={C}")
    left = x.narrow(dim, 0, split_dims).contiguous()
    aux_dim = C - split_dims
    if aux_dim == 0:
        return left, x.new_empty((x.size(0), 0))
    right = x.narrow(dim, split_dims, aux_dim)
    return left, right[..., 0, 0].contiguous()


class utils_multiScale_model_trans(nn.Module):
    """XBM model: same-FOV multiscale pathology encoding with optional clinical fusion.

    Expected input shape: `(B, split_dims + clin_dim, M, N)`. For the final
    WGD/WGD-TP53 configs, `split_dims=1536` and `clin_dim=57`.
    """

    def __init__(
        self,
        split_dims: int = 1536,
        clin_dim: int = 57,
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
        use_multiscale: bool = True,
        view_index: int = 0,
    ):
        super().__init__()
        self.split_dims = int(split_dims)
        self.clin_dim = int(clin_dim)
        self.trans_perciever = bool(trans_perciever)
        self.use_multiscale = bool(use_multiscale)
        self.view_index = int(view_index)

        if Fusion_PyramidProgressive:
            self.Scale_Fusion = FC_AttnPool_Fusion(dim=self.split_dims)
        else:
            self.Scale_Fusion = CSMIL_MoE_Multiscale(num_experts=4)

        self.linear_C = None
        if self.clin_dim > 0:
            self.linear_C = nn.Sequential(
                nn.LayerNorm(self.clin_dim),
                nn.Linear(self.clin_dim, Cross_embed_dim),
                nn.GELU(),
                nn.LayerNorm(Cross_embed_dim),
            )
        self.linear_H = nn.Linear(self.split_dims, Cross_embed_dim)
        self.CrossBlock = MultiLayerCrossModalAttention(
            num_layers=Cross_num_layers,
            embed_dim=Cross_embed_dim,
            num_heads=Cross_num_heads,
        )
        self.pool = Pooling(Cross_embed_dim, hidden=Cross_embed_dim * 2)

        if self.trans_perciever:
            self.trans = PerceiverIOSequenceEncoder(
                in_dim=Cross_embed_dim,
                dim=Cross_embed_dim,
                depth=3,
                heads=4,
                num_latents=int(ceil(Cross_embed_dim / 4)),
                dropout=0.2,
                ffn_mult=4.0,
                add_pos_emb=True,
                track_attn=True,
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
                use_MQA=use_MQA,
            )
            self.enc_return_attention = enc_return_attention
            self.enc_last_only = enc_last_only
            self.enc_topk = enc_topk
            self.enc_cpu_offload = enc_cpu_offload

        self.slide_token = nn.Parameter(torch.randn(1, 1, Cross_embed_dim) * 0.02)
        if Cross_embed_dim % joint_heads != 0:
            raise ValueError(f"Cross_embed_dim={Cross_embed_dim} must be divisible by joint_heads={joint_heads}")
        joint_layer = nn.TransformerEncoderLayer(
            d_model=Cross_embed_dim,
            nhead=joint_heads,
            dim_feedforward=Cross_embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.joint_encoder = nn.TransformerEncoder(joint_layer, num_layers=1)
        self.joint_norm = nn.LayerNorm(Cross_embed_dim)
        if regression:
            self.head = regression_head(input_dim=Cross_embed_dim, output_dim=Classify_dim)
        else:
            self.head = ClassificationHead_layernorm(input_dim=Cross_embed_dim, output_dim=Classify_dim)

    def _fuse_pathology_features(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 4:
            raise ValueError(f"Expected pathology tensor (B,C,M,N), got {tuple(feat.shape)}")
        if self.use_multiscale:
            fused = self.Scale_Fusion(feat)
            if fused.ndim == 4:
                fused = fused.squeeze(-1)
        else:
            if not (0 <= self.view_index < feat.size(-1)):
                raise ValueError(f"view_index={self.view_index} is out of range for N={feat.size(-1)}")
            fused = feat[..., self.view_index]
        return fused.contiguous()

    def _make_query_tokens(self, feat_enc: torch.Tensor, clin: torch.Tensor) -> torch.Tensor:
        B = feat_enc.size(0)
        slide = self.slide_token.expand(B, -1, -1)
        if self.clin_dim == 0:
            return slide
        if clin.dim() != 2 or clin.size(1) != self.clin_dim:
            raise ValueError(f"Auxiliary feature shape mismatch: got {tuple(clin.shape)}, expected (B,{self.clin_dim})")
        clin_tok = self.linear_C(clin).unsqueeze(1)
        return torch.cat([slide, clin_tok], dim=1)

    def forward(self, x: torch.Tensor):
        feat, clin = split_tensor(x, split_dims=self.split_dims, dim=1)
        if clin.size(1) != self.clin_dim:
            raise ValueError(f"Input has {clin.size(1)} auxiliary channels, but clin_dim={self.clin_dim}")
        feat_fused = self._fuse_pathology_features(feat)
        feat_proj = self.linear_H(feat_fused.transpose(1, 2))
        if self.trans_perciever:
            feat_enc, enc_attn = self.trans(feat_proj)
        else:
            feat_enc, enc_attn = self.trans(
                feat_proj,
                return_attention=self.enc_return_attention,
                last_only=self.enc_last_only,
                topk=self.enc_topk,
                cpu_offload=self.enc_cpu_offload,
            )
        query = self._make_query_tokens(feat_enc, clin)
        cross_out, attn_weights = self.CrossBlock(query=query, key=feat_enc, value=feat_enc)
        joint_out = self.joint_encoder(cross_out)
        slide_emb = self.joint_norm(joint_out[:, 0])
        output = self.head(slide_emb).to(dtype=torch.float32)
        self.multi_score = {
            "enc_attn": enc_attn,
            "cross_subtype": attn_weights,
            "feat_histo": feat_enc,
            "feat_before_head": slide_emb,
        }
        return output


XBMModel = utils_multiScale_model_trans
MultiScaleCrossModalXBM = utils_multiScale_model_trans
