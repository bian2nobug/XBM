"""Pathology-only XBM-style model for modality ablation."""

from __future__ import annotations

from math import ceil

import torch
import torch.nn as nn

from model.submodel.CSMIL_Fusion import CSMIL_MoE_Multiscale
from model.submodel.ClassificationHead_layernorm import ClassificationHead_layernorm
from model.submodel.FC_AttnPool_Fusion import FC_AttnPool_Fusion
from model.submodel.LongTokenSeqEncoder import PerceiverIOSequenceEncoder
from model.submodel.TokenViT_2 import TokenViT
from model.submodel.regression_head import regression_head
from model.submodel.utils_trans_cross_fusion import AttnPool1D


class PathologyOnlyXBM(nn.Module):
    """Pathology-only ablation model.

    Expected input shape is `(B, split_dims, M, N)` or
    `(B, split_dims + aux_dim, M, N)`. Only the first `split_dims` channels are
    used. Clinical or other auxiliary channels are ignored.
    """

    def __init__(
        self,
        split_dims: int = 1536,
        Cross_embed_dim: int = 256,
        Pooling=AttnPool1D,
        Classify_dim: int = 1,
        Fusion_PyramidProgressive: bool = True,
        trans_perciever: bool = True,
        regression: bool = False,
        use_geglu: bool = True,
        use_MQA: bool = True,
        use_multiscale: bool = True,
        view_index: int = 0,
        num_tokens: int = 1500,
        joint_heads: int = 4,
        **_: object,
    ):
        super().__init__()
        self.split_dims = int(split_dims)
        self.use_multiscale = bool(use_multiscale)
        self.view_index = int(view_index)
        self.trans_perciever = bool(trans_perciever)
        self.num_tokens = int(num_tokens)

        self.Scale_Fusion = (
            FC_AttnPool_Fusion(dim=self.split_dims)
            if Fusion_PyramidProgressive
            else CSMIL_MoE_Multiscale(num_experts=4)
        )
        self.linear_H = nn.Linear(self.split_dims, Cross_embed_dim)

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
        else:
            self.trans = TokenViT(
                in_dim=Cross_embed_dim,
                num_tokens=self.num_tokens,
                dim=Cross_embed_dim,
                depth=1,
                heads=4,
                dim_head=int(ceil(Cross_embed_dim / 4)),
                mlp_dim=Cross_embed_dim * 2,
                use_geglu=use_geglu,
                use_MQA=use_MQA,
            )

        self.pool = Pooling(Cross_embed_dim, hidden=Cross_embed_dim * 2)
        if Cross_embed_dim % joint_heads != 0:
            raise ValueError(f"Cross_embed_dim={Cross_embed_dim} must be divisible by joint_heads={joint_heads}")
        self.slide_token = nn.Parameter(torch.randn(1, 1, Cross_embed_dim) * 0.02)
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
        self.head = (
            regression_head(input_dim=Cross_embed_dim, output_dim=Classify_dim)
            if regression
            else ClassificationHead_layernorm(input_dim=Cross_embed_dim, output_dim=Classify_dim)
        )

    def _fuse_pathology(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 4 or feat.size(1) < self.split_dims:
            raise ValueError(f"Expected input shape (B,>= {self.split_dims},M,N), got {tuple(feat.shape)}")
        feat = feat[:, :self.split_dims]
        if self.use_multiscale:
            fused = self.Scale_Fusion(feat)
            return fused.squeeze(-1) if fused.ndim == 4 else fused
        if not (0 <= self.view_index < feat.size(-1)):
            raise ValueError(f"view_index={self.view_index} is out of range for N={feat.size(-1)}")
        return feat[..., self.view_index].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_fused = self._fuse_pathology(x)
        feat_proj = self.linear_H(feat_fused.transpose(1, 2))
        if self.trans_perciever:
            feat_enc, enc_attn = self.trans(feat_proj)
        else:
            feat_enc, enc_attn = self.trans(feat_proj, return_attention=True, last_only=True)
        B = feat_enc.size(0)
        slide = self.slide_token.expand(B, -1, -1)
        joint = self.joint_encoder(slide + self.pool(feat_enc).unsqueeze(1))
        slide_emb = self.joint_norm(joint[:, 0])
        self.multi_score = {"enc_attn": enc_attn, "feat_histo": feat_enc, "feat_before_head": slide_emb}
        return self.head(slide_emb).to(dtype=torch.float32)


# Backward-compatible aliases for legacy naming.
utils_multiScale_single_histo = PathologyOnlyXBM
PathologyOnlyModel = PathologyOnlyXBM

__all__ = ["PathologyOnlyXBM", "PathologyOnlyModel", "utils_multiScale_single_histo"]
