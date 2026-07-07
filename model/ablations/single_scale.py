"""Single-scale pathology-clinical model for multiscale ablation."""

from __future__ import annotations

from math import ceil
from typing import Tuple

import torch
import torch.nn as nn

from model.submodel.ClassificationHead_layernorm import ClassificationHead_layernorm
from model.submodel.CrossModalMultiHeadAttention import MultiLayerCrossModalAttention
from model.submodel.LongTokenSeqEncoder import PerceiverIOSequenceEncoder
from model.submodel.TokenViT_2 import TokenViT
from model.submodel.regression_head import regression_head
from model.submodel.utils_trans_cross_fusion import AttnPool1D


def split_tensor(x: torch.Tensor, split_dims: int = 1536, dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape={tuple(x.shape)}")
    C = x.size(dim)
    if not (1 <= split_dims < C):
        raise ValueError(f"split_dims must be in [1, {C - 1}], got split_dims={split_dims}, C={C}")
    left, right = torch.split(x, [split_dims, C - split_dims], dim=dim)
    return left.contiguous(), right[..., 0, 0].contiguous()


class SingleScaleXBM(nn.Module):
    """XBM-style model using one selected view instead of 21-view same-FOV fusion."""

    def __init__(
        self,
        split_dims: int = 1536,
        clin_dim: int = 57,
        Cross_num_layers: int = 2,
        Cross_embed_dim: int = 256,
        Cross_num_heads: int = 2,
        Pooling=AttnPool1D,
        Classify_dim: int = 1,
        trans_perciever: bool = True,
        regression: bool = False,
        use_geglu: bool = True,
        use_MQA: bool = True,
        view_index: int = 0,
        num_tokens: int = 1500,
        joint_heads: int = 4,
        **_: object,
    ):
        super().__init__()
        self.split_dims = int(split_dims)
        self.clin_dim = int(clin_dim)
        self.view_index = int(view_index)
        self.trans_perciever = bool(trans_perciever)
        self.num_tokens = int(num_tokens)

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
        self.head = (
            regression_head(input_dim=Cross_embed_dim, output_dim=Classify_dim)
            if regression
            else ClassificationHead_layernorm(input_dim=Cross_embed_dim, output_dim=Classify_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, clin = split_tensor(x, split_dims=self.split_dims, dim=1)
        if clin.size(1) != self.clin_dim:
            raise ValueError(f"Clinical feature dimension mismatch: expected {self.clin_dim}, got {clin.size(1)}")
        if not (0 <= self.view_index < feat.size(-1)):
            raise ValueError(f"view_index={self.view_index} is out of range for N={feat.size(-1)}")
        feat_single = feat[..., self.view_index].contiguous()
        feat_proj = self.linear_H(feat_single.transpose(1, 2))
        if self.trans_perciever:
            feat_enc, enc_attn = self.trans(feat_proj)
        else:
            feat_enc, enc_attn = self.trans(feat_proj, return_attention=True, last_only=True)

        B = feat_enc.size(0)
        slide = self.slide_token.expand(B, -1, -1)
        clin_tok = self.linear_C(clin).unsqueeze(1)
        query = torch.cat([slide, clin_tok], dim=1)
        cross_out, attn_weights = self.CrossBlock(query=query, key=feat_enc, value=feat_enc)
        joint_out = self.joint_encoder(cross_out)
        slide_emb = self.joint_norm(joint_out[:, 0])
        self.multi_score = {"enc_attn": enc_attn, "cross_subtype": attn_weights}
        return self.head(slide_emb).to(dtype=torch.float32)


# Backward-compatible alias for legacy single-scale task naming.
singleFCattntransmil = SingleScaleXBM

__all__ = ["SingleScaleXBM", "singleFCattntransmil", "split_tensor"]
