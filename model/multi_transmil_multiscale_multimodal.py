"""
Multi-scale compatible TransMIL baseline.

Design principle
----------------
1) When use_multiscale=False and use_multimodal=False, the model follows the
   standard TransMIL aggregation pipeline:
      instance projection -> square padding -> CLS token -> TransLayer
      -> PPEG positional encoding -> TransLayer -> CLS classification.

2) When use_multiscale=True, only a scale-attention pooling module is added
   before the TransMIL backbone. The downstream TransMIL backbone is unchanged,
   which makes the multi-scale ablation fair and easy to interpret.

3) Optional patient-level auxiliary variables are fused after slide-level
   TransMIL aggregation. They are not inserted into the spatial instance grid by
   default, because that would violate the spatial-token assumption of PPEG.

Expected 4D input
-----------------
x: (B, n_feats + aux_dim, M, N)
   B: batch size
   n_feats: pathology feature dimension, e.g. 1536 for Prov-GigaPath
   aux_dim: optional patient-level auxiliary feature dimension
   M: number of tile instances per slide
   N: number of scale/view features per instance

Classic single-scale TransMIL mode
----------------------------------
model = MultiScaleMultiModalTransMIL(
    n_feats=1536,
    n_out=1,
    aux_dim=0,
    use_multiscale=False,
    use_multimodal=False,
    view_index=0,
)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from nystrom_attention import NystromAttention
    _HAS_NYSTROM = True
except Exception:
    NystromAttention = None
    _HAS_NYSTROM = False


class ScaleAttentionPooling(nn.Module):
    """
    Attention pooling over the scale/view dimension.

    Input:  x (B, C, M, N)
    Output: y (B, C, M)
    """

    def __init__(self, dim: int = 1536) -> None:
        super().__init__()
        self.scorer = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B,C,M,N), got shape={tuple(x.shape)}")

        # (B,C,M,N) -> (B,M,N,C)
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        score = self.scorer(x_perm)              # (B,M,N,1)
        weight = F.softmax(score, dim=2)         # (B,M,N,1)
        fused = (weight * x_perm).sum(dim=2)     # (B,M,C)
        fused = fused.permute(0, 2, 1).contiguous()  # (B,C,M)

        if return_attention:
            return fused, weight.squeeze(-1)     # weight: (B,M,N)
        return fused


class TransLayer(nn.Module):
    """
    TransMIL-style transformer layer.

    It uses Nyström attention when the optional `nystrom-attention` package is
    available. Otherwise, it falls back to PyTorch MultiheadAttention so the file
    remains runnable in a minimal environment.
    """

    def __init__(
        self,
        dim: int = 512,
        heads: int = 8,
        dropout: float = 0.1,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.use_nystrom = _HAS_NYSTROM

        if self.use_nystrom:
            self.attn = NystromAttention(
                dim=dim,
                dim_head=max(dim // heads, 1),
                heads=heads,
                num_landmarks=num_landmarks,
                pinv_iterations=pinv_iterations,
                residual=True,
                dropout=dropout,
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.use_nystrom:
            h = self.attn(h)
        else:
            h, _ = self.attn(h, h, h, need_weights=False)
        return x + h


class PPEG(nn.Module):
    """
    Pyramid Position Encoding Generator used in TransMIL.

    The CLS token is kept separate. Patch tokens are reshaped to a square grid,
    passed through depth-wise convolutions, and then flattened back.
    """

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.proj_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.proj_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.proj_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        cls_token = x[:, :1, :]          # (B,1,D)
        feat_token = x[:, 1:, :]         # (B,H*W,D)
        B, _, D = feat_token.shape

        cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, D, H, W)
        cnn_feat = cnn_feat + self.proj_7(cnn_feat) + self.proj_5(cnn_feat) + self.proj_3(cnn_feat)
        feat_token = cnn_feat.flatten(2).transpose(1, 2).contiguous()

        return torch.cat((cls_token, feat_token), dim=1)


class TransMILBackbone(nn.Module):
    """
    Standard TransMIL slide-level aggregator.

    Input:  instances (B, M, n_feats)
    Output: slide embedding (B, d_model)
    """

    def __init__(
        self,
        n_feats: int = 1536,
        d_model: int = 512,
        heads: int = 8,
        dropout: float = 0.1,
        num_landmarks: int = 256,
    ) -> None:
        super().__init__()
        self.n_feats = n_feats
        self.d_model = d_model

        self.instance_proj = nn.Sequential(
            nn.Linear(n_feats, d_model),
            nn.ReLU(inplace=True),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.layer1 = TransLayer(dim=d_model, heads=heads, dropout=dropout, num_landmarks=num_landmarks)
        self.ppeg = PPEG(dim=d_model)
        self.layer2 = TransLayer(dim=d_model, heads=heads, dropout=dropout, num_landmarks=num_landmarks)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _pad_to_square(x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        """
        Pad instance tokens to a square number by repeating early tokens,
        following the common TransMIL implementation style.
        """
        B, L, D = x.shape
        if L <= 0:
            raise ValueError("The instance sequence is empty.")

        H = int(math.ceil(math.sqrt(L)))
        W = H
        target_len = H * W
        add_len = target_len - L

        if add_len > 0:
            repeat_token = x[:, :add_len, :]
            if repeat_token.size(1) < add_len:
                repeat_factor = int(math.ceil(add_len / max(repeat_token.size(1), 1)))
                repeat_token = repeat_token.repeat(1, repeat_factor, 1)[:, :add_len, :]
            x = torch.cat([x, repeat_token], dim=1)

        return x, H, W, add_len

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        if instances.ndim != 3:
            raise ValueError(f"Expected instances with shape (B,M,C), got {tuple(instances.shape)}")
        if instances.size(-1) != self.n_feats:
            raise ValueError(f"Expected feature dim={self.n_feats}, got {instances.size(-1)}")

        h = self.instance_proj(instances)       # (B,M,D)
        h, H, W, _ = self._pad_to_square(h)     # (B,H*W,D)

        cls_token = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat((cls_token, h), dim=1)    # (B,1+H*W,D)

        h = self.layer1(h)
        h = self.ppeg(h, H, W)
        h = self.layer2(h)
        h = self.norm(h)

        return h[:, 0]                          # CLS slide embedding


class MultiScaleMultiModalTransMIL(nn.Module):
    """
    TransMIL baseline with optional multi-scale input fusion and optional
    patient-level auxiliary fusion.

    Set use_multiscale=False and use_multimodal=False for the classic
    single-scale TransMIL baseline.
    """

    def __init__(
        self,
        n_feats: int = 1536,
        n_out: int = 1,
        aux_dim: int = 0,
        use_multiscale: bool = False,
        use_multimodal: bool = False,
        view_index: int = 0,
        d_model: int = 512,
        heads: int = 8,
        dropout: float = 0.1,
        num_landmarks: int = 256,
    ) -> None:
        super().__init__()
        self.n_feats = n_feats
        self.n_out = n_out
        self.aux_dim = aux_dim
        self.use_multiscale = use_multiscale
        self.use_multimodal = bool(use_multimodal and aux_dim > 0)
        self.view_index = view_index

        self.scale_fusion = ScaleAttentionPooling(dim=n_feats)
        self.backbone = TransMILBackbone(
            n_feats=n_feats,
            d_model=d_model,
            heads=heads,
            dropout=dropout,
            num_landmarks=num_landmarks,
        )

        if self.use_multimodal:
            self.aux_proj = nn.Sequential(
                nn.Linear(aux_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_out),
            )
        else:
            self.head = nn.Linear(d_model, n_out)

    def _split_4d_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B, n_feats + aux_dim, M, N)
        returns:
            img: (B, n_feats, M, N)
            aux: (B, aux_dim) or None
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B,C,M,N), got {tuple(x.shape)}")
        if x.size(1) < self.n_feats:
            raise ValueError(f"Input channel dim={x.size(1)} is smaller than n_feats={self.n_feats}")

        img = x[:, :self.n_feats, :, :].contiguous()
        aux = None

        if self.use_multimodal:
            expected_c = self.n_feats + self.aux_dim
            if x.size(1) != expected_c:
                raise ValueError(f"Expected channel dim={expected_c}, got {x.size(1)}")
            aux = x[:, self.n_feats:, :, :][..., 0, 0].contiguous()

        return img, aux

    def _prepare_instances(self, img: torch.Tensor, return_scale_attention: bool = False):
        """
        img: (B,n_feats,M,N)
        returns instances: (B,M,n_feats)
        """
        scale_attention = None

        if self.use_multiscale:
            if return_scale_attention:
                img, scale_attention = self.scale_fusion(img, return_attention=True)  # (B,C,M), (B,M,N)
            else:
                img = self.scale_fusion(img)                                         # (B,C,M)
        else:
            if not (0 <= self.view_index < img.size(-1)):
                raise IndexError(f"view_index={self.view_index} is out of range for N={img.size(-1)}")
            img = img[..., self.view_index]                                          # (B,C,M)

        instances = img.permute(0, 2, 1).contiguous()                                # (B,M,C)
        return instances, scale_attention

    def forward(self, x: torch.Tensor, return_features: bool = False, return_scale_attention: bool = False):
        """
        Returns logits by default.

        If return_features=True, returns a dict containing logits and slide_emb.
        If return_scale_attention=True and use_multiscale=True, also returns
        scale attention weights with shape (B,M,N).
        """
        if x.ndim == 3:
            # Direct classic TransMIL input: (B,M,n_feats)
            if x.size(-1) != self.n_feats:
                raise ValueError(f"Expected 3D input feature dim={self.n_feats}, got {x.size(-1)}")
            instances = x.contiguous()
            aux = None
            scale_attention = None
        else:
            img, aux = self._split_4d_input(x)
            instances, scale_attention = self._prepare_instances(img, return_scale_attention=return_scale_attention)

        slide_emb = self.backbone(instances)                                         # (B,D)

        if self.use_multimodal:
            if aux is None:
                raise ValueError("Auxiliary features are required when use_multimodal=True.")
            aux_emb = self.aux_proj(aux)
            pred_emb = torch.cat([slide_emb, aux_emb], dim=-1)
        else:
            pred_emb = slide_emb

        logits = self.head(pred_emb)

        if return_features or return_scale_attention:
            output = {"logits": logits, "slide_emb": slide_emb}
            if return_scale_attention:
                output["scale_attention"] = scale_attention
            return output
        return logits


# Backward-compatible aliases.
ClassicTransMIL = TransMILBackbone
MultiScaleCompatibleTransMIL = MultiScaleMultiModalTransMIL


if __name__ == "__main__":
    torch.manual_seed(0)

    B, M, N = 2, 1500, 21
    n_feats, aux_dim, n_out = 1536, 57, 4
    x = torch.randn(B, n_feats + aux_dim, M, N)

    # 1) Classic single-scale TransMIL baseline.
    model_classic = MultiScaleMultiModalTransMIL(
        n_feats=n_feats,
        n_out=n_out,
        aux_dim=0,
        use_multiscale=False,
        use_multimodal=False,
        view_index=0,
    )
    print("classic:", model_classic(x[:, :n_feats]).shape)

    # 2) Multi-scale TransMIL baseline.
    model_ms = MultiScaleMultiModalTransMIL(
        n_feats=n_feats,
        n_out=n_out,
        aux_dim=0,
        use_multiscale=True,
        use_multimodal=False,
    )
    print("multiscale:", model_ms(x[:, :n_feats]).shape)

    # 3) Multi-scale + auxiliary fusion.
    model_ms_aux = MultiScaleMultiModalTransMIL(
        n_feats=n_feats,
        n_out=n_out,
        aux_dim=aux_dim,
        use_multiscale=True,
        use_multimodal=True,
    )
    out = model_ms_aux(x, return_features=True, return_scale_attention=True)
    print("multiscale+aux:", out["logits"].shape, out["slide_emb"].shape, out["scale_attention"].shape)
