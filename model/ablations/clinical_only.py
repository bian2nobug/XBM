"""Clinical-only model for modality ablation."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.submodel.ClassificationHead_layernorm import ClassificationHead_layernorm
from model.submodel.regression_head import regression_head
from model.submodel.utils_trans_cross_fusion import AttnPool1D


class ClinicalOnlyModel(nn.Module):
    """Clinical-only predictor.

    Accepts either a clinical matrix `(B, clin_dim)` or the full XBM tensor
    `(B, split_dims + clin_dim, M, N)`. In the latter case, pathology channels
    are ignored and clinical channels are read from `[..., 0, 0]`.
    """

    def __init__(
        self,
        split_dims: int = 1536,
        clin_dim: int = 57,
        Cross_embed_dim: int = 256,
        Pooling=AttnPool1D,
        Classify_dim: int = 1,
        regression: bool = False,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.split_dims = int(split_dims)
        self.clin_dim = int(clin_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.clin_dim),
            nn.Linear(self.clin_dim, Cross_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(Cross_embed_dim),
        )
        self.pool = Pooling(Cross_embed_dim, hidden=Cross_embed_dim * 2)
        self.head = (
            regression_head(input_dim=Cross_embed_dim, output_dim=Classify_dim)
            if regression
            else ClassificationHead_layernorm(input_dim=Cross_embed_dim, output_dim=Classify_dim)
        )

    def _extract_clinical(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            clin = x
        elif x.ndim == 4:
            if x.size(1) < self.split_dims + self.clin_dim:
                raise ValueError(
                    f"Expected at least {self.split_dims + self.clin_dim} channels, got {x.size(1)}"
                )
            clin = x[:, self.split_dims:self.split_dims + self.clin_dim, 0, 0]
        else:
            raise ValueError(f"Expected input shape (B,{self.clin_dim}) or (B,C,M,N), got {tuple(x.shape)}")
        if clin.size(1) != self.clin_dim:
            raise ValueError(f"Clinical feature dimension mismatch: expected {self.clin_dim}, got {clin.size(1)}")
        return clin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clin = self._extract_clinical(x)
        token = self.encoder(clin).unsqueeze(1)
        pooled = self.pool(token)
        self.multi_score = {"clinical_token": token}
        return self.head(pooled).to(dtype=torch.float32)


# Backward-compatible alias used by legacy task configs.
utils_multiScale_model_trans = ClinicalOnlyModel

__all__ = ["ClinicalOnlyModel", "utils_multiScale_model_trans"]
