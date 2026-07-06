"""
===============================================================================
Reducers - attribution reducers (XBM_MSS_WGD)
===============================================================================

[Purpose]
    Reduce raw IG attribution by summing over specified dims, projecting to different granularities

[Built-in reducers]
    - MultiScaleMILReducer: multi-scale MIL model (pathology + clinical fusion)
    - SequenceReducer: sequence model (NLP/time-series)
    - ImageReducer: image model (CNN/ViT)
    - CustomReducer: custom reduction function

[Dependency]
    ig_core.BaseAttributionReducer
===============================================================================
"""

import torch
from typing import Dict, List, Optional
from ig_core import BaseAttributionReducer


class MultiScaleMILReducer(BaseAttributionReducer):
    """
    Multi-scale MIL attribution reducer

    Reduces attribution only for the pathology branch (no clinical/modality).

    Data structure assumptions:
        - Input: [B, C, M, N]  (the first pathology_dim channels of C are pathology features)
        - N = multi-scale channels (e.g. 1+4+16=21 for 5x+10x+20x)

        Raw attribution [B, C, 1500, 21]
                    |
                    v
        pathology [B,1536,1500,21]
            |
            +--> sum(1,3) -> instance [B,1500]  <- patch importance
            +--> slice by scale -> sum(1,2,3) -> scale [B,3]  <- scale importance
            +--> sum(2,3) -> feature [B,1536]  <- feature importance

    Args:
        pathology_dim: pathology feature dimension (e.g. 1536)
        clinical_dim: clinical feature dimension (used only to determine the pathology slice boundary, not attributed)
        scale_splits: list of scale split points (e.g. [1, 5, 21] meaning 0:1, 1:5, 5:21)
        scale_names: list of scale names (e.g. ['5x', '10x', '20x'])
    """
    
    def __init__(self,
                 pathology_dim: int = 1536,
                 clinical_dim: int = 57,
                 scale_splits: List[int] = [1, 5, 21],
                 scale_names: List[str] = ['5x', '10x', '20x']):
        self.pathology_dim = pathology_dim
        self.clinical_dim = clinical_dim
        self.scale_splits = scale_splits
        self.scale_names = scale_names
    
    def reduce(self, attributions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Reduce attribution to multiple analysis granularities

        Args:
            attributions: raw IG attribution [B, C, M, N]

        Returns:
            dict: multi-level attribution results
        """
        B = attributions.shape[0]

        # take only the pathology attribution (first pathology_dim channels), no clinical/modality
        attr_pathology = attributions[:, :self.pathology_dim, :, :]

        # instance-level attribution (patch importance)
        instance_attribution = attr_pathology.sum(dim=(1, 3))

        # scale-level attribution
        scale_attrs = []
        prev_idx = 0
        for i, split_idx in enumerate(self.scale_splits):
            scale_attr = attr_pathology[..., prev_idx:split_idx].sum(dim=(1, 2, 3))
            scale_attrs.append(scale_attr)
            prev_idx = split_idx
        scale_attribution = torch.stack(scale_attrs, dim=1)

        # feature-level attribution
        feature_attribution = attr_pathology.sum(dim=(2, 3))

        return {
            'instance_attribution': instance_attribution,
            'scale_attribution': scale_attribution,
            'scale_names': self.scale_names,
            'feature_attribution': feature_attribution,
            'full_attribution': attributions
        }


class SequenceReducer(BaseAttributionReducer):
    """Sequence model attribution reducer"""
    
    def __init__(self, aggregate_embedding: bool = True):
        self.aggregate_embedding = aggregate_embedding
    
    def reduce(self, attributions: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.aggregate_embedding:
            token_attribution = attributions.sum(dim=-1)
        else:
            token_attribution = attributions
        return {
            'token_attribution': token_attribution,
            'total': attributions.sum(),
            'full_attribution': attributions
        }


class ImageReducer(BaseAttributionReducer):
    """Image model attribution reducer"""
    
    def __init__(self, aggregate_channels: bool = True):
        self.aggregate_channels = aggregate_channels
    
    def reduce(self, attributions: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.aggregate_channels:
            spatial_attribution = attributions.sum(dim=1)
        else:
            spatial_attribution = attributions
        channel_attribution = attributions.sum(dim=(2, 3))
        return {
            'spatial_attribution': spatial_attribution,
            'channel_attribution': channel_attribution,
            'total': attributions.sum(),
            'full_attribution': attributions
        }


class CustomReducer(BaseAttributionReducer):
    """Custom attribution reducer"""
    
    def __init__(self, reduce_fn):
        self.reduce_fn = reduce_fn
    
    def reduce(self, attributions: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.reduce_fn(attributions)

