"""Additional ablation model definitions.

These modules are provided for architectural reference and for users who need
pathology-only, clinical-only, or single-scale model definitions.
"""

from .clinical_only import ClinicalOnlyModel
from .pathology_only import PathologyOnlyModel, PathologyOnlyXBM
from .single_scale import SingleScaleXBM

__all__ = ["ClinicalOnlyModel", "PathologyOnlyModel", "PathologyOnlyXBM", "SingleScaleXBM"]
