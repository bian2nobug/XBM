# Model Inventory

| Group | File | Main class / alias | Use case |
|---|---|---|---|
| Main | `model/XBM.py` | `XBMModel`, `utils_multiScale_model_trans` | Final XBM model entry point |
| Existing baseline | `model/multi_attnmil.py` | Attention-MIL classes | AttMIL comparison |
| Existing baseline | `model/multi_transmil_multiscale_multimodal.py` | TransMIL-style classes | TransMIL comparison |
| Added ablation | `model/ablations/clinical_only.py` | `ClinicalOnlyModel` | Clinical-only modality ablation |
| Added ablation | `model/ablations/pathology_only.py` | `PathologyOnlyXBM` | Pathology-only modality ablation |
| Added ablation | `model/ablations/single_scale.py` | `SingleScaleXBM` | Single-scale / no-multiscale ablation |
| Added baseline | `model/baselines/transmil_lite.py` | `MultiScaleMultiModalTransMIL_Lite` | TransMIL-Lite architecture reference |
| Added component | `model/components_extra/coam_fusion.py` | `CSMIL_PyramidProgressive` | COAM/Pyramid-style multiscale fusion |
| Added component | `model/components_extra/transformer.py` | `Transformer_Encoder` | Standard Transformer encoder reference |
| Added component | `model/components_extra/attention.py` | `Attention` | Standard MHA reference |
| Added component | `model/components_extra/feedforward.py` | `FeedForward` | Standard FFN reference |
| Added component | `model/components_extra/mlp_encoder.py` | `MLPEncoder` | MIL-style feature encoder |
| Added component | `model/components_extra/gated_attention.py` | `Gated_attention` | MIL-style gated attention |

No trained checkpoints, WSI data, feature tensors, or ablation result tables are included in this repository.


These files provide model and component definitions. They are not intended to represent a complete end-to-end ablation-reproduction workflow. See `../THIRD_PARTY_NOTICES.md` for third-party software and method references.
