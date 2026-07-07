# Model Folder Notes

This folder keeps the main XBM implementation unchanged and adds the model definitions needed to make the architecture described in the manuscript traceable from the repository.

## Main model

- `XBM.py`
  - Main XBM implementation.
  - Default final-task input uses `split_dims=1536` and `clin_dim=57`.
  - Supports pathology-only operation with `clin_dim=0`.
  - Uses FC-AttnPooling for same-FOV scale fusion by default.
  - This file was not changed when adding the additional ablation/baseline components.

## Existing baseline models

- `multi_attnmil.py`
  - Attention-MIL baseline.
  - Supports multiscale and single-view modes.

- `multi_transmil_multiscale_multimodal.py`
  - TransMIL-style baseline with optional multiscale and multimodal inputs.

## Added model definitions

These files are provided as architecture definitions only. Full baseline or ablation reproduction scripts are intentionally not included.

| Path | Purpose | Source in task/model archive |
|---|---|---|
| `model/ablations/clinical_only.py` | Clinical-only modality ablation model | `utils_clincial.py` / `Config_clinical_1_12` |
| `model/ablations/pathology_only.py` | Pathology-only modality ablation model | `utils_multiscale_single_histo.py` / `Config_WGD_TP53_singlehis_1_12` |
| `model/ablations/single_scale.py` | Single-scale model for multiscale ablation | `singleFCattntransmil.py` / `Config_subtask19_1_26` |
| `model/baselines/transmil_lite.py` | TransMIL-Lite baseline with low-rank QKV, MQA, and GEGLU | `MultiScaleMultiModalTransMIL_Lite.py` / `Config_subtask17_1_26` |
| `model/components_extra/coam_fusion.py` | COAM/Pyramid-style same-FOV multiscale fusion component | `MultiScale/COAM_Fusion.py` |
| `model/components_extra/transformer.py` | Standard Transformer encoder component | `Transformer.py` |
| `model/components_extra/attention.py` | Standard multi-head self-attention component | `multihead_Attention.py` |
| `model/components_extra/feedforward.py` | Transformer feed-forward component | `FeedForward.py` |
| `model/components_extra/mlp_encoder.py` | MLP encoder component for MIL-style models | `MLPEncoder.py` |
| `model/components_extra/gated_attention.py` | Gated attention component for MIL-style models | `Gated_attention.py` |

## Submodules retained from the previous version

```text
submodel/
├── FC_AttnPool_Fusion.py
├── CSMIL_Fusion.py
├── CrossModalMultiHeadAttention.py
├── utils_trans_cross_fusion.py
├── ClassificationHead_layernorm.py
├── regression_head.py
├── LongTokenSeqEncoder.py
└── TokenViT_2.py
```

The existing files above were kept in place. Additional components were placed in separate folders to avoid changing the main model or its original submodules.
