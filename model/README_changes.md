# Revised main model files

This folder contains the revised main model files only. Put the `submodel/` folder at the same level when using `utils_multiscale_trans_attnpooling_change.py`.

## Files

- `multi_attnmil.py`
  - Cleaned Attention MIL baseline.
  - Supports `use_multiscale=True/False` and `view_index`.
  - Supports optional auxiliary token.
  - Fixes the auxiliary-token masking issue by using an explicit `valid_mask`.
  - Uses `LayerNorm` head by default for small-batch MIL stability.

- `multi_transmil_multiscale_multimodal.py`
  - Standard-compatible TransMIL baseline with optional scale-attention input fusion.
  - `use_multiscale=False` and `use_multimodal=False` gives single-scale TransMIL mode.

- `utils_multiscale_trans_attnpooling_change.py`
  - XBM main model.
  - Removed hard-coded server paths and `sys.path.append`.
  - Imports dependencies from `submodel/`.
  - Added `use_multiscale` and `view_index` so the multi-scale branch can be switched off for ablation.

## Required layout

```text
project/
├── multi_attnmil.py
├── multi_transmil_multiscale_multimodal.py
├── utils_multiscale_trans_attnpooling_change.py
└── submodel/
    ├── FC_AttnPool_Fusion.py
    ├── CSMIL_Fusion.py
    ├── CrossModalMultiHeadAttention.py
    ├── utils_trans_cross_fusion.py
    ├── ClassificationHead_layernorm.py
    ├── regression_head.py
    ├── LongTokenSeqEncoder.py
    └── TokenViT_2.py
```
