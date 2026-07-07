# XBM Interpretability and WSI Heatmap Analysis

This folder contains command-line utilities for model inspection and WSI-level visualization in XBM. The scripts should be run from the repository root.

## Folder Layout

```text
analysis/interpretability/
├── _common.py
├── run_random_heatmap_smoke.py
├── run_random_model_forward_smoke.py
├── run_integrated_gradients.py
├── run_ig_heatmap.py
├── run_cross_axg.py
├── run_axg_heatmap.py
├── run_scale_fusion_heatmap.py
├── examples/
│   └── tcga_smoke_commands.sh
└── shared/
    ├── heatmap_tools/
    ├── ig_core/
    └── model_lib/
```

`shared/model_lib/` provides the interpretability-compatible XBM adapter used by the attribution scripts. The main training model is `model/XBM.py`.

## Common Inputs

The attribution scripts use four common inputs:

```text
multi_tensor.pth     # torch tensor shaped [N, C, M, 21]
label_tensor.pt      # torch tensor shaped [N]
sample_id.txt        # one SampleID per line
best_model.pt        # XBM checkpoint
```

For the final WGD x TP53 setting, typical model arguments are:

```bash
--split-dims 1536
--clin-dim 57
--class-dim 4
```

For binary WGD, use `--class-dim 1`.

## Coordinate Inputs

WSI heatmap scripts read coordinates from an H5 file. Supported coordinate keys are:

```text
locations
locations_5x_in_20x
coords
coordinates
```

Use `--coord-index-npy` when model-instance order is stored separately from the H5 coordinate order. When the model scores are already aligned to the first `N` H5 coordinates, add `--allow-first-n-coords`.

## 1. WSI Heatmap Smoke Test

```bash
python analysis/interpretability/run_random_heatmap_smoke.py \
  --wsi /path/to/SampleID/HE.svs \
  --h5 /path/to/SampleID/HE.h5 \
  --out-dir /tmp/xbm_random_heatmap \
  --patch-level 2 \
  --patch-size 512 \
  --normalize-method rank \
  --save-thumbnail
```

## 2. XBM Forward and Scale-Fusion Extraction

```bash
python analysis/interpretability/run_random_model_forward_smoke.py \
  --data-path /path/to/multi_tensor.pth \
  --label-path /path/to/label_tensor.pt \
  --sample-id-path /path/to/sample_id.txt \
  --sample-id SampleID \
  --checkpoint /path/to/best_model.pt \
  --out-dir /tmp/xbm_scale_fusion \
  --clin-dim 57 \
  --class-dim 4
```

Output:

```text
/tmp/xbm_scale_fusion/SampleID_scale_fusion.pth
```

## 3. Integrated Gradients

```bash
python analysis/interpretability/run_integrated_gradients.py \
  --data-path /path/to/multi_tensor.pth \
  --label-path /path/to/label_tensor.pt \
  --sample-id-path /path/to/sample_id.txt \
  --sample-id SampleID \
  --checkpoint /path/to/best_model.pt \
  --out-dir /tmp/xbm_ig \
  --clin-dim 57 \
  --class-dim 4 \
  --n-steps 50
```

Output:

```text
/tmp/xbm_ig/SampleID_ig.pth
```

The saved object contains instance-, scale-, and feature-level attribution arrays.

## 4. IG WSI Heatmap

```bash
python analysis/interpretability/run_ig_heatmap.py \
  --ig-path /tmp/xbm_ig/SampleID_ig.pth \
  --wsi /path/to/SampleID/HE.svs \
  --h5 /path/to/SampleID/HE.h5 \
  --coord-index-npy /path/to/instance_coordinate_indices.npy \
  --out-dir /tmp/xbm_ig_heatmap \
  --patch-level 2 \
  --patch-size 512 \
  --normalize-method rank \
  --save-thumbnail
```

When using the first `N` coordinates in the H5 file:

```bash
python analysis/interpretability/run_ig_heatmap.py \
  --ig-path /tmp/xbm_ig/SampleID_ig.pth \
  --wsi /path/to/SampleID/HE.svs \
  --h5 /path/to/SampleID/HE.h5 \
  --allow-first-n-coords \
  --out-dir /tmp/xbm_ig_heatmap \
  --patch-level 2 \
  --patch-size 512
```

## 5. Cross-Attention AxG and WSI Heatmap

```bash
python analysis/interpretability/run_cross_axg.py \
  --data-path /path/to/multi_tensor.pth \
  --label-path /path/to/label_tensor.pt \
  --sample-id-path /path/to/sample_id.txt \
  --sample-id SampleID \
  --checkpoint /path/to/best_model.pt \
  --out-dir /tmp/xbm_axg \
  --clin-dim 57 \
  --class-dim 4

python analysis/interpretability/run_axg_heatmap.py \
  --axg-path /tmp/xbm_axg/SampleID_cross_axg.pth \
  --wsi /path/to/SampleID/HE.svs \
  --h5 /path/to/SampleID/HE.h5 \
  --coord-index-npy /path/to/instance_coordinate_indices.npy \
  --out-dir /tmp/xbm_axg_heatmap \
  --patch-level 2 \
  --patch-size 512 \
  --normalize-method rank \
  --save-thumbnail
```

## 6. Scale-Fusion Heatmaps

```bash
python analysis/interpretability/run_scale_fusion_heatmap.py \
  --scale-path /tmp/xbm_scale_fusion/SampleID_scale_fusion.pth \
  --wsi /path/to/SampleID/HE.svs \
  --h5 /path/to/SampleID/HE.h5 \
  --coord-index-npy /path/to/instance_coordinate_indices.npy \
  --out-dir /tmp/xbm_scale_fusion_heatmap \
  --patch-level 2 \
  --patch-size 512 \
  --normalize-method rank \
  --save-thumbnail
```

This produces one heatmap folder per scale:

```text
scale_5x_heatmap/
scale_10x_heatmap/
scale_20x_heatmap/
scale_fusion_summary.csv
scale_fusion_instance_scores.csv
```

## Complete Template

A full command template is provided at:

```text
analysis/interpretability/examples/tcga_smoke_commands.sh
```

Edit the paths at the top of the file before running.
