# XBM Interpretability and WSI Heatmap Analysis

This folder contains XBM interpretability utilities integrated with the main
repository. The scripts cover WSI heatmap generation and attribution workflows
for model inspection:

1. WSI heatmap generation;
2. FC-AttnPooling scale-fusion extraction and 5x/10x/20x heatmaps;
3. Integrated Gradients instance-, scale-, and feature-level attribution;
4. Integrated Gradients WSI heatmaps;
5. cross-attention Attention x Gradient (AxG) attribution and WSI heatmaps.

The scripts are command-line based and are intended to be run from the repository
root.

## Folder layout

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

`shared/model_lib/` is kept as an interpretability-compatible model adapter so
that the gradient and scale-fusion hooks remain reproducible. The main training
model remains `model/XBM.py`.

## System dependencies

WSI heatmap generation requires OpenSlide. A conda-based installation is usually
more robust on servers:

```bash
conda install -c conda-forge openslide libstdcxx-ng libgcc-ng
```

Python dependencies are listed in the repository-level `requirements.txt`.

## Coordinate convention

For the tested TCGA pathomics H5 files, the coordinate key was `locations` and
coordinates were in 20x space. The corresponding heatmap parameters were:

```bash
--patch-level 2
--patch-size 512
```

Other cohorts may use a different coordinate level. Check the H5 coordinate
range against the WSI level-0 dimensions before generating heatmaps.

## 1. WSI heatmap smoke test

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

## 2. XBM forward and scale-fusion extraction

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

## 4. IG WSI heatmap

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

## 5. Cross-attention AxG and WSI heatmap

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

## 6. Scale-fusion heatmaps

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

This produces three heatmap folders, one each for 5x, 10x, and 20x scale
contribution, plus `scale_fusion_summary.csv` and
`scale_fusion_instance_scores.csv`.
