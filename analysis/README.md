# Downstream Analysis

This folder contains downstream analysis utilities for XBM.

```text
analysis/
├── morphology/         # Prototype-Leiden clustering and module-level GLM
├── genomics/           # XBM-WGD genomic-association analysis
└── interpretability/   # IG, AxG, scale-fusion, and WSI heatmap workflows
```

## Morphology

```bash
python analysis/morphology/run_prototype_leiden.py \
  --features /path/to/tile_features.pt \
  --out-dir /path/to/morphology_cluster_out \
  --h5-root /path/to/multiscale_h5_root

Rscript analysis/morphology/run_module_quasibinomial_glm.R \
  --composition /path/to/morphology_cluster_out/tile_cluster_composition_by_sample.csv \
  --labels /path/to/wgd_labels.csv \
  --out-dir /path/to/module_glm_out \
  --sample-col SampleID \
  --wgd-col WGD
```

## Genomics

```bash
python analysis/genomics/prepare_xbm_wgd_score_table.py \
  --predictions runs/WGD_XBM/predictions_test.csv \
  --genomics /path/to/genomics_metrics.csv \
  --out /path/to/xbm_wgd_genomics_table.csv \
  --prediction-col prediction

Rscript analysis/genomics/run_xbm_wgd_genomic_association.R \
  --input /path/to/xbm_wgd_genomics_table.csv \
  --out-dir /path/to/genomics_association_out
```

## Interpretability

The interpretability folder includes WSI heatmap rendering, Integrated Gradients, cross-attention AxG, and 5x/10x/20x scale-fusion heatmaps. See `analysis/interpretability/README.md` for full commands.

## R Dependencies

```r
install.packages(c("readr", "dplyr", "tidyr", "ggplot2"))
```
