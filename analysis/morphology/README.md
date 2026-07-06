# Morphologic Clustering and Module Analysis

This folder implements the downstream analyses described in Supplementary
Method 3:

1. tile-level feature standardization and PCA;
2. MiniBatch K-means prototype extraction;
3. KNN graph construction on prototypes;
4. Leiden clustering of prototypes;
5. nearest-prototype assignment of all valid tiles;
6. sample-level cluster composition;
7. tile-count-weighted quasi-binomial GLM for morphologic modules.

Default parameters match the manuscript analysis:

```text
PCA dimensions: 20
prototype number: 1000
prototype KNN: 15
Leiden resolution: 0.8
random seed: 42
representative tiles per cluster: 30
```

## 1. Run Prototype-Leiden Clustering

```bash
python analysis/morphology/run_prototype_leiden.py \
  --features /path/to/tile_features.pt \
  --out-dir /path/to/morphology_cluster_out \
  --h5-root /path/to/multiscale_h5_root
```

Expected feature file:

```text
torch .pt/.pth dictionary with:
  feat_histo_NDT or features: tensor shaped (N, D, T) or (N, T, D)
  sample_ids: sample identifiers, length N
```

If `--h5-root` is provided, valid tile counts are read from
`SampleID/HE_noskip.h5`. If it is not provided, valid tiles are inferred from
non-zero feature vectors.

Main outputs:

```text
tile_cluster_composition_by_sample.csv
tile_cluster_labels_NT.npy
tile_cluster_labels_NT.pth
representative_tiles_by_cluster.csv
tile_cluster_size.csv
prototype_leiden_labels.csv
run_parameters.json
```

## 2. Run Module-Level Quasi-Binomial GLM

```bash
Rscript analysis/morphology/run_module_quasibinomial_glm.R \
  --composition /path/to/tile_cluster_composition_by_sample.csv \
  --labels /path/to/wgd_labels.csv \
  --out-dir /path/to/module_glm_out \
  --sample-col SampleID \
  --wgd-col WGD
```

Default module definitions:

```text
Tumor_region       = C7 + C8 + C9 + C5 + C0
Tumor_gland        = C7 + C8 + C9
Invasive_front     = C5
Solid_tumor        = C0
TME_hub            = C3
Normal_background  = C1 + C2 + C4 + C6
Invasion_TME       = C3 + C5 + C0
```

To override them, pass a CSV with columns `Module` and `Clusters`:

```csv
Module,Clusters
Tumor_region,C7;C8;C9;C5;C0
TME_hub,C3
```

Main outputs:

```text
cluster_wilcoxon_tests.csv
module_quasibinomial_glm.csv
module_prop_long.csv
module_count_long.csv
module_forest_plot.pdf
module_prop_boxplot.pdf
```

