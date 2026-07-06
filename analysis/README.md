# Downstream Analysis

This folder contains downstream analysis code that corresponds to the manuscript
sections on:

1. tile-level morphologic clustering and module-level quasi-binomial analysis;
2. genomic characterization of the XBM-WGD score;
3. interpretability and WSI heatmap workflows.

The scripts are cleaned, parameterized versions of the analysis code used for
the manuscript. Local absolute paths, intermediate plotting experiments,
survival/KM/Cox screening scripts, and unrelated model-comparison scripts are
not included.

## Folder Layout

```text
analysis/
├── morphology/
│   ├── run_prototype_leiden.py
│   ├── run_module_quasibinomial_glm.R
│   └── README.md
├── genomics/
│   ├── prepare_xbm_wgd_score_table.py
│   ├── run_xbm_wgd_genomic_association.R
│   └── README.md
└── interpretability/
    ├── run_integrated_gradients.py
    ├── run_cross_axg.py
    ├── run_scale_fusion_heatmap.py
    └── README.md
```

## Python Dependencies

The morphology clustering script requires the core repository dependencies plus:

```bash
pip install python-igraph leidenalg joblib
```

Optional UMAP output requires:

```bash
pip install umap-learn
```

## R Dependencies

The R scripts use:

```r
install.packages(c("readr", "dplyr", "tidyr", "ggplot2"))
```


## Interpretability Notes

The interpretability folder includes scripts for WSI heatmaps, Integrated
Gradients, cross-attention AxG, and scale-fusion heatmaps. These scripts require
OpenSlide for WSI rendering.
