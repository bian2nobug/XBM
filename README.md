# XBM: Cross-scale and Cross-modal Biomarker Prediction Model

XBM is a weakly supervised computational pathology framework for patient-level molecular phenotype prediction from colorectal cancer H&E whole-slide images. The repository contains preprocessing, same-field-of-view multiscale feature construction, model training, evaluation, downstream analysis, and interpretability utilities.

## Main Features

- WSI preprocessing with tissue/tile coordinate extraction, optional downsampling, H&E color normalization, and HDF5 conversion.
- Same-field-of-view multiscale construction with 5x, 10x, and 20x branches.
- Prov-GigaPath feature extraction and 21-view pyramid representation per same-FOV instance.
- XBM model with FC-AttnPooling scale fusion, pathology sequence encoding, cross-modal clinical fusion, and classification/regression heads.
- Training, evaluation, cross-validation, and patient-level split scripts.
- Downstream morphology, genomic-association, Integrated Gradients, cross-attention AxG, and scale-fusion heatmap workflows.

## Repository Layout

```text
XBM/
├── configs/                    # Preprocessing, multiscale, training, and final task configs
├── docs/                       # Data preparation, clinical-variable, and training notes
├── examples/                   # Lightweight example CSV templates
├── scripts/                    # CLI entry points for preprocessing, training, evaluation, CV, labels, smoke test
├── xbm_preprocess/             # WSI preprocessing utilities
├── xbm_multiscale/             # Same-FOV multiscale and pyramid construction
├── xbm_training/               # Dataset, metrics, trainer, and model-loading utilities
├── model/                      # Main XBM, baselines, and added ablation/component definitions
├── analysis/
│   ├── morphology/             # Prototype-Leiden clustering and module-level GLM
│   ├── genomics/               # XBM-WGD genomic-association analysis
│   └── interpretability/       # IG, AxG, scale-fusion, and WSI heatmap scripts
├── requirements.txt
├── environment.yml
├── LICENSE
└── CITATION.cff
```

`model/XBM.py` remains the main model entry point. Additional ablation and baseline architecture definitions are placed under `model/ablations/`, `model/baselines/`, and `model/components_extra/`; see `model/MODEL_INVENTORY.md` for the model-level mapping.

The additional model files are provided as architecture definitions and component references. This public code package does not include a full end-to-end ablation-reproduction suite or result-table regeneration pipeline.

## Installation

Create a clean Python environment and install the package dependencies:

```bash
conda create -n xbm python=3.10 -y
conda activate xbm
conda install -c conda-forge openslide libstdcxx-ng libgcc-ng -y
pip install -r requirements.txt
```

Alternatively, create the environment from the provided conda file:

```bash
conda env create -f environment.yml
conda activate xbm
```

OpenSlide is required for WSI reading and heatmap rendering. The optional UMAP export in morphology analysis requires `umap-learn`, which is included in `requirements.txt`. R-based downstream scripts require `readr`, `dplyr`, `tidyr`, and `ggplot2`.

## Data Layout

The repository does not redistribute WSIs, molecular labels, clinical tables, pretrained checkpoints, or trained model weights. Prepare local files using this layout:

```text
raw_slides/
└── SampleID/
    └── HE.svs
```

After preprocessing and multiscale construction, the expected intermediate outputs are:

```text
tile_root/
└── SampleID/
    └── HE.h5

Pyramid_OUT/
└── SampleID/
    ├── pyramid.pt
    └── order.json
```

Each `pyramid.pt` should contain a pathology feature tensor shaped `(C, M, 21)`, where `C=1536` for Prov-GigaPath features, `M` is the number of same-FOV instances, and `21 = 1 + 4 + 16` corresponds to 5x, 10x, and 20x views.

## Clinical Features

The final XBM task configs use:

```yaml
clin_dim: 57
```

Clinical variables should be supplied as a numeric table with one row per sample. If `clinical_cols: null`, all numeric columns except the sample, label, and split columns are used. For strict reproducibility, provide the ordered list of 57 encoded clinical columns explicitly in the config.

See `docs/clinical_variables.md` for the expected clinical table format.

## Step 1: WSI Preprocessing

Edit `configs/preprocess.example.yaml`, then run:

```bash
python scripts/run_preprocess.py --config configs/preprocess.example.yaml
```

The preprocessing stage covers CLAM coordinate extraction, tile extraction, optional 40x-to-20x downsampling, HistomicsTK color normalization, and HDF5 conversion.

## Step 2: Same-FOV Multiscale Feature Construction

Place the Prov-GigaPath checkpoint locally, for example:

```text
model/prov-gigapath/pytorch_model.bin
```

Then edit `configs/multiscale.example.yaml` and run:

```bash
python scripts/run_multiscale.py --config configs/multiscale.example.yaml
```

This stage builds 5x/10x/20x same-FOV HDF5 files, extracts Prov-GigaPath features, and creates the 21-view pyramid representation used by XBM.

## Step 3: Training

Example training run:

```bash
python scripts/run_train.py --config configs/train.example.yaml
```

Final binary WGD task:

```bash
python scripts/run_train.py --config configs/wgd_xbm.final.yaml
```

WGD x TP53 four-class task:

```bash
python scripts/run_train.py --config configs/wgd_tp53_xbm.final.yaml
```

Evaluate a checkpoint:

```bash
python scripts/run_evaluate.py \
  --config configs/wgd_xbm.final.yaml \
  --checkpoint runs/WGD_XBM/best_model.pt \
  --split test
```

Training outputs are written to the configured `training.output_dir`:

```text
runs/WGD_XBM/
├── best_model.pt
├── last_model.pt
├── history.csv
├── predictions_val.csv
├── predictions_test.csv
└── summary.json
```

## Cross-Validation

If the label table contains a `cv_fold` column, run:

```bash
python scripts/run_cv.py \
  --config configs/wgd_xbm.final.yaml \
  --fold-col cv_fold \
  --folds 5
```

Patient-level splitting can be generated with:

```bash
python scripts/make_patient_splits.py \
  --input labels_wgd.csv \
  --output labels_wgd_with_splits.csv \
  --patient-col PatientID \
  --label-col WGD
```

## Label Preparation

The binary WGD task expects labels such as:

```text
WGD-negative
WGD-positive
```

The WGD x TP53 four-class task can be constructed with:

```bash
python scripts/build_wgd_tp53_labels.py \
  --input labels_wgd.csv \
  --output labels_wgd_tp53.csv
```

See `examples/labels_wgd.example.csv`, `examples/labels_wgd_tp53.example.csv`, and `docs/data_preparation.md` for required columns.

## Downstream Analysis

Morphologic clustering and module-level quasi-binomial GLM:

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

Genomic characterization of the XBM-WGD score:

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

## Interpretability and WSI Heatmaps

Interpretability scripts are under `analysis/interpretability/` and include:

```text
run_random_heatmap_smoke.py
run_random_model_forward_smoke.py
run_integrated_gradients.py
run_ig_heatmap.py
run_cross_axg.py
run_axg_heatmap.py
run_scale_fusion_heatmap.py
```

Example one-sample IG workflow:

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

python analysis/interpretability/run_ig_heatmap.py \
  --ig-path /tmp/xbm_ig/SampleID_ig.pth \
  --wsi /path/to/SampleID/HE.svs \
  --h5 /path/to/SampleID/HE.h5 \
  --coord-index-npy /path/to/instance_coordinate_indices.npy \
  --out-dir /tmp/xbm_ig_heatmap \
  --patch-level 2 \
  --patch-size 512
```

Use `analysis/interpretability/examples/tcga_smoke_commands.sh` as a complete command template.

## Smoke Test

A CPU-only smoke test is included for checking installation and script wiring:

```bash
python scripts/run_smoke_test.py
```

The test creates temporary toy data under `/tmp/xbm_train_smoke`, trains a small model, and evaluates the configured test split.

## Intended Use and License Notice

This repository is intended for research use in computational pathology and biomedical machine learning. It is not intended for clinical diagnosis, treatment decision-making, or deployment as a medical device.

The source code in this repository is released under the MIT License unless otherwise noted. Third-party methods, repositories, pretrained models, datasets, and software tools that inspired or supported this work are acknowledged in `THIRD_PARTY_NOTICES.md`.

Users are responsible for complying with the licenses and terms of use of any third-party software, pretrained models, datasets, or external tools used together with this repository.
