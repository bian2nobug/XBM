# XBM: Cross-scale and Cross-modal Biomarker Prediction Model

This repository provides the core code for XBM, a weakly supervised computational
pathology framework for patient-level molecular phenotype prediction from
colorectal cancer H&E whole-slide images.

The current release includes:

1. WSI preprocessing and tile HDF5 construction.
2. Same-field-of-view multiscale construction.
3. Prov-GigaPath feature extraction for 5x, 10x, and 20x image branches.
4. Same-FOV pyramid feature construction with 21 views.
5. XBM model and baseline model definitions.
6. Downstream training, evaluation, patient-level split, and cross-validation scripts.
7. Morphologic clustering, module-level GLM, and genomic characterization scripts.
8. Data-format documentation and lightweight example CSV files.
9. Interpretability and WSI heatmap scripts for IG, cross-attention AxG, and scale-fusion maps.

## Repository Structure

```text
XBM/
├── README.md
├── requirements.txt
├── configs/
│   ├── preprocess.example.yaml
│   ├── multiscale.example.yaml
│   ├── train.example.yaml
│   ├── wgd_xbm.final.yaml
│   ├── wgd_tp53_xbm.final.yaml
│   └── wgd_xbm.external_cptac.yaml
├── docs/
│   ├── data_preparation.md
│   ├── clinical_variables.md
│   └── training_protocol.md
├── examples/
│   ├── labels_wgd.example.csv
│   ├── labels_wgd_tp53.example.csv
│   ├── clinical.example.csv
│   └── splits.example.csv
├── scripts/
│   ├── run_preprocess.py
│   ├── run_multiscale.py
│   ├── run_train.py
│   ├── run_evaluate.py
│   ├── run_cv.py
│   ├── make_patient_splits.py
│   └── build_wgd_tp53_labels.py
├── analysis/
│   ├── morphology/
│   │   ├── run_prototype_leiden.py
│   │   └── run_module_quasibinomial_glm.R
│   ├── genomics/
│   │   ├── prepare_xbm_wgd_score_table.py
│   │   └── run_xbm_wgd_genomic_association.R
│   └── interpretability/
│       ├── run_integrated_gradients.py
│       ├── run_cross_axg.py
│       └── run_scale_fusion_heatmap.py
├── xbm_preprocess/
├── xbm_multiscale/
├── xbm_training/
└── model/
    ├── XBM.py
    ├── multi_attnmil.py
    ├── multi_transmil_multiscale_multimodal.py
    ├── prov-gigapath/
    │   └── .gitkeep
    └── submodel/
```

## Step 1: WSI Preprocessing

The preprocessing step uses CLAM tissue segmentation and tile-coordinate
extraction, followed by tile extraction, optional 40x-to-20x downsampling,
HistomicsTK color normalization, and HDF5 conversion.

Expected input layout:

```text
raw_slides/
└── SampleID/
    └── HE.svs
```

Run:

```bash
python scripts/run_preprocess.py --config configs/preprocess.example.yaml
```

The final preprocessing HDF5 files contain:

```text
HE_images
locations
```

## Step 2: Multiscale Feature Construction

The multiscale step takes preprocessed HDF5 tile files and builds the
same-field-of-view multiscale feature representation used by XBM.

Run:

```bash
python scripts/run_multiscale.py --config configs/multiscale.example.yaml
```

This step performs:

```text
HE.h5
  -> HE_noskip.h5 with HE_images_5x / HE_images_10x / HE_images_20x
  -> Prov-GigaPath features in M5x_2 / M10x_2 / M20x_2
  -> pyramid.pt with pyr_feat and pyr_mask
```

The final same-FOV pyramid tensor has shape:

```text
(C, M, 21)
```

where `C=1536` for Prov-GigaPath, `M` is the number of 5x anchor fields, and
`21 = 1 + 4 + 16` corresponds to one 5x view, four 10x views, and sixteen 20x
views.

## Prov-GigaPath Checkpoint

The Prov-GigaPath pretrained checkpoint is not included in this repository.
Download it separately and place it here:

```text
model/prov-gigapath/pytorch_model.bin
```

The default multiscale config points to:

```yaml
paths:
  provgigapath_checkpoint: model/prov-gigapath/pytorch_model.bin
```

Do not commit the checkpoint file to GitHub.

## Outputs

Preprocessing outputs:

```text
tile_root/
└── SampleID/
    └── HE.h5
```

Multiscale HDF5 outputs:

```text
multiscale_h5_root/
└── SampleID/
    └── HE_noskip.h5
```

Prov-GigaPath feature outputs:

```text
M5x_2/SampleID/HE_images_5x_prov_gigapath_feature.npy
M10x_2/SampleID/HE_images_10x_prov_gigapath_feature.npy
M20x_2/SampleID/HE_images_20x_prov_gigapath_feature.npy
```

Pyramid outputs:

```text
Pyramid_OUT/
└── SampleID/
    ├── pyramid.pt
    └── order.json
```

`pyramid.pt` contains:

```text
pyr_feat
pyr_mask
order
sample_id
```

## Model Code

The model folder contains:

```text
XBM.py
multi_attnmil.py
multi_transmil_multiscale_multimodal.py
submodel/
```

`XBM.py` is the main XBM model. `multi_attnmil.py` and
`multi_transmil_multiscale_multimodal.py` provide baseline model definitions.

## Step 3: Downstream Training and Evaluation

The training step takes `pyramid.pt` files from Step 2, joins them with task
labels and optional numeric clinical features, and trains a patient-level
prediction model.

Example training run:

```bash
python scripts/run_train.py --config configs/train.example.yaml
```

Final WGD training config:

```bash
python scripts/run_train.py --config configs/wgd_xbm.final.yaml
```

WGD x TP53 four-class config:

```bash
python scripts/run_train.py --config configs/wgd_tp53_xbm.final.yaml
```

Evaluate an existing checkpoint:

```bash
python scripts/run_evaluate.py \
  --config configs/train.example.yaml \
  --checkpoint runs/WGD_XBM/best_model.pt \
  --split test
```

The label table should contain a sample ID column, a target label column, and a
split column such as `train`, `val`, `test`, or `external`.

See `docs/data_preparation.md` and `examples/` for the expected data format.

Training outputs:

```text
runs/WGD_XBM/
├── best_model.pt
├── last_model.pt
├── history.csv
├── predictions_val.csv
├── predictions_test.csv
└── summary.json
```

## Data and Label Preparation

TCGA-CRC and CPTAC-COAD data are not redistributed in this repository. Users
should prepare local metadata tables and slide-derived feature files.

For TCGA-CRC, WGD labels should be derived from the sample-level `Genome
Doublings` annotation in the cBioPortal Colorectal Adenocarcinoma TCGA
PanCancer Atlas dataset. Samples with `Genome Doublings >= 1` are WGD-positive;
samples with `Genome Doublings = 0` are WGD-negative.

For CPTAC-COAD, the manuscript used the CPTAC pan-cancer proteogenomic WGD
annotations reported by Chang et al. CPTAC-COAD is intended for external
validation and should not be used for model selection.

The primary binary WGD task is restricted to MSS CRC samples. The WGD x TP53
task can be constructed with:

```bash
python scripts/build_wgd_tp53_labels.py \
  --input labels_wgd.csv \
  --output labels_wgd_tp53.csv
```

Patient-level train/test and cross-validation split columns can be generated
with:

```bash
python scripts/make_patient_splits.py \
  --input labels_wgd.csv \
  --output labels_wgd_with_splits.csv \
  --patient-col PatientID \
  --label-col WGD
```

## Cross-Validation

If the label table includes a `cv_fold` column, run five-fold training with:

```bash
python scripts/run_cv.py \
  --config configs/wgd_xbm.final.yaml \
  --fold-col cv_fold \
  --folds 5
```

The held-out TCGA test set and CPTAC external samples should remain outside the
fold-level model-selection process.

## Multiclass Training

For WGD x TP53 and other multiclass tasks, the trainer supports class-weighted
cross-entropy and label smoothing:

```yaml
training:
  task_type: multiclass
  class_weights: auto
  label_smoothing: 0.05
```

Multiclass evaluation reports both macro- and micro-AUROC.

## Downstream Analyses

The released downstream analysis code is under `analysis/`.

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

See `analysis/README.md` and the subfolder README files for required columns,
default manuscript parameters, and optional arguments.

## Interpretability and WSI Heatmaps

Interpretability scripts are provided under `analysis/interpretability/`. The
pipeline covers WSI heatmap generation, model forward with scale-fusion
extraction, Integrated Gradients, IG heatmaps, cross-attention Attention x
Gradient (AxG), AxG heatmaps, and 5x/10x/20x scale-fusion heatmaps.

Example WSI heatmap smoke test:

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

Example one-sample IG and heatmap workflow:

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

For TCGA-style pathomics H5 files tested in this project, `locations` were 20x
coordinates, so `--patch-level 2 --patch-size 512` was used.

OpenSlide must be available for WSI rendering. On servers, install the system
component with:

```bash
conda install -c conda-forge openslide libstdcxx-ng libgcc-ng
```

See `analysis/interpretability/README.md` for the complete interpretability
and heatmap commands.

## Repository Notes

Do not commit:

```text
raw WSIs
intermediate .npy files
HDF5 tile files
pyramid tensors
model checkpoints
training runs
paper PDFs
manuscript files
```

Commit only source code, configs, README files, and lightweight placeholders such
as `model/prov-gigapath/.gitkeep`.
