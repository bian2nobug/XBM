# Data Preparation

This repository does not redistribute TCGA or CPTAC whole-slide images,
clinical tables, or molecular annotation files. Users should download the
source data from the original data portals and prepare lightweight CSV metadata
tables before training.

## Cohorts

### TCGA-CRC

TCGA-CRC includes TCGA-COAD and TCGA-READ. In the manuscript, TCGA-CRC was used
as the primary cohort for model development, five-fold cross-validation,
held-out testing, baseline comparison, and downstream analyses.

Required inputs:

- H&E whole-slide images.
- Clinical variables.
- MSI/MSS status.
- TP53 mutation status.
- WGD status.

For TCGA-CRC, WGD status was derived from the sample-level `Genome Doublings`
annotation in the cBioPortal Colorectal Adenocarcinoma TCGA PanCancer Atlas
dataset (`coadread_tcga_pan_can_atlas_2018`). Samples with `Genome Doublings >=
1` are treated as WGD-positive, and samples with `Genome Doublings = 0` are
treated as WGD-negative.

### CPTAC-COAD

CPTAC-COAD was used only for external validation in the manuscript. The same
pathology preprocessing, feature extraction, and same-FOV multiscale
construction pipeline should be applied to CPTAC-COAD slides.

For CPTAC-COAD, WGD labels were taken from the CPTAC pan-cancer proteogenomic
WGD annotations reported by Chang et al. In that study, allele-specific copy
number profiles were inferred using FACETS v0.16.0, and tumors with more than
50% of the autosomal genome showing major copy number >= 2 were classified as
WGD-positive.

## Main WGD Task

The primary WGD prediction task was restricted to MSS CRC samples. This
restriction was used to reduce confounding from the distinct molecular and
histologic profile of MSI-H/dMMR CRC.

Expected label table columns:

```text
SampleID,WGD,MSI_status,split
```

Recommended WGD values:

```text
WGD-negative
WGD-positive
```

The `split` column should contain `train`, `val`, `test`, or `external`.
External CPTAC samples should use `external` when evaluated together with a
TCGA-oriented config, or `test` when evaluated with a CPTAC-only config.

## WGD x TP53 Four-Class Task

The WGD x TP53 task combines binary WGD status and TP53 mutation status into
four classes:

```text
TP53-WT_WGD-negative
TP53-MUT_WGD-negative
TP53-WT_WGD-positive
TP53-MUT_WGD-positive
```

Use `scripts/build_wgd_tp53_labels.py` to construct this column from a table
containing `WGD` and `TP53_status`.

## Patient-Level Splitting

All training, validation, and test splits should be assigned at the patient
level. If multiple WSIs are available for one patient, all slides from that
patient must remain in the same split.

Use `scripts/make_patient_splits.py` to create an 80:20 held-out split and
five cross-validation folds inside the training-development set.

## Example Files

See:

- `examples/labels_wgd.example.csv`
- `examples/labels_wgd_tp53.example.csv`
- `examples/clinical.example.csv`
- `examples/splits.example.csv`

These files show the expected format only. They are not real patient data.

