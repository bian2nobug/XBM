# Training Protocol

This document summarizes the training protocol used by the manuscript and how
it maps to this repository.

## Main Binary WGD Task

1. Restrict the primary cohort to MSS CRC samples with available WGD labels.
2. Split TCGA-CRC at the patient level into an 80% training-development set and
   a 20% held-out test set.
3. Use five-fold cross-validation inside the training-development set for model
   selection and hyperparameter optimization.
4. Reserve the held-out TCGA test set and CPTAC-COAD external cohort for final
   evaluation only.

The final binary WGD config is:

```bash
python scripts/run_train.py --config configs/wgd_xbm.final.yaml
```

## WGD x TP53 Four-Class Task

Construct a four-class label from WGD and TP53 mutation status:

```bash
python scripts/build_wgd_tp53_labels.py \
  --input labels_wgd.csv \
  --output labels_wgd_tp53.csv
```

Train with:

```bash
python scripts/run_train.py --config configs/wgd_tp53_xbm.final.yaml
```

For multiclass tasks, this repository supports class-weighted cross-entropy and
label smoothing through:

```yaml
training:
  class_weights: auto
  label_smoothing: 0.05
```

## Cross-Validation

If a label table contains a `cv_fold` column, five-fold training can be launched
with:

```bash
python scripts/run_cv.py \
  --config configs/wgd_xbm.final.yaml \
  --fold-col cv_fold \
  --folds 5
```

For each fold, samples with the current fold index are assigned to validation,
and the remaining training-development samples are assigned to training. Samples
with `split=test` or `split=external` are not used for fold model selection.

## External Validation

Evaluate CPTAC-COAD using a trained checkpoint:

```bash
python scripts/run_evaluate.py \
  --config configs/wgd_xbm.external_cptac.yaml \
  --checkpoint runs/WGD_XBM/best_model.pt \
  --split external
```

If the CPTAC label CSV contains only CPTAC samples, the split can also be named
`test`; adjust the command accordingly.

## Hyperparameter Optimization

The manuscript used Tree-structured Parzen Estimator Bayesian optimization with
trial pruning. The `optuna` package is included in `requirements.txt`. The
current repository provides the final configuration files and the training
entry points; full hyperparameter search spaces can be added as separate Optuna
study scripts if needed.

## Downstream Analysis Scripts

Integrated Gradients, Leiden morphologic clustering, quasi-binomial GLM module
analysis, and genomic instability analyses are downstream analysis components.
They are provided under the `analysis/` directory in this repository.

