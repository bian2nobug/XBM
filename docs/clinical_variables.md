# Clinical Variables

XBM supports structured clinical variables as numeric inputs concatenated to the pathology feature tensor. In the released final configuration, the clinical table is represented as a 57-dimensional numeric vector for each sample.

## Input Format

The clinical CSV should contain one row per sample:

```text
SampleID,<numeric clinical feature 1>,<numeric clinical feature 2>,...
```

All columns used by the model must be numeric. Categorical variables should be one-hot encoded before training. Missing values should be imputed before model input; the released data loader fills any remaining missing numeric values with `0.0` as a final guard.

## Cross-Cohort Harmonization

Use the same clinical encoding template across cohorts:

1. Define the one-hot template from the development cohort.
2. Apply the same column order to external cohorts.
3. Add missing one-hot columns with value `0`.
4. Drop or map cohort-specific categories before model input.
5. Use the same ordered `clinical_cols` list in training and external-validation configs.

## Clinical Dimension

The final XBM configs use:

```yaml
clin_dim: 57
clinical_cols: null
```

When `clinical_cols` is `null`, all numeric columns except the sample, label, and split columns are used. For strict reproducibility, replace `clinical_cols: null` with the exact ordered list of 57 encoded clinical columns used to build the feature tensor.

## Pathology-Only and Clinical-Only Settings

For pathology-only XBM experiments, set `clin_dim: 0` and omit `clinical_csv`. The current `model/XBM.py` supports this setting by using the slide token as the cross-attention query.

For clinical-only experiments, use a separate tabular baseline model. XBM is designed for pathology-token input and is not intended to serve as a pure tabular classifier.
