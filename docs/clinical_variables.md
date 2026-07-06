# Clinical Variables

XBM supports structured clinical variables as numeric inputs concatenated to
the pathology feature tensor. The manuscript used 15 source clinical variables,
which were harmonized across TCGA-CRC and CPTAC-COAD and then converted into a
numeric one-hot encoded table.

## Input Format

The training code expects a CSV file with one row per sample:

```text
SampleID,<numeric clinical feature 1>,<numeric clinical feature 2>,...
```

All clinical feature columns used by the model must be numeric. Categorical
variables should be converted to one-hot columns before training. Missing values
should be imputed before model input; the released data loader fills remaining
missing numeric values with 0.0 as a final guard.

## Harmonization Across Cohorts

To keep TCGA and CPTAC inputs consistent:

1. Define the one-hot encoding template from the TCGA-CRC training cohort.
2. Apply the same column order to CPTAC-COAD.
3. Add missing one-hot columns to CPTAC-COAD with value 0.
4. Drop CPTAC-only categories that are absent from the TCGA template unless a
   predefined harmonization rule maps them to an existing category.
5. Use the same `clinical_cols` list in all training and external validation
   configs.

## Clinical Dimension

The default final configs use:

```yaml
clin_dim: 41
clinical_cols: null
```

When `clinical_cols` is `null`, all numeric columns except the sample, label,
and split columns are used. For strict reproducibility, it is recommended to
replace `clinical_cols: null` with the exact ordered list of the 41 encoded
clinical columns used in the manuscript.

## Pathology-Only and Clinical-Only Settings

For pathology-only experiments, provide no `clinical_csv` and set:

```yaml
clin_dim: 0
```

For clinical-only experiments, use a clinical-only model or a separate baseline
script. The XBM architecture expects pathology tokens and is not intended to be
used as a pure tabular classifier without modification.

