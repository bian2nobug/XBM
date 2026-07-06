# Genomic Characterization of the XBM-WGD Score

This folder implements the downstream analysis used to characterize the XBM-WGD
score against genomic instability metrics.

The analysis includes:

1. merging XBM prediction scores with a sample-level genomics table;
2. applying the manuscript cutoff for predicted WGD status;
3. assigning XBM-WGD score quartiles;
4. Wilcoxon tests for predicted WGD-positive versus WGD-negative groups;
5. Kruskal-Wallis tests across score quartiles;
6. partial Spearman correlation adjusted for available covariates;
7. optional analyses for true genomic WGD status and TP53 mutation status.

## 1. Prepare the Score Table

```bash
python analysis/genomics/prepare_xbm_wgd_score_table.py \
  --predictions /path/to/predictions_test.csv \
  --genomics /path/to/genomics_metrics.csv \
  --out /path/to/xbm_wgd_genomics_table.csv \
  --prediction-col prediction \
  --sample-col SampleID \
  --cutoff 0.397
```

Expected prediction CSV:

```text
SampleID,prediction
TCGA-CRC-001,0.72
TCGA-CRC-002,0.18
```

Expected genomics CSV:

```text
SampleID,Ploidy,wGII,pLOH,HRD_LOH,HRD_AI,HRD_LST,HRDsum,age_at_diagnosis,sex,anatomical_site,cancer_type
```

Output columns added by the script:

```text
XBM_WGD_score
XBM_WGD_pred_group
XBM_WGD_quartile
```

## 2. Run Genomic Association Analysis

```bash
Rscript analysis/genomics/run_xbm_wgd_genomic_association.R \
  --input /path/to/xbm_wgd_genomics_table.csv \
  --out-dir /path/to/genomics_association_out \
  --score-col XBM_WGD_score \
  --pred-group-col XBM_WGD_pred_group \
  --covariates age_at_diagnosis,sex,anatomical_site,cancer_type
```

If the table uses project-specific metric names, pass them explicitly:

```bash
Rscript analysis/genomics/run_xbm_wgd_genomic_association.R \
  --input /path/to/xbm_wgd_genomics_table.csv \
  --out-dir /path/to/genomics_association_out \
  --metrics Ploidy,wGII,aneuploidy_score,pLOH,HRD_LOH,HRD_AI,HRD_LST,HRDsum
```

Optional columns:

```text
true WGD status: pass --true-wgd-col WGD_status
TP53 status:     pass --tp53-col TP53_status
```

Main outputs:

```text
predicted_group_wilcoxon_bhfdr.csv
score_quartile_kruskal_bhfdr.csv
partial_spearman_bhfdr.csv
fig7_predicted_group_boxplots.pdf
fig7_score_quartile_boxplots.pdf
```

