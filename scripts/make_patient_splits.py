#!/usr/bin/env python
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level holdout and CV splits.")
    parser.add_argument("--input", required=True, help="Input sample-level CSV.")
    parser.add_argument("--output", required=True, help="Output CSV with split and cv_fold columns.")
    parser.add_argument("--sample-col", default="SampleID")
    parser.add_argument("--patient-col", default="PatientID")
    parser.add_argument("--label-col", default="WGD")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=8766)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    for col in (args.sample_col, args.patient_col, args.label_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    patients = df[[args.patient_col, args.label_col]].drop_duplicates(args.patient_col)
    stratify = patients[args.label_col] if patients[args.label_col].nunique() > 1 else None
    train_patients, test_patients = train_test_split(
        patients,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    df["split"] = np.where(df[args.patient_col].isin(test_patients[args.patient_col]), "test", "train")
    df["cv_fold"] = ""

    cv_patients = train_patients.reset_index(drop=True)
    cv_stratify = cv_patients[args.label_col] if cv_patients[args.label_col].nunique() > 1 else None
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    y = cv_stratify if cv_stratify is not None else np.zeros(len(cv_patients), dtype=int)
    for fold, (_, val_idx) in enumerate(splitter.split(cv_patients, y)):
        fold_patients = set(cv_patients.loc[val_idx, args.patient_col])
        df.loc[df[args.patient_col].isin(fold_patients), "cv_fold"] = fold

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

