#!/usr/bin/env python3
"""Merge XBM-WGD prediction scores with sample-level genomic metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the sample-level table for XBM-WGD genomic association analysis."
    )
    parser.add_argument("--predictions", required=True, help="Prediction CSV or torch .pt/.pth file.")
    parser.add_argument("--genomics", required=True, help="Sample-level genomics CSV.")
    parser.add_argument("--out", required=True, help="Output merged CSV.")
    parser.add_argument("--sample-col", default="SampleID", help="Sample identifier column.")
    parser.add_argument("--prediction-col", default="prediction", help="Prediction-score column in CSV.")
    parser.add_argument("--score-name", default="XBM_WGD_score", help="Output score column name.")
    parser.add_argument("--cutoff", type=float, default=0.397, help="Cutoff for XBM-WGD-positive status.")
    parser.add_argument(
        "--sample-ids",
        default=None,
        help="Optional CSV with sample IDs for torch prediction files.",
    )
    return parser.parse_args()


def _torch_load(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu")


def load_predictions(path: Path, sample_col: str, prediction_col: str, sample_ids_path: str | None) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        required = {sample_col, prediction_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Prediction file is missing required columns: {sorted(missing)}")
        return df[[sample_col, prediction_col]].copy()

    obj = _torch_load(path)
    if not isinstance(obj, dict):
        raise ValueError("Torch prediction file must contain a dictionary.")

    score_key = None
    for key in (prediction_col, "prediction", "predictions", "scores", "probabilities", "prob"):
        if key in obj:
            score_key = key
            break
    if score_key is None:
        raise ValueError("Could not find prediction scores in torch file.")

    scores = np.asarray(obj[score_key], dtype=float).reshape(-1)
    if "sample_ids" in obj:
        sample_ids = list(obj["sample_ids"])
    elif sample_ids_path:
        sid_df = pd.read_csv(sample_ids_path)
        if sample_col not in sid_df.columns:
            raise ValueError(f"--sample-ids file must contain column {sample_col}.")
        sample_ids = sid_df[sample_col].astype(str).tolist()
    else:
        raise ValueError("Torch prediction file must include sample_ids or use --sample-ids.")

    if len(sample_ids) != len(scores):
        raise ValueError(f"sample_ids length {len(sample_ids)} != scores length {len(scores)}.")

    return pd.DataFrame({sample_col: sample_ids, prediction_col: scores})


def add_score_groups(df: pd.DataFrame, sample_col: str, prediction_col: str, score_name: str, cutoff: float) -> pd.DataFrame:
    out = df.copy()
    out[score_name] = pd.to_numeric(out[prediction_col], errors="coerce")
    if out[score_name].isna().any():
        bad = out.loc[out[score_name].isna(), sample_col].head(10).tolist()
        raise ValueError(f"Non-numeric prediction scores found for samples: {bad}")

    out["XBM_WGD_pred_group"] = np.where(
        out[score_name] >= cutoff,
        "XBM-WGD-positive",
        "XBM-WGD-negative",
    )

    ranks = out[score_name].rank(method="first")
    out["XBM_WGD_quartile"] = pd.qcut(ranks, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    return out.drop(columns=[prediction_col]) if prediction_col != score_name else out


def main() -> None:
    args = parse_args()
    pred = load_predictions(Path(args.predictions), args.sample_col, args.prediction_col, args.sample_ids)
    genomics = pd.read_csv(args.genomics)
    if args.sample_col not in genomics.columns:
        raise ValueError(f"Genomics file is missing sample column {args.sample_col}.")

    pred[args.sample_col] = pred[args.sample_col].astype(str)
    genomics[args.sample_col] = genomics[args.sample_col].astype(str)

    merged = genomics.merge(pred, on=args.sample_col, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping samples between predictions and genomics table.")

    merged = add_score_groups(merged, args.sample_col, args.prediction_col, args.score_name, args.cutoff)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} merged samples to {out_path}")


if __name__ == "__main__":
    main()

