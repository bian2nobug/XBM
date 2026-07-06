#!/usr/bin/env python
from __future__ import annotations

import argparse

import pandas as pd


def normalize_wgd(value: object) -> str:
    text = str(value).strip()
    lower = text.lower()
    if lower in {"1", "true", "positive", "wgd-positive", "wgd_positive", "wgd+"}:
        return "WGD-positive"
    if lower in {"0", "false", "negative", "wgd-negative", "wgd_negative", "wgd-"}:
        return "WGD-negative"
    if text in {"WGD-positive", "WGD-negative"}:
        return text
    raise ValueError(f"Unsupported WGD value: {value!r}")


def normalize_tp53(value: object) -> str:
    text = str(value).strip()
    lower = text.lower()
    if lower in {"mut", "mutant", "mutation", "tp53-mut", "tp53_mut", "1", "true"}:
        return "TP53-MUT"
    if lower in {"wt", "wildtype", "wild-type", "tp53-wt", "tp53_wt", "0", "false"}:
        return "TP53-WT"
    if text in {"TP53-MUT", "TP53-WT"}:
        return text
    raise ValueError(f"Unsupported TP53 value: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WGD x TP53 four-class labels.")
    parser.add_argument("--input", required=True, help="Input CSV containing WGD and TP53 columns.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--wgd-col", default="WGD", help="WGD status column.")
    parser.add_argument("--tp53-col", default="TP53_status", help="TP53 status column.")
    parser.add_argument("--out-col", default="WGD_TP53", help="Output four-class label column.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.wgd_col not in df.columns:
        raise ValueError(f"Missing WGD column: {args.wgd_col}")
    if args.tp53_col not in df.columns:
        raise ValueError(f"Missing TP53 column: {args.tp53_col}")

    wgd = df[args.wgd_col].map(normalize_wgd)
    tp53 = df[args.tp53_col].map(normalize_tp53)
    df[args.wgd_col] = wgd
    df[args.tp53_col] = tp53
    df[args.out_col] = tp53 + "_" + wgd
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

