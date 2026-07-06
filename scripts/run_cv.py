#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xbm_training.config import load_config
from xbm_training.trainer import run_training


def normalize_fold_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fold-wise training from a label CSV with cv_fold.")
    parser.add_argument("--config", required=True, help="Base YAML config.")
    parser.add_argument("--fold-col", default="cv_fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--work-dir", default=None, help="Optional directory for generated fold label CSVs.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    labels_csv = Path(base_cfg["data"]["labels_csv"])
    split_col = base_cfg["data"].get("split_col", "split")
    output_root = Path(base_cfg.get("training", {}).get("output_dir", "runs/xbm_task"))
    work_dir = Path(args.work_dir) if args.work_dir else output_root / "cv_labels"
    work_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_csv)
    if args.fold_col not in labels.columns:
        raise ValueError(f"Label CSV has no fold column: {args.fold_col}")
    if split_col not in labels.columns:
        raise ValueError(f"Label CSV has no split column: {split_col}")

    summaries = []
    fold_values = labels[args.fold_col].map(normalize_fold_value)
    for fold in range(args.folds):
        fold_labels = labels.copy()
        in_development = fold_labels[split_col].astype(str).str.lower().isin({"train", "val"})
        is_val = in_development & (fold_values == str(fold))
        is_train = in_development & ~is_val
        fold_labels.loc[is_train, split_col] = "train"
        fold_labels.loc[is_val, split_col] = "val"
        fold_csv = work_dir / f"labels_fold{fold}.csv"
        fold_labels.to_csv(fold_csv, index=False)

        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["labels_csv"] = str(fold_csv)
        cfg.setdefault("training", {})["output_dir"] = str(output_root / f"fold_{fold}")
        result = run_training(cfg)
        summaries.append({"fold": fold, **result})

    summary_path = output_root / "cv_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(summary_path)


if __name__ == "__main__":
    main()
