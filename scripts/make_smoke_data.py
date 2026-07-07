#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    root = Path("/tmp/xbm_train_smoke")
    pyramid_root = root / "pyramid"
    pyramid_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(8766)
    rows = []
    clinical = []
    splits = ["train", "train", "train", "train", "val", "val", "test", "test"]
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    for i, (split, label) in enumerate(zip(splits, labels)):
        sid = f"S{i + 1}"
        sample_dir = pyramid_root / sid
        sample_dir.mkdir(parents=True, exist_ok=True)
        feat = torch.tensor(rng.normal(size=(8, 6 + (i % 3), 2)), dtype=torch.float32)
        torch.save({"pyr_feat": feat}, sample_dir / "pyramid.pt")
        rows.append({"SampleID": sid, "label": label, "split": split})
        clinical.append({"SampleID": sid, "clin_1": float(i), "clin_2": float(label), "clin_3": 1.0})
    pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)
    pd.DataFrame(clinical).to_csv(root / "clinical.csv", index=False)
    print(root)


if __name__ == "__main__":
    main()
