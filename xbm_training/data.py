from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def normalize_sample_id(value: object) -> str:
    return str(value).strip()


def _find_feature_file(root: Path, sample_id: str, file_name: str) -> Path:
    candidates = [
        root / sample_id / file_name,
        root / f"{sample_id}.pt",
        root / f"{sample_id}.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No feature file found for sample_id={sample_id} under {root}")


def _load_pyramid_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("pyr_feat", "features", "feat", "x"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise KeyError(f"{path} is a dict but has no pyr_feat/features/feat/x key")
    tensor = torch.as_tensor(obj, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.transpose(0, 1).unsqueeze(-1)
    if tensor.ndim != 3:
        raise ValueError(f"Expected feature tensor (C,M,N), got shape={tuple(tensor.shape)} from {path}")
    return tensor.contiguous()


def _pad_or_crop_tiles(x: torch.Tensor, max_tiles: int, random_crop: bool = False) -> torch.Tensor:
    c, m, n = x.shape
    if m > max_tiles:
        if random_crop:
            idx = torch.randperm(m)[:max_tiles].sort().values
            return x[:, idx, :]
        return x[:, :max_tiles, :]
    if m < max_tiles:
        pad = torch.zeros(c, max_tiles - m, n, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)
    return x


def _append_clinical(x: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
    if clinical.numel() == 0:
        return x
    _, m, n = x.shape
    clin = clinical.to(dtype=x.dtype).view(-1, 1, 1).expand(-1, m, n)
    return torch.cat([x, clin], dim=0)


@dataclass
class SampleRecord:
    sample_id: str
    feature_path: Path
    label: float | int
    split: str


class PyramidDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        clinical: Dict[str, torch.Tensor],
        max_tiles: int,
        feature_dim: Optional[int],
        clin_dim: int,
        random_crop: bool = False,
    ):
        self.records = list(records)
        self.clinical = clinical
        self.max_tiles = int(max_tiles)
        self.feature_dim = feature_dim
        self.clin_dim = int(clin_dim)
        self.random_crop = bool(random_crop)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        x = _load_pyramid_tensor(rec.feature_path)
        if self.feature_dim is not None and x.shape[0] != int(self.feature_dim):
            raise ValueError(
                f"{rec.sample_id}: expected feature_dim={self.feature_dim}, got {x.shape[0]}"
            )
        x = _pad_or_crop_tiles(x, self.max_tiles, random_crop=self.random_crop)
        clinical = self.clinical.get(rec.sample_id)
        if clinical is None:
            clinical = torch.zeros(self.clin_dim, dtype=torch.float32)
        if clinical.numel() != self.clin_dim:
            raise ValueError(
                f"{rec.sample_id}: expected clin_dim={self.clin_dim}, got {clinical.numel()}"
            )
        x = _append_clinical(x, clinical)
        y = torch.tensor(rec.label)
        return x, y, rec.sample_id


def _encode_labels(
    labels: pd.Series,
    task_type: str,
    positive_label: Optional[str] = None,
) -> Tuple[pd.Series, Dict[str, int]]:
    if task_type == "regression":
        return labels.astype(float), {}

    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.notna().all():
        values = numeric.astype(int if task_type == "multiclass" else float)
        return values, {}

    raw = labels.astype(str).str.strip()
    classes = sorted(raw.unique().tolist())
    if task_type == "binary":
        if len(classes) != 2:
            raise ValueError(f"Binary task requires 2 classes, got {classes}")
        if positive_label is None:
            positive_label = classes[-1]
        encoded = raw.map(lambda v: 1.0 if v == positive_label else 0.0)
        return encoded, {classes[0]: 0, classes[1]: 1}

    mapping = {name: idx for idx, name in enumerate(classes)}
    return raw.map(mapping).astype(int), mapping


def _load_clinical(data_cfg: dict, sample_ids: Iterable[str]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    clinical_csv = data_cfg.get("clinical_csv")
    clin_dim = int(data_cfg.get("clin_dim", 0) or 0)
    if not clinical_csv:
        return {}, []

    sample_col = data_cfg.get("sample_col", "SampleID")
    df = pd.read_csv(clinical_csv)
    if sample_col not in df.columns:
        raise ValueError(f"clinical_csv has no sample column: {sample_col}")
    df[sample_col] = df[sample_col].map(normalize_sample_id)

    clinical_cols = data_cfg.get("clinical_cols")
    if clinical_cols is None:
        excluded = {sample_col, data_cfg.get("label_col", "label"), data_cfg.get("split_col", "split")}
        clinical_cols = [
            col for col in df.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
        ]
    clinical_cols = list(clinical_cols)
    if len(clinical_cols) != clin_dim:
        raise ValueError(
            f"clinical_cols length ({len(clinical_cols)}) must match data.clin_dim ({clin_dim})"
        )

    keep = df[[sample_col] + clinical_cols].copy()
    keep[clinical_cols] = keep[clinical_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    sample_set = {normalize_sample_id(x) for x in sample_ids}
    clinical = {
        row[sample_col]: torch.tensor(row[clinical_cols].to_numpy(dtype=np.float32))
        for _, row in keep.iterrows()
        if row[sample_col] in sample_set
    }
    return clinical, clinical_cols


def _assign_splits(df: pd.DataFrame, data_cfg: dict, task_type: str) -> pd.DataFrame:
    split_col = data_cfg.get("split_col")
    if split_col and split_col in df.columns:
        df["split"] = df[split_col].astype(str).str.lower().str.strip()
        return df

    test_size = float(data_cfg.get("test_size", 0.2))
    val_size = float(data_cfg.get("val_size", 0.2))
    seed = int(data_cfg.get("seed", 8766))
    stratify = df["label_encoded"] if task_type != "regression" and df["label_encoded"].nunique() > 1 else None

    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    stratify_train = (
        train_val["label_encoded"]
        if task_type != "regression" and train_val["label_encoded"].nunique() > 1
        else None
    )
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed, stratify=stratify_train)
    out = pd.concat([
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test"),
    ], axis=0)
    return out


def build_datasets(cfg: dict) -> Tuple[Dict[str, PyramidDataset], dict]:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    task_type = train_cfg.get("task_type", "binary")

    pyramid_root = Path(data_cfg["pyramid_root"])
    labels_csv = data_cfg["labels_csv"]
    sample_col = data_cfg.get("sample_col", "SampleID")
    label_col = data_cfg.get("label_col", "label")
    file_name = data_cfg.get("feature_file", "pyramid.pt")

    labels_df = pd.read_csv(labels_csv)
    if sample_col not in labels_df.columns or label_col not in labels_df.columns:
        raise ValueError(f"labels_csv must contain {sample_col!r} and {label_col!r}")
    labels_df = labels_df[[c for c in labels_df.columns]].copy()
    labels_df[sample_col] = labels_df[sample_col].map(normalize_sample_id)
    labels_df = labels_df.dropna(subset=[sample_col, label_col])
    encoded, label_mapping = _encode_labels(labels_df[label_col], task_type, data_cfg.get("positive_label"))
    labels_df["label_encoded"] = encoded
    labels_df = _assign_splits(labels_df, data_cfg, task_type)

    records: List[SampleRecord] = []
    missing = []
    for _, row in labels_df.iterrows():
        sid = row[sample_col]
        try:
            path = _find_feature_file(pyramid_root, sid, file_name)
        except FileNotFoundError:
            missing.append(sid)
            continue
        records.append(SampleRecord(sid, path, row["label_encoded"], row["split"]))
    if not records:
        raise ValueError("No labeled samples with feature files were found")

    clinical, clinical_cols = _load_clinical(data_cfg, [r.sample_id for r in records])
    max_tiles = int(data_cfg.get("max_tiles", 1500))
    feature_dim = data_cfg.get("feature_dim", 1536)
    feature_dim = None if feature_dim in (None, "auto") else int(feature_dim)
    clin_dim = int(data_cfg.get("clin_dim", 0) or 0)

    datasets: Dict[str, PyramidDataset] = {}
    for split in ("train", "val", "test", "external"):
        split_records = [r for r in records if r.split == split]
        if not split_records:
            continue
        datasets[split] = PyramidDataset(
            split_records,
            clinical=clinical,
            max_tiles=max_tiles,
            feature_dim=feature_dim,
            clin_dim=clin_dim,
            random_crop=bool(data_cfg.get("random_crop_train", False) and split == "train"),
        )

    meta = {
        "label_mapping": label_mapping,
        "clinical_cols": clinical_cols,
        "missing_feature_samples": missing,
        "num_records": len(records),
        "splits": {name: len(ds) for name, ds in datasets.items()},
    }
    return datasets, meta
