"""Build same-FOV 21-view pyramid features from 5x/10x/20x features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch


def infer_step_1d(coords_1d: np.ndarray) -> int:
    values = np.unique(coords_1d)
    if values.size <= 1:
        return 256
    diffs = np.diff(np.sort(values))
    step = int(np.median(diffs)) if diffs.size else 256
    return max(step, 1)


def to_int_xy(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"coordinates must be (N,2), got {arr.shape}")
    return np.rint(arr).astype(np.int64) if not np.issubdtype(arr.dtype, np.integer) else arr.astype(np.int64)


def ensure_feature_nc(feature: np.ndarray) -> np.ndarray:
    """Convert feature matrices to (N, C)."""
    arr = np.asarray(feature)
    if arr.ndim != 2:
        raise ValueError(f"feature must be 2D, got {arr.shape}")
    n, c = arr.shape
    if c in {768, 1024, 1536, 2048, 3072}:
        return arr
    if n in {768, 1024, 1536, 2048, 3072}:
        return arr.T
    return arr if c > n else arr.T


def load_feature_matrix(path: str) -> np.ndarray:
    """Load .npy, .pt, or .pth feature matrix and return a numpy array."""
    feature_path = Path(path)
    if feature_path.suffix.lower() == ".npy":
        obj = np.load(feature_path)
    elif feature_path.suffix.lower() in {".pt", ".pth"}:
        obj = torch.load(feature_path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
        elif isinstance(obj, dict):
            tensor = None
            for key in ["features", "feature", "feat", "embeddings", "tensor"]:
                if key in obj and isinstance(obj[key], torch.Tensor):
                    tensor = obj[key]
                    break
            if tensor is None:
                tensor = next((v for v in obj.values() if isinstance(v, torch.Tensor) and v.ndim == 2), None)
            if tensor is None:
                raise ValueError(f"No 2D tensor found in {path}")
            obj = tensor.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported torch object type in {path}: {type(obj)}")
    else:
        raise ValueError(f"Unsupported feature file extension: {feature_path.suffix}")
    return ensure_feature_nc(obj)


def build_index(coords: np.ndarray) -> Dict[Tuple[int, int], int]:
    mapping = {}
    for idx, (x, y) in enumerate(coords):
        key = (int(x), int(y))
        if key not in mapping:
            mapping[key] = idx
    return mapping


def build_pyramid_features_by_coord(
    loc5_in20: np.ndarray,
    loc10_in20: np.ndarray,
    loc20: np.ndarray,
    feat5: np.ndarray,
    feat10: np.ndarray,
    feat20: np.ndarray,
    dtype: torch.dtype = torch.float32,
    fill_zero_for_missing: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build a (C, M, 21) same-FOV multiscale feature tensor."""
    loc5 = to_int_xy(loc5_in20)
    loc10 = to_int_xy(loc10_in20)
    loc20 = to_int_xy(loc20)
    f5 = ensure_feature_nc(feat5)
    f10 = ensure_feature_nc(feat10)
    f20 = ensure_feature_nc(feat20)

    channels = f5.shape[1]
    if f10.shape[1] != channels or f20.shape[1] != channels:
        raise ValueError(f"Feature dimensions differ: 5x={f5.shape}, 10x={f10.shape}, 20x={f20.shape}")

    n5 = loc5.shape[0]
    step20_x = infer_step_1d(loc20[:, 0]) if loc20.size else 256
    step20_y = infer_step_1d(loc20[:, 1]) if loc20.size else 256
    map10 = build_index(loc10)
    map20 = build_index(loc20)

    order_info = {
        "slots": ["5x_center", "10x_TL", "10x_TR", "10x_BL", "10x_BR"]
        + [f"20x_r{row}c{col}" for row in range(4) for col in range(4)]
    }
    pyramid = torch.zeros((channels, n5, 21), dtype=dtype)
    mask = torch.zeros((n5, 21), dtype=torch.bool)
    if n5 > 0:
        pyramid[:, :, 0] = torch.from_numpy(f5.T).to(dtype)
        mask[:, 0] = True

    offsets10 = [(0, 0), (2 * step20_x, 0), (0, 2 * step20_y), (2 * step20_x, 2 * step20_y)]
    offsets20 = [(col * step20_x, row * step20_y) for row in range(4) for col in range(4)]

    for idx in range(n5):
        x0, y0 = int(loc5[idx, 0]), int(loc5[idx, 1])
        for slot, (dx, dy) in enumerate(offsets10, start=1):
            source_idx = map10.get((x0 + dx, y0 + dy), -1)
            if source_idx != -1:
                pyramid[:, idx, slot] = torch.from_numpy(f10[source_idx]).to(dtype)
                mask[idx, slot] = True
            elif not fill_zero_for_missing:
                raise KeyError(f"Missing 10x coordinate {(x0 + dx, y0 + dy)}")

        for slot, (dx, dy) in enumerate(offsets20, start=5):
            source_idx = map20.get((x0 + dx, y0 + dy), -1)
            if source_idx != -1:
                pyramid[:, idx, slot] = torch.from_numpy(f20[source_idx]).to(dtype)
                mask[idx, slot] = True
            elif not fill_zero_for_missing:
                raise KeyError(f"Missing 20x coordinate {(x0 + dx, y0 + dy)}")

    return pyramid, mask, order_info


def find_feature_file(sample_feature_dir: str, preferred_name: Optional[str] = None) -> str:
    directory = Path(sample_feature_dir)
    if preferred_name:
        candidate = directory / preferred_name
        if candidate.is_file():
            return str(candidate)
    candidates = sorted([p for p in directory.iterdir() if p.suffix.lower() in {".npy", ".pt", ".pth"}])
    if not candidates:
        raise FileNotFoundError(f"No feature file found in {directory}")
    return str(candidates[0])


def read_multiscale_locations(h5_path: str):
    with h5py.File(h5_path, "r") as handle:
        loc5 = handle["locations_5x_in_20x"][:]
        loc10 = handle["locations_10x_in_20x"][:]
        if "locations_20x_raw" in handle:
            loc20 = handle["locations_20x_raw"][:]
        elif "locations_20x_overwritten" in handle:
            loc20 = handle["locations_20x_overwritten"][:]
        else:
            loc20 = handle["locations_20x"][:]
    return loc5, loc10, loc20


def run_pyramid_all(
    h5_root: str,
    feat5_root: str,
    feat10_root: str,
    feat20_root: str,
    output_root: str,
    h5_name: str = "HE_noskip.h5",
    preferred_feature_name: Optional[str] = None,
    dtype: str = "float32",
    fill_zero_for_missing: bool = True,
    sample_ids: Optional[Sequence[str]] = None,
    save_pt: bool = True,
    save_npy: bool = False,
    skip_existing: bool = True,
) -> dict:
    """Build per-sample pyramid features from multiscale h5 files and feature roots."""
    h5_root_path = Path(h5_root)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    torch_dtype = {"float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}.get(dtype.lower(), torch.float32)

    selected = set(sample_ids) if sample_ids else None
    cases = [
        p.name for p in sorted(h5_root_path.iterdir())
        if p.is_dir() and (p / h5_name).is_file() and (selected is None or p.name in selected)
    ]

    ok = skipped = failed = 0
    for case_id in cases:
        out_dir = output_root_path / case_id
        pt_path = out_dir / "pyramid.pt"
        if skip_existing and save_pt and pt_path.exists():
            skipped += 1
            print(f"[skip] {case_id}: existing {pt_path}")
            continue

        try:
            loc5, loc10, loc20 = read_multiscale_locations(str(h5_root_path / case_id / h5_name))
            feat5 = load_feature_matrix(find_feature_file(Path(feat5_root) / case_id, preferred_feature_name))
            feat10 = load_feature_matrix(find_feature_file(Path(feat10_root) / case_id, preferred_feature_name))
            feat20 = load_feature_matrix(find_feature_file(Path(feat20_root) / case_id, preferred_feature_name))

            pyramid, mask, order = build_pyramid_features_by_coord(
                loc5_in20=loc5,
                loc10_in20=loc10,
                loc20=loc20,
                feat5=feat5,
                feat10=feat10,
                feat20=feat20,
                dtype=torch_dtype,
                fill_zero_for_missing=fill_zero_for_missing,
            )

            out_dir.mkdir(parents=True, exist_ok=True)
            if save_pt:
                torch.save({"pyr_feat": pyramid, "pyr_mask": mask, "order": order, "sample_id": case_id}, pt_path)
            if save_npy:
                np.save(out_dir / "pyramid_feat.npy", pyramid.cpu().numpy())
                np.save(out_dir / "pyramid_mask.npy", mask.cpu().numpy())
            with open(out_dir / "order.json", "w", encoding="utf-8") as handle:
                json.dump(order, handle, ensure_ascii=False, indent=2)

            ok += 1
            print(f"[ok] {case_id} -> {tuple(pyramid.shape)}")
        except Exception as exc:
            failed += 1
            print(f"[fail] {case_id}: {exc}")

    summary = {"ok": ok, "skipped": skipped, "failed": failed, "total": len(cases)}
    print(f"[summary] {summary}")
    return summary
