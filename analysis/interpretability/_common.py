#!/usr/bin/env python3
"""Shared utilities for XBM interpretability scripts."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

INTERP_DIR = Path(__file__).resolve().parent
SHARED_DIR = INTERP_DIR / "shared"
HEATMAP_TOOLS_DIR = SHARED_DIR / "heatmap_tools" / "tools"
IG_CORE_DIR = SHARED_DIR / "ig_core"
MODEL_LIB_DIR = SHARED_DIR / "model_lib"
GRADIENT_MODEL_LIB_DIR = MODEL_LIB_DIR / "gradient"

for _path in (HEATMAP_TOOLS_DIR, IG_CORE_DIR, MODEL_LIB_DIR, GRADIENT_MODEL_LIB_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

LEVEL_TO_MAG = {0: 5, 1: 10, 2: 20, 3: 40}
COORD_KEYS = ("locations", "locations_5x_in_20x", "coords", "coordinates")


def positive_int(value: str) -> int:
    value_int = int(value)
    if value_int <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value_int


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def read_sample_ids(sample_path: str | Path) -> list[str]:
    with open(sample_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_tensor_sample(
    data_path: str | Path,
    label_path: str | Path,
    sample_id_path: str | Path,
    sample_id: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, list[str]]:
    data = torch.load(data_path, map_location="cpu")
    labels = torch.load(label_path, map_location="cpu")
    names = read_sample_ids(sample_id_path)

    if not torch.is_tensor(data):
        raise TypeError(f"DATA_PATH must contain a torch.Tensor, got {type(data)}")
    if not torch.is_tensor(labels):
        raise TypeError(f"LABEL_PATH must contain a torch.Tensor, got {type(labels)}")
    if data.shape[0] != labels.shape[0] or data.shape[0] != len(names):
        raise ValueError(
            f"Inconsistent sample counts: data={data.shape[0]}, labels={labels.shape[0]}, sample_ids={len(names)}"
        )
    if sample_id not in names:
        raise ValueError(f"Sample {sample_id!r} not found in {sample_id_path}")

    idx = names.index(sample_id)
    x = data[idx:idx + 1].float().to(device)
    y = labels[idx:idx + 1].long().to(device)
    return x, y, idx, names


def create_xbm_model(
    *,
    split_dims: int,
    clin_dim: int,
    class_dim: int,
    cross_num_layers: int = 2,
    cross_embed_dim: int = 256,
    cross_num_heads: int = 2,
    fusion_pyramid_progressive: bool = True,
    trans_perciever: bool = True,
    regression: bool = False,
    joint_heads: int = 4,
    gradient: bool = False,
    device: torch.device,
):
    if gradient:
        from utils_multiscale_trans_attnpooling_change_gradient import utils_multiScale_model_trans
        model = utils_multiScale_model_trans(
            split_dims=split_dims,
            clin_dim=clin_dim,
            Cross_num_layers=cross_num_layers,
            Cross_embed_dim=cross_embed_dim,
            Cross_num_heads=cross_num_heads,
            Classify_dim=class_dim,
            Fusion_PyramidProgressive=fusion_pyramid_progressive,
            trans_perciever=trans_perciever,
            regression=regression,
            joint_heads=joint_heads,
            for_gradient=True,
        )
    else:
        from utils_multiscale_trans_attnpooling_change import utils_multiScale_model_trans
        model = utils_multiScale_model_trans(
            split_dims=split_dims,
            clin_dim=clin_dim,
            Cross_num_layers=cross_num_layers,
            Cross_embed_dim=cross_embed_dim,
            Cross_num_heads=cross_num_heads,
            Classify_dim=class_dim,
            Fusion_PyramidProgressive=fusion_pyramid_progressive,
            trans_perciever=trans_perciever,
            regression=regression,
            joint_heads=joint_heads,
        )
    return model.to(device)


def extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Could not find a state_dict in object of type {type(obj)}")


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint: Optional[str], strict: bool = False) -> None:
    if not checkpoint:
        print("checkpoint: none")
        return
    obj = torch.load(checkpoint, map_location="cpu")
    state = extract_state_dict(obj)
    cleaned = {}
    for key, value in state.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    print(f"checkpoint loaded: {checkpoint}")
    if missing:
        print(f"missing keys: {len(missing)}")
    if unexpected:
        print(f"unexpected keys: {len(unexpected)}")


def load_h5_coordinates(
    h5_path: str | Path,
    coords_key: str = "auto",
    n: Optional[int] = None,
    coord_index_npy: Optional[str | Path] = None,
) -> np.ndarray:
    import h5py

    with h5py.File(h5_path, "r") as handle:
        if coords_key == "auto":
            key = next((k for k in COORD_KEYS if k in handle), None)
        else:
            key = coords_key
        if key is None or key not in handle:
            raise KeyError(f"No coordinate dataset found in {h5_path}; tried {COORD_KEYS}")
        coords = np.asarray(handle[key][()])
        if "lens" in handle:
            lens = int(np.asarray(handle["lens"][()]).reshape(-1)[0])
            coords = coords[:lens]
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Coordinates must be shaped (N, 2+), got {coords.shape}")
    coords = coords[:, :2].astype(np.float32)

    if coord_index_npy:
        indices = np.load(coord_index_npy)
        indices = np.asarray(indices).reshape(-1).astype(int)
        coords = coords[indices]
        if n is not None and len(coords) != n:
            raise ValueError(f"coord_index_npy selected {len(coords)} coordinates, expected n={n}")
    elif n is not None:
        if len(coords) < n:
            raise ValueError(f"H5 contains only {len(coords)} coordinates but n={n} scores were supplied")
        coords = coords[:n]
        print(
            "--coord-index-npy was not provided; using the first "
            f"{n} coordinates from the H5 file."
        )
    return coords


def normalize_scores(scores: np.ndarray, mode: str = "abs-max") -> np.ndarray:
    scores = np.asarray(scores).reshape(-1).astype(np.float32)
    if mode == "abs-max":
        scores = np.abs(scores)
        return scores / max(float(scores.max()), 1e-8)
    if mode == "minmax":
        scores = scores - float(scores.min())
        return scores / max(float(scores.max()), 1e-8)
    if mode == "none":
        return scores
    raise ValueError(f"Unknown score normalization mode: {mode}")


def heatmap_params(normalize_method: str, wsi_name: str) -> Dict[str, object]:
    return {
        "thumbnail_size_scale": (0.125, 0.125),
        "style": "JET",
        "alpha": None,
        "normalize_method": normalize_method,
        "smooth": True,
        "wsi_name": wsi_name,
        "add_colorbar": True,
        "add_title": True,
    }


def generate_heatmap(
    *,
    wsi_path: str,
    coords: np.ndarray,
    scores: np.ndarray,
    out_dir: str | Path,
    heatmap_subdir: str,
    patch_size: int,
    patch_level: int,
    normalize_method: str = "rank",
    wsi_name: Optional[str] = None,
    save_thumbnail: bool = True,
) -> None:
    from HeatmapGenerator import heatmap_main

    out_dir = Path(out_dir)
    thumbnail_dir = out_dir / "thumbnail"
    heatmap_dir = out_dir / heatmap_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    if wsi_name is None:
        wsi_name = Path(wsi_path).stem

    heatmap_main(
        wsi_path=wsi_path,
        coordinates=np.asarray(coords, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32).reshape(-1),
        patch_size=(patch_size, patch_size),
        patch_level=patch_level,
        thumbnail_dir=str(thumbnail_dir),
        heatmap_dir=str(heatmap_dir),
        heatmap_params=heatmap_params(normalize_method, wsi_name),
        save_thumbail=save_thumbnail,
    )


def save_xy_scores_csv(path: str | Path, coords: np.ndarray, scores: np.ndarray, score_name: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", score_name])
        for (x, y), s in zip(coords[:, :2], scores):
            writer.writerow([float(x), float(y), float(s)])
