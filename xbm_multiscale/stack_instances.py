"""Stack per-sample pyramid features into a padded tensor."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def load_pyramid_tensor(path: str) -> torch.Tensor:
    """Load a pyramid tensor from .pt/.pth or .npy."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        tensor = torch.from_numpy(np.load(p))
    elif p.suffix.lower() in {".pt", ".pth"}:
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            tensor = obj
        elif isinstance(obj, dict):
            tensor = obj.get("pyr_feat", None)
            if not isinstance(tensor, torch.Tensor):
                tensor = next((v for v in obj.values() if isinstance(v, torch.Tensor) and v.ndim == 3), None)
        else:
            tensor = None
        if tensor is None:
            raise ValueError(f"No 3D tensor found in {path}")
    else:
        raise ValueError(f"Unsupported pyramid file: {path}")

    if tensor.ndim != 3:
        raise ValueError(f"Pyramid tensor must be 3D, got {tuple(tensor.shape)}")
    return tensor


def to_channels_m_views(tensor: torch.Tensor, channels: int = 1536, views: int = 21) -> Tuple[torch.Tensor, int]:
    shape = tuple(tensor.shape)
    if channels not in shape or views not in shape:
        raise ValueError(f"Shape does not contain channels={channels} and views={views}: {shape}")
    i_channels = shape.index(channels)
    i_views = shape.index(views)
    i_m = [i for i in range(3) if i not in (i_channels, i_views)][0]
    m = shape[i_m]
    return tensor.permute(i_channels, i_m, i_views).contiguous(), int(m)


def find_pyramid_files(root: str, filename: str = "pyramid.pt") -> List[Path]:
    root_path = Path(root)
    files = sorted(root_path.glob(f"**/{filename}"))
    if not files and filename == "pyramid.pt":
        files = sorted(root_path.glob("**/pyramid_feat.npy"))
    return files


def stack_pyramid_features(
    pyramid_root: str,
    output_name: str = "stacked_fp16_N_M.pt",
    pyramid_filename: str = "pyramid.pt",
    dtype: str = "float16",
    channels: int = 1536,
    views: int = 21,
) -> dict:
    """Pad and stack all per-sample pyramid tensors into (N, C, maxM, V)."""
    files = find_pyramid_files(pyramid_root, filename=pyramid_filename)
    records = []
    for path in files:
        try:
            tensor = load_pyramid_tensor(str(path))
            tensor, m = to_channels_m_views(tensor, channels=channels, views=views)
            if m > 0:
                records.append((path, tensor, m))
        except Exception as exc:
            print(f"[skip] {path}: {exc}")

    if not records:
        raise RuntimeError(f"No valid pyramid tensors found under {pyramid_root}")

    max_m = max(m for _, _, m in records)
    out_dtype = {"float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}.get(dtype.lower(), torch.float16)
    stacked = torch.zeros((len(records), channels, max_m, views), dtype=out_dtype)
    lengths = torch.empty((len(records),), dtype=torch.int32)
    paths = []
    sample_ids = []
    for idx, (path, tensor, m) in enumerate(records):
        stacked[idx, :, :m, :] = tensor.to(out_dtype)
        lengths[idx] = m
        paths.append(str(path))
        sample_ids.append(path.parent.name)

    mask = torch.arange(max_m).unsqueeze(0) < lengths.unsqueeze(1)
    save_path = Path(pyramid_root) / output_name
    torch.save(
        {
            "tensor": stacked,
            "mask": mask,
            "lengths": lengths,
            "paths": paths,
            "sample_ids": sample_ids,
            "maxM": int(max_m),
        },
        save_path,
    )
    summary = {"n": len(records), "maxM": int(max_m), "output": str(save_path), "shape": tuple(stacked.shape)}
    print(f"[ok] stacked -> {summary}")
    return summary
