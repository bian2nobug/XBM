"""Downsample tile arrays and corresponding coordinates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def downsample_image_array(images: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample an image batch by an integer factor."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return images

    output_size = (images.shape[2] // factor, images.shape[1] // factor)
    return np.stack([cv2.resize(img, output_size, interpolation=cv2.INTER_AREA) for img in images], axis=0)


def downsample_sample_dir(sample_dir: str, factor: int = 2, skip_existing: bool = False) -> None:
    """Downsample all raw tile npy files in one sample folder."""
    sample_dir = Path(sample_dir)
    if skip_existing and (sample_dir / "HE.h5").exists():
        print(f"[downsample] skip existing final h5: {sample_dir.name}")
        return

    for npy_path in sorted(sample_dir.glob("*.npy")):
        name = npy_path.name
        if "_DownSample" in name or "_norm" in name:
            continue

        out_path = npy_path.with_name(npy_path.stem + "_DownSample.npy")
        if skip_existing and out_path.exists():
            print(f"[downsample] skip existing: {out_path}")
            continue

        data = np.load(npy_path)
        if name.endswith("_site.npy"):
            np.save(out_path, data / factor)
        else:
            np.save(out_path, downsample_image_array(data, factor=factor))


def downsample_tile_directory(
    tile_root: str,
    factor: int = 2,
    num_workers: int = 8,
    skip_existing: bool = False,
) -> None:
    """Downsample all sample folders under a tile directory."""
    root = Path(tile_root)
    sample_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(downsample_sample_dir, str(sample_dir), factor, skip_existing): sample_dir
            for sample_dir in sample_dirs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="downsample"):
            sample_dir = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[downsample] failed sample={sample_dir.name}: {exc}")
