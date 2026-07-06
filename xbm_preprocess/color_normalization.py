"""HistomicsTK deconvolution-based color normalization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def _load_target(target_path: str) -> np.ndarray:
    target = cv2.imread(str(target_path))
    if target is None:
        raise FileNotFoundError(f"Cannot read target image: {target_path}")
    return cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.uint8)


def _to_uint8_rgb(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and np.nanmax(arr) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def normalize_tile_file(
    tile_file: str,
    target: np.ndarray,
    site_file: str,
    output_file: Optional[str] = None,
) -> int:
    """Normalize one tile array and keep only successfully normalized tiles."""
    from histomicstk.preprocessing.color_normalization import (
        deconvolution_based_normalization,
    )

    tile_file = Path(tile_file)
    site_file = Path(site_file)
    output_file = Path(output_file) if output_file is not None else tile_file.with_name(tile_file.stem + "_norm.npy")

    data = _to_uint8_rgb(np.load(tile_file))
    kept_images = []
    kept_indices = []
    for idx, img in enumerate(data):
        try:
            normalized_img = deconvolution_based_normalization(
                img,
                im_target=target,
                stain_unmixing_routine_params={
                    "stains": ["hematoxylin", "eosin"],
                    "stain_unmixing_method": "macenko_pca",
                },
            )
            kept_images.append(_to_uint8_rgb(normalized_img))
            kept_indices.append(idx)
        except Exception as exc:
            print(f"[normalize] drop tile index={idx} in {tile_file.name}: {exc}")

    sites = np.load(site_file)
    np.save(site_file, sites[kept_indices])
    np.save(output_file, np.asarray(kept_images))
    return len(kept_images)


def _normalization_targets(tile_root: str, use_downsample: bool = True) -> List[Tuple[Path, Path]]:
    root = Path(tile_root)
    pairs = []
    for tile_file in root.rglob("*.npy"):
        name = tile_file.name
        if "_site" in name or "_norm" in name:
            continue
        if use_downsample and not name.endswith("_DownSample.npy"):
            continue
        if not use_downsample and name.endswith("_DownSample.npy"):
            continue

        if use_downsample:
            site_file = tile_file.with_name(name.replace("_DownSample.npy", "_site_DownSample.npy"))
        else:
            site_file = tile_file.with_name(name.replace(".npy", "_site.npy"))
        if site_file.exists():
            pairs.append((tile_file, site_file))
        else:
            print(f"[normalize] missing site file for {tile_file}")
    return pairs


def normalize_tile_directory(
    tile_root: str,
    target_path: str,
    use_downsample: bool = True,
    num_workers: int = 8,
    skip_existing: bool = False,
) -> None:
    """Normalize all tile npy files under a directory."""
    target = _load_target(target_path)
    pairs = _normalization_targets(tile_root, use_downsample=use_downsample)
    if skip_existing:
        pairs = [(tile, site) for tile, site in pairs if not tile.with_name(tile.stem + "_norm.npy").exists()]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(normalize_tile_file, str(tile), target, str(site)): tile
            for tile, site in pairs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="normalize"):
            tile = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[normalize] failed file={tile}: {exc}")
