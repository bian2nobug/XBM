#!/usr/bin/env python3
"""WSI heatmap smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import LEVEL_TO_MAG, generate_heatmap, load_h5_coordinates, normalize_scores, save_xy_scores_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a WSI heatmap from patch-level scores.")
    p.add_argument("--wsi", required=True, help="Path to WSI file, for example HE.svs.")
    p.add_argument("--h5", default=None, help="Optional H5 file containing tile coordinates.")
    p.add_argument("--coords-key", default="auto", help="H5 coordinate key, or auto.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--patch-level", type=int, default=2, choices=sorted(LEVEL_TO_MAG), help="0=5x, 1=10x, 2=20x, 3=40x.")
    p.add_argument("--patch-size", type=int, default=512, help="Patch size at patch-level coordinates.")
    p.add_argument("--num-patches", type=int, default=1000, help="Number of random patches if --h5 is omitted.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-mode", choices=["uniform", "hotspots"], default="hotspots")
    p.add_argument("--normalize-method", default="rank")
    p.add_argument("--save-thumbnail", action="store_true")
    return p.parse_args()


def _read_wsi_magnification(slide) -> float:
    for key in ("openslide.objective-power", "aperio.AppMag"):
        value = slide.properties.get(key)
        if value is not None:
            return float(value)
    return 40.0


def _sample_random_coordinates(wsi_path: str, num_patches: int, patch_size: int, patch_level: int, seed: int) -> np.ndarray:
    import openslide

    rng = np.random.default_rng(seed)
    slide = openslide.OpenSlide(wsi_path)
    width0, height0 = slide.dimensions
    wsi_mag = _read_wsi_magnification(slide)
    slide.close()

    patch_mag = LEVEL_TO_MAG[patch_level]
    downsample = wsi_mag / patch_mag
    coord_width = max(1, int(width0 / downsample) - patch_size)
    coord_height = max(1, int(height0 / downsample) - patch_size)
    xs = rng.integers(0, coord_width, size=num_patches)
    ys = rng.integers(0, coord_height, size=num_patches)
    return np.column_stack([xs, ys]).astype(np.float32)


def _make_scores(coords: np.ndarray, mode: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mode == "uniform":
        return rng.random(coords.shape[0]).astype(np.float32)

    base = rng.random(coords.shape[0]) * 0.15
    centers = coords[rng.choice(coords.shape[0], size=min(5, coords.shape[0]), replace=False)]
    scores = base.copy()
    for center in centers:
        dist = np.linalg.norm(coords - center[None, :], axis=1)
        sigma = max(float(np.percentile(dist, 20)), 1.0)
        scores += np.exp(-(dist ** 2) / (2 * sigma ** 2))
    return normalize_scores(scores, mode="abs-max")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.h5:
        coords = load_h5_coordinates(args.h5, args.coords_key)
    else:
        coords = _sample_random_coordinates(args.wsi, args.num_patches, args.patch_size, args.patch_level, args.seed)
    scores = _make_scores(coords, args.score_mode, args.seed)

    np.save(out_dir / "random_coordinates.npy", coords)
    np.save(out_dir / "random_weights.npy", scores)
    save_xy_scores_csv(out_dir / "random_weights.csv", coords, scores, "random_weight")

    generate_heatmap(
        wsi_path=args.wsi,
        coords=coords,
        scores=scores,
        out_dir=out_dir,
        heatmap_subdir="random_weight_heatmap",
        patch_size=args.patch_size,
        patch_level=args.patch_level,
        normalize_method=args.normalize_method,
        wsi_name=Path(args.wsi).stem,
        save_thumbnail=args.save_thumbnail,
    )
    print(f"WSI heatmap smoke test finished: {out_dir}")


if __name__ == "__main__":
    main()
