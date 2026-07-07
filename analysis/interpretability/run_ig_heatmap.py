#!/usr/bin/env python3
"""Generate a WSI heatmap from XBM Integrated Gradients instance attribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from _common import generate_heatmap, load_h5_coordinates, normalize_scores, save_xy_scores_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map IG instance attribution to a WSI heatmap.")
    p.add_argument("--ig-path", required=True, help="Path to *_ig.pth produced by run_integrated_gradients.py.")
    p.add_argument("--wsi", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--coords-key", default="auto")
    p.add_argument("--coord-index-npy", default=None, help="Optional indices mapping model instances to H5 coordinates.")
    p.add_argument("--allow-first-n-coords", action="store_true", help="Use the first N H5 coordinates when no coordinate-index file is supplied.")
    p.add_argument("--patch-level", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--normalize-method", default="rank")
    p.add_argument("--score-normalization", choices=["abs-max", "minmax", "none"], default="abs-max")
    p.add_argument("--save-thumbnail", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(args.ig_path, map_location="cpu")
    if "instance_attribution" not in obj:
        raise KeyError(f"instance_attribution not found in {args.ig_path}")
    raw_scores = obj["instance_attribution"].detach().cpu().float().numpy()
    scores = normalize_scores(raw_scores, mode=args.score_normalization)
    coords = load_h5_coordinates(args.h5, args.coords_key, n=len(scores), coord_index_npy=args.coord_index_npy, allow_first_n=args.allow_first_n_coords)

    np.save(out_dir / "ig_scores.npy", raw_scores)
    np.save(out_dir / "ig_scores_for_heatmap.npy", scores)
    np.save(out_dir / "ig_coordinates.npy", coords)
    save_xy_scores_csv(out_dir / "ig_instance_scores.csv", coords, scores, "ig_score")

    sample_id = obj.get("sample_id", Path(args.ig_path).stem)
    print("sample:", sample_id)
    print("coords:", coords.shape)
    print("scores:", scores.shape)
    print("score min/max:", float(scores.min()), float(scores.max()))

    generate_heatmap(
        wsi_path=args.wsi,
        coords=coords,
        scores=scores,
        out_dir=out_dir,
        heatmap_subdir="ig_heatmap",
        patch_size=args.patch_size,
        patch_level=args.patch_level,
        normalize_method=args.normalize_method,
        wsi_name=f"{sample_id}_IG",
        save_thumbnail=args.save_thumbnail,
    )
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
