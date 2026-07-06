#!/usr/bin/env python3
"""Generate a WSI heatmap from cross-attention AxG instance scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from _common import generate_heatmap, load_h5_coordinates, normalize_scores, save_xy_scores_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map cross-attention AxG scores to a WSI heatmap.")
    p.add_argument("--axg-path", required=True, help="Path to *_cross_axg.pth produced by run_cross_axg.py.")
    p.add_argument("--wsi", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--coords-key", default="auto")
    p.add_argument("--coord-index-npy", default=None, help="Optional indices mapping model instances to H5 coordinates.")
    p.add_argument("--patch-level", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--normalize-method", default="rank")
    p.add_argument("--score-normalization", choices=["abs-max", "minmax", "none"], default="none")
    p.add_argument("--save-thumbnail", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(args.axg_path, map_location="cpu")
    if "instance_axg" not in obj:
        raise KeyError(f"instance_axg not found in {args.axg_path}")
    raw_scores = obj["instance_axg"].detach().cpu().float().numpy()
    scores = normalize_scores(raw_scores, mode=args.score_normalization)
    coords = load_h5_coordinates(args.h5, args.coords_key, n=len(scores), coord_index_npy=args.coord_index_npy)

    np.save(out_dir / "axg_scores.npy", raw_scores)
    np.save(out_dir / "axg_scores_for_heatmap.npy", scores)
    np.save(out_dir / "axg_coordinates.npy", coords)
    save_xy_scores_csv(out_dir / "axg_instance_scores.csv", coords, scores, "axg_score")

    sample_id = obj.get("sample_id", Path(args.axg_path).stem)
    print("sample:", sample_id)
    print("coords:", coords.shape)
    print("scores:", scores.shape)
    print("score min/max:", float(scores.min()), float(scores.max()))

    generate_heatmap(
        wsi_path=args.wsi,
        coords=coords,
        scores=scores,
        out_dir=out_dir,
        heatmap_subdir="axg_heatmap",
        patch_size=args.patch_size,
        patch_level=args.patch_level,
        normalize_method=args.normalize_method,
        wsi_name=f"{sample_id}_AxG",
        save_thumbnail=args.save_thumbnail,
    )
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
