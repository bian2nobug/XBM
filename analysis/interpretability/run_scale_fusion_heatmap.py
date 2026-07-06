#!/usr/bin/env python3
"""Generate 5x/10x/20x WSI heatmaps from FC-AttnPooling scale-fusion weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import generate_heatmap, load_h5_coordinates, normalize_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate scale-fusion contribution heatmaps.")
    p.add_argument("--scale-path", required=True, help="Path to scale-fusion .pth produced by the forward extraction script.")
    p.add_argument("--wsi", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--coords-key", default="auto")
    p.add_argument("--coord-index-npy", default=None, help="Optional indices mapping model instances to H5 coordinates.")
    p.add_argument("--patch-level", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--normalize-method", default="rank")
    p.add_argument("--score-normalization", choices=["minmax", "none"], default="minmax")
    p.add_argument("--save-thumbnail", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(args.scale_path, map_location="cpu")
    scale_scores = {
        "5x": obj["scale_5x"].detach().cpu().float().numpy().reshape(-1),
        "10x": obj["scale_10x"].detach().cpu().float().numpy().reshape(-1),
        "20x": obj["scale_20x"].detach().cpu().float().numpy().reshape(-1),
    }
    n = len(scale_scores["5x"])
    coords = load_h5_coordinates(args.h5, args.coords_key, n=n, coord_index_npy=args.coord_index_npy)

    sample_id = obj.get("sample_id", Path(args.scale_path).stem)
    summary = {
        "sample_id": sample_id,
        "label": obj.get("label", None),
        "scale_5x_mean": float(scale_scores["5x"].mean()),
        "scale_10x_mean": float(scale_scores["10x"].mean()),
        "scale_20x_mean": float(scale_scores["20x"].mean()),
        "scale_5x_min": float(scale_scores["5x"].min()),
        "scale_10x_min": float(scale_scores["10x"].min()),
        "scale_20x_min": float(scale_scores["20x"].min()),
        "scale_5x_max": float(scale_scores["5x"].max()),
        "scale_10x_max": float(scale_scores["10x"].max()),
        "scale_20x_max": float(scale_scores["20x"].max()),
        "scale_sum_mean": float((scale_scores["5x"] + scale_scores["10x"] + scale_scores["20x"]).mean()),
    }

    pd.DataFrame([summary]).to_csv(out_dir / "scale_fusion_summary.csv", index=False)
    pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "scale_5x": scale_scores["5x"],
            "scale_10x": scale_scores["10x"],
            "scale_20x": scale_scores["20x"],
        }
    ).to_csv(out_dir / "scale_fusion_instance_scores.csv", index=False)

    np.save(out_dir / "scale_coordinates.npy", coords)
    for scale_name, raw in scale_scores.items():
        np.save(out_dir / f"scale_{scale_name}.npy", raw)

    print("sample:", sample_id)
    print("coords:", coords.shape)
    print("5x mean:", summary["scale_5x_mean"])
    print("10x mean:", summary["scale_10x_mean"])
    print("20x mean:", summary["scale_20x_mean"])
    print("sum mean:", summary["scale_sum_mean"])

    for scale_name, raw_scores in scale_scores.items():
        scores = normalize_scores(raw_scores, mode=args.score_normalization)
        print(f"\nGenerating scale {scale_name} heatmap...")
        print("score min/max:", float(scores.min()), float(scores.max()))
        generate_heatmap(
            wsi_path=args.wsi,
            coords=coords,
            scores=scores,
            out_dir=out_dir,
            heatmap_subdir=f"scale_{scale_name}_heatmap",
            patch_size=args.patch_size,
            patch_level=args.patch_level,
            normalize_method=args.normalize_method,
            wsi_name=f"{sample_id}_scale_{scale_name}",
            save_thumbnail=args.save_thumbnail,
        )
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
