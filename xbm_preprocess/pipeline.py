"""Config-driven preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .clam_coords import run_clam_for_slides
from .color_normalization import normalize_tile_directory
from .downsample import downsample_tile_directory
from .h5_conversion import convert_npy_tiles_to_h5
from .tile_extraction import extract_tiles_from_clam_coords


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_preprocess_pipeline(config_path: str) -> None:
    """Run selected preprocessing stages from a YAML config."""
    cfg = load_config(config_path)
    paths = cfg["paths"]
    stages = cfg.get("stages", {})
    common = cfg.get("common", {})
    skip_existing = bool(common.get("skip_existing", True))
    num_workers = int(common.get("num_workers", 8))

    clam_output_root = paths["clam_output_root"]

    if stages.get("run_clam", True):
        run_clam_for_slides(
            slide_root=paths["slide_root"],
            output_root=clam_output_root,
            clam_script=paths["clam_script"],
            preset_map=paths["preset_map"],
            segment_parameter=cfg.get("clam", {}).get("segment_parameter", "tcga"),
            tile_size=int(cfg.get("clam", {}).get("tile_size", 256)),
            step_size=int(cfg.get("clam", {}).get("step_size", 256)),
            magnification_table=paths.get("magnification_table"),
            num_workers=num_workers,
            skip_existing=skip_existing,
        )

    magnification_dir = cfg.get("tiles", {}).get("magnification_dir", "Magnification20")
    tile_root = str(Path(clam_output_root) / magnification_dir)

    if stages.get("extract_tiles", True):
        extract_tiles_from_clam_coords(
            slide_root=paths["slide_root"],
            clam_output_root=tile_root,
            tile_size=int(cfg.get("tiles", {}).get("tile_size", 256)),
            num_workers=num_workers,
            skip_existing=skip_existing,
        )

    if stages.get("downsample", False):
        downsample_tile_directory(
            tile_root=tile_root,
            factor=int(cfg.get("downsample", {}).get("factor", 2)),
            num_workers=num_workers,
            skip_existing=skip_existing,
        )

    if stages.get("normalize", True):
        normalize_tile_directory(
            tile_root=tile_root,
            target_path=paths["normalization_target"],
            use_downsample=bool(cfg.get("normalization", {}).get("use_downsample", False)),
            num_workers=num_workers,
            skip_existing=skip_existing,
        )

    if stages.get("convert_h5", True):
        convert_npy_tiles_to_h5(
            input_root=tile_root,
            output_root=paths["h5_output_root"],
            use_downsample=bool(cfg.get("h5", {}).get("use_downsample", False)),
            compression=cfg.get("h5", {}).get("compression", "gzip"),
            copy_clam_artifacts=bool(cfg.get("h5", {}).get("copy_clam_artifacts", True)),
            skip_existing=skip_existing,
            cleanup_npy=bool(cfg.get("h5", {}).get("cleanup_npy", False)),
        )
