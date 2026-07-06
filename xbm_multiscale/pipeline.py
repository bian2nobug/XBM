"""Config-driven multiscale construction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_multiscale_pipeline(config_path: str, dry_run: bool = False) -> None:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    stages = cfg.get("stages", {})
    common = cfg.get("common", {})

    h5_root = paths["multiscale_h5_root"]
    if dry_run:
        print("[dry-run] no files will be written")
        print(f"[dry-run] tile_h5_root: {paths['tile_h5_root']}")
        print(f"[dry-run] multiscale_h5_root: {h5_root}")
        print(f"[dry-run] prov-gigapath checkpoint: {paths.get('provgigapath_checkpoint')}")
        print(f"[dry-run] feat5_root: {paths['feat5_root']}")
        print(f"[dry-run] feat10_root: {paths['feat10_root']}")
        print(f"[dry-run] feat20_root: {paths['feat20_root']}")
        print(f"[dry-run] pyramid_root: {paths['pyramid_root']}")
        print(f"[dry-run] enabled stages: {', '.join([k for k, v in stages.items() if v])}")
        tile_root = Path(paths["tile_h5_root"])
        if tile_root.exists():
            sample_count = len([p for p in tile_root.iterdir() if p.is_dir()])
            print(f"[dry-run] sample folders under tile_h5_root: {sample_count}")
        return

    if stages.get("build_multiscale_h5", True):
        from .multiscale_h5 import build_multiscale_h5_tree

        multiscale_cfg = cfg.get("multiscale_h5", {})
        build_multiscale_h5_tree(
            tile_root=paths["tile_h5_root"],
            output_root=h5_root,
            input_h5_name=multiscale_cfg.get("input_h5_name", "HE.h5"),
            output_h5_name=multiscale_cfg.get("output_h5_name", "HE_noskip.h5"),
            pad_mode=multiscale_cfg.get("pad_mode", "skip"),
            compression_opts=int(multiscale_cfg.get("compression_opts", 4)),
            fixed_step_20x=tuple(multiscale_cfg.get("fixed_step_20x", [256, 256])),
            sample_ids=common.get("sample_ids"),
            skip_existing=bool(common.get("skip_existing", True)),
        )

    if stages.get("extract_provgigapath_features", True):
        from .feature_extraction import extract_multiscale_feature_roots

        feature_cfg = cfg.get("feature_extraction", {})
        extract_multiscale_feature_roots(
            input_root=h5_root,
            feat5_root=paths["feat5_root"],
            feat10_root=paths["feat10_root"],
            feat20_root=paths["feat20_root"],
            checkpoint_path=paths["provgigapath_checkpoint"],
            h5_name=feature_cfg.get("h5_name", "HE_noskip.h5"),
            output_suffix=feature_cfg.get("output_suffix", "prov_gigapath_feature.npy"),
            device=feature_cfg.get("device", "cuda:0"),
            batch_size=int(feature_cfg.get("batch_size", 16)),
            num_workers=int(feature_cfg.get("num_workers", 0)),
            image_size=int(feature_cfg.get("image_size", 224)),
            model_name=feature_cfg.get("model_name", "hf_hub:prov-gigapath/prov-gigapath"),
            sample_ids=common.get("sample_ids"),
            skip_existing=bool(common.get("skip_existing", True)),
        )

    if stages.get("build_pyramid_features", True):
        from .pyramid_features import run_pyramid_all

        pyramid_cfg = cfg.get("pyramid", {})
        run_pyramid_all(
            h5_root=h5_root,
            feat5_root=paths["feat5_root"],
            feat10_root=paths["feat10_root"],
            feat20_root=paths["feat20_root"],
            output_root=paths["pyramid_root"],
            h5_name=pyramid_cfg.get("h5_name", "HE_noskip.h5"),
            preferred_feature_name=pyramid_cfg.get("preferred_feature_name"),
            dtype=pyramid_cfg.get("dtype", "float32"),
            fill_zero_for_missing=bool(pyramid_cfg.get("fill_zero_for_missing", True)),
            sample_ids=common.get("sample_ids"),
            save_pt=bool(pyramid_cfg.get("save_pt", True)),
            save_npy=bool(pyramid_cfg.get("save_npy", False)),
            skip_existing=bool(common.get("skip_existing", True)),
        )

    if stages.get("stack_instances", False):
        from .stack_instances import stack_pyramid_features

        stack_cfg = cfg.get("stack", {})
        stack_pyramid_features(
            pyramid_root=paths["pyramid_root"],
            output_name=stack_cfg.get("output_name", "stacked_fp16_N_M.pt"),
            pyramid_filename=stack_cfg.get("pyramid_filename", "pyramid.pt"),
            dtype=stack_cfg.get("dtype", "float16"),
            channels=int(stack_cfg.get("channels", 1536)),
            views=int(stack_cfg.get("views", 21)),
        )
