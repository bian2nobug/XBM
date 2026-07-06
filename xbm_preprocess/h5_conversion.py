"""Convert normalized tile npy arrays into compact HDF5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import h5py
import numpy as np


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


def convert_one_sample_to_h5(
    sample_dir: str,
    output_root: str,
    use_downsample: bool = True,
    compression: Optional[str] = "gzip",
    copy_clam_artifacts: bool = True,
    skip_existing: bool = False,
    cleanup_npy: bool = False,
) -> List[Path]:
    """Convert normalized npy tile arrays in one sample folder to h5 files."""
    sample_dir = Path(sample_dir)
    output_root = Path(output_root)
    out_dir = output_root / sample_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_DownSample_norm.npy" if use_downsample else "_norm.npy"
    written = []
    for tile_file in sorted(sample_dir.glob(f"*{suffix}")):
        if use_downsample:
            prefix = tile_file.name.replace("_DownSample_norm.npy", "")
            site_file = sample_dir / f"{prefix}_site_DownSample.npy"
        else:
            prefix = tile_file.name.replace("_norm.npy", "")
            site_file = sample_dir / f"{prefix}_site.npy"

        if not site_file.exists():
            print(f"[h5] missing locations for {tile_file}")
            continue

        h5_path = out_dir / f"{prefix}.h5"
        if skip_existing and h5_path.exists():
            written.append(h5_path)
            continue

        images = np.load(tile_file)
        locations = np.load(site_file)
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset("HE_images", data=images, compression=compression)
            handle.create_dataset("locations", data=locations, compression=compression)
        written.append(h5_path)

        if cleanup_npy:
            tile_file.unlink(missing_ok=True)
            site_file.unlink(missing_ok=True)

    if copy_clam_artifacts:
        for rel in [
            "masks/HE.jpg",
            "patches/HE.h5",
            "stitches/HE.jpg",
            "process_list_autogen.csv",
        ]:
            _copy_if_exists(sample_dir / rel, out_dir / rel)

    return written


def convert_npy_tiles_to_h5(
    input_root: str,
    output_root: str,
    use_downsample: bool = True,
    compression: Optional[str] = "gzip",
    copy_clam_artifacts: bool = True,
    skip_existing: bool = False,
    cleanup_npy: bool = False,
) -> List[Path]:
    """Convert all normalized sample folders under input_root to h5."""
    root = Path(input_root)
    all_written: List[Path] = []
    sample_dirs: Iterable[Path] = sorted(p for p in root.iterdir() if p.is_dir())
    for sample_dir in sample_dirs:
        all_written.extend(
            convert_one_sample_to_h5(
                sample_dir=str(sample_dir),
                output_root=output_root,
                use_downsample=use_downsample,
                compression=compression,
                copy_clam_artifacts=copy_clam_artifacts,
                skip_existing=skip_existing,
                cleanup_npy=cleanup_npy,
            )
        )
    return all_written
