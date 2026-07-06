"""Extract image tiles from WSI files using CLAM coordinate h5 files."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import openslide
from tqdm import tqdm


def _read_first_dataset(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        if "coords" in handle:
            return np.asarray(handle["coords"][:])
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"No dataset found in {h5_path}")
        return np.asarray(handle[keys[0]][:])


def _find_slide_file(sample_slide_dir: Path, stem: str, extensions: tuple[str, ...]) -> Path:
    for ext in extensions:
        candidate = sample_slide_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No slide file found for stem={stem} in {sample_slide_dir}")


def extract_tiles_for_sample(
    sample_id: str,
    slide_root: str,
    clam_output_root: str,
    tile_size: int = 256,
    slide_extensions: tuple[str, ...] = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs"),
    skip_existing: bool = False,
) -> None:
    """Extract all tiles for one sample from CLAM coordinate files."""
    slide_root = Path(slide_root)
    sample_dir = Path(clam_output_root) / sample_id
    patches_dir = sample_dir / "patches"
    if not patches_dir.exists():
        raise FileNotFoundError(f"Missing CLAM patches directory: {patches_dir}")

    for coord_h5 in sorted(patches_dir.glob("*.h5")):
        stem = coord_h5.stem
        output_tile = sample_dir / f"{stem}.npy"
        output_site = sample_dir / f"{stem}_site.npy"
        if skip_existing and output_tile.exists() and output_site.exists():
            print(f"[tile] skip existing: {sample_id}/{stem}")
            continue

        slide_path = _find_slide_file(slide_root / sample_id, stem, slide_extensions)
        coords = _read_first_dataset(coord_h5)
        slide = openslide.OpenSlide(str(slide_path))
        try:
            tiles = [
                np.asarray(slide.read_region((int(x), int(y)), 0, (tile_size, tile_size)))[:, :, :3]
                for x, y in tqdm(coords, desc=f"{sample_id}/{stem}", leave=False)
            ]
        finally:
            slide.close()

        np.save(output_tile, np.stack(tiles, axis=0))
        np.save(output_site, coords)


def extract_tiles_from_clam_coords(
    slide_root: str,
    clam_output_root: str,
    tile_size: int = 256,
    num_workers: int = 8,
    skip_existing: bool = False,
) -> None:
    """Extract tile arrays for all samples with CLAM coordinates."""
    root = Path(clam_output_root)
    sample_ids = sorted(p.name for p in root.iterdir() if p.is_dir())
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                extract_tiles_for_sample,
                sample_id=sample_id,
                slide_root=slide_root,
                clam_output_root=clam_output_root,
                tile_size=tile_size,
                skip_existing=skip_existing,
            ): sample_id
            for sample_id in sample_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="extract tiles"):
            sample_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[tile] failed sample={sample_id}: {exc}")
