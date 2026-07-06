"""Build 20x/10x/5x multiscale HDF5 files from 20x tile HDF5 inputs."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set, Tuple, Union

import h5py
import numpy as np


try:
    import cv2

    def downsample_2x(image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

except ImportError:
    from PIL import Image

    def downsample_2x(image: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(image).resize((256, 256), resample=Image.BILINEAR))


REQUIRED_MULTISCALE_KEYS = [
    "HE_images_20x",
    "locations_20x_raw",
    "locations_20x_overwritten",
    "HE_images_10x",
    "locations_10x_in_20x",
    "locations_10x_in_10x",
    "HE_images_5x",
    "locations_5x_in_20x",
    "locations_5x_in_5x",
]


def normalize_locations(locations: np.ndarray) -> np.ndarray:
    """Normalize coordinate arrays to shape (N, 2)."""
    arr = np.asarray(locations)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    if getattr(arr.dtype, "fields", None):
        keys = list(arr.dtype.fields)
        if "x" in keys and "y" in keys:
            out = np.column_stack([arr["x"], arr["y"]])
        elif "X" in keys and "Y" in keys:
            out = np.column_stack([arr["X"], arr["Y"]])
        else:
            out = np.column_stack([arr[keys[0]], arr[keys[1]]])
        return np.rint(out).astype(np.int64)

    if arr.dtype == object:
        return np.rint(np.vstack([np.asarray(x) for x in arr])).astype(np.int64)

    if arr.ndim == 2 and arr.shape[1] == 2:
        return np.rint(arr).astype(np.int64) if not np.issubdtype(arr.dtype, np.integer) else arr.astype(np.int64)

    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError(f"Odd-length coordinate array cannot be reshaped to (N,2): {arr.shape}")
        return np.rint(arr.reshape(-1, 2)).astype(np.int64)

    raise ValueError(f"Cannot normalize locations with shape={arr.shape}, dtype={arr.dtype}")


def build_grid(locations: np.ndarray, fixed_step: Optional[Tuple[int, int]] = None):
    """Build a coordinate-to-index grid from top-left tile coordinates."""
    locs = normalize_locations(locations)
    if locs.shape[0] == 0:
        step_x, step_y = fixed_step or (256, 256)
        return defaultdict(lambda: -1), (0, 0, step_x, step_y)

    xs = np.unique(locs[:, 0])
    ys = np.unique(locs[:, 1])
    xs.sort()
    ys.sort()

    if fixed_step is None:
        step_x = int(np.median(np.diff(xs))) if len(xs) > 1 else 256
        step_y = int(np.median(np.diff(ys))) if len(ys) > 1 else 256
    else:
        step_x, step_y = fixed_step

    step_x = max(int(step_x), 1)
    step_y = max(int(step_y), 1)
    x0, y0 = int(xs.min()), int(ys.min())

    ix = np.rint((locs[:, 0] - x0) / step_x).astype(int)
    iy = np.rint((locs[:, 1] - y0) / step_y).astype(int)

    grid = defaultdict(lambda: -1)
    for idx, row_col in enumerate(zip(iy, ix)):
        grid[row_col] = idx
    return grid, (x0, y0, step_x, step_y)


def stitch_2x2(img00: np.ndarray, img01: np.ndarray, img10: np.ndarray, img11: np.ndarray) -> np.ndarray:
    """Stitch four 256x256 tiles and downsample to one 256x256 tile."""
    big = np.empty((512, 512, 3), dtype=img00.dtype)
    big[:256, :256] = img00
    big[:256, 256:] = img01
    big[256:, :256] = img10
    big[256:, 256:] = img11
    return downsample_2x(big)


def make_10x_from_20x(
    images20: np.ndarray,
    locations20: np.ndarray,
    pad_mode: str = "skip",
    fixed_step_20x: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct 10x-equivalent tiles from 20x 2x2 tile groups."""
    locs20 = normalize_locations(locations20)
    grid, _ = build_grid(locs20, fixed_step=fixed_step_20x)
    used_top_left = []
    images10 = []
    locs20_overwritten = locs20.copy()

    if not grid:
        return (
            np.empty((0, 256, 256, 3), dtype=images20.dtype),
            np.empty((0, 2), dtype=locs20.dtype),
            locs20_overwritten,
            np.empty((0, 2), dtype=locs20.dtype),
        )

    rows = sorted({row for row, _ in grid.keys()})
    cols = sorted({col for _, col in grid.keys()})
    for row in rows:
        if row % 2:
            continue
        for col in cols:
            if col % 2:
                continue
            idx00 = grid[(row, col)]
            idx01 = grid[(row, col + 1)]
            idx10 = grid[(row + 1, col)]
            idx11 = grid[(row + 1, col + 1)]

            if -1 in (idx00, idx01, idx10, idx11):
                if pad_mode == "skip":
                    continue

                def pick(idx: int) -> np.ndarray:
                    return images20[idx] if idx != -1 else np.zeros_like(images20[0])

                image = stitch_2x2(pick(idx00), pick(idx01), pick(idx10), pick(idx11))
                anchor_idx = next((x for x in (idx00, idx01, idx10, idx11) if x != -1), None)
                top_left = locs20[anchor_idx] if anchor_idx is not None else np.array([0, 0], dtype=locs20.dtype)
            else:
                image = stitch_2x2(images20[idx00], images20[idx01], images20[idx10], images20[idx11])
                locs20_overwritten[[idx00, idx01, idx10, idx11]] = locs20[idx00]
                top_left = locs20[idx00]

            images10.append(image)
            used_top_left.append(np.asarray(top_left))

    if images10:
        images10_arr = np.stack(images10, axis=0)
        locs10_in20 = np.vstack(used_top_left).astype(locs20.dtype)
    else:
        images10_arr = np.empty((0, 256, 256, 3), dtype=images20.dtype)
        locs10_in20 = np.empty((0, 2), dtype=locs20.dtype)

    locs10_in10 = (locs10_in20 // 2).astype(np.int64) if locs10_in20.size else np.empty((0, 2), dtype=np.int64)
    return images10_arr, locs10_in20, locs20_overwritten, locs10_in10


def make_5x_from_10x(
    images10: np.ndarray,
    locations10_in20: np.ndarray,
    pad_mode: str = "skip",
    fixed_step_10x_in20: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct 5x-equivalent tiles from 10x 2x2 tile groups."""
    locs10 = normalize_locations(locations10_in20)
    if images10.shape[0] == 0 or locs10.shape[0] == 0:
        empty_images = np.empty((0, 256, 256, 3), dtype=images10.dtype)
        empty_locs = np.empty((0, 2), dtype=locs10.dtype)
        return empty_images, empty_locs, locs10, empty_locs
    images5, locs5_in20, locs10_overwritten, _ = make_10x_from_20x(
        images10,
        locs10,
        pad_mode=pad_mode,
        fixed_step_20x=fixed_step_10x_in20,
    )
    locs5_in5 = (locs5_in20 // 4).astype(np.int64) if locs5_in20.size else np.empty((0, 2), dtype=np.int64)
    return images5, locs5_in20, locs10_overwritten, locs5_in5


def file_has_multiscale_keys(h5_path: Path) -> bool:
    try:
        with h5py.File(h5_path, "r") as handle:
            return all(key in handle for key in REQUIRED_MULTISCALE_KEYS)
    except Exception:
        return False


def build_multiscale_h5_for_sample(
    input_h5: str,
    output_h5: str,
    pad_mode: str = "skip",
    compression_opts: int = 4,
    fixed_step_20x: Optional[Tuple[int, int]] = (256, 256),
    skip_existing: bool = True,
) -> bool:
    """Create one HE_noskip.h5 containing 20x, 10x, and 5x tile arrays."""
    input_h5_path = Path(input_h5)
    output_h5_path = Path(output_h5)
    if skip_existing and output_h5_path.exists() and file_has_multiscale_keys(output_h5_path):
        return False

    with h5py.File(input_h5_path, "r") as handle:
        images20 = handle["HE_images"][:]
        locs20_raw = handle["locations"][:]
    locs20 = normalize_locations(locs20_raw)

    images10, locs10_in20, locs20_over, locs10_in10 = make_10x_from_20x(
        images20,
        locs20,
        pad_mode=pad_mode,
        fixed_step_20x=fixed_step_20x,
    )
    images5, locs5_in20, _, locs5_in5 = make_5x_from_10x(
        images10,
        locs10_in20,
        pad_mode=pad_mode,
        fixed_step_10x_in20=(fixed_step_20x[0] * 2, fixed_step_20x[1] * 2) if fixed_step_20x else None,
    )

    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_h5 = output_h5_path.with_suffix(".tmp.h5")
    with h5py.File(tmp_h5, "w") as handle:
        handle.create_dataset("HE_images_20x", data=images20, compression="gzip", compression_opts=compression_opts, chunks=True)
        handle.create_dataset("locations_20x_raw", data=locs20)
        handle.create_dataset("locations_20x_overwritten", data=locs20_over)
        handle.create_dataset("HE_images_10x", data=images10, compression="gzip", compression_opts=compression_opts, chunks=True)
        handle.create_dataset("locations_10x_in_20x", data=locs10_in20)
        handle.create_dataset("locations_10x_in_10x", data=locs10_in10)
        handle.create_dataset("HE_images_5x", data=images5, compression="gzip", compression_opts=compression_opts, chunks=True)
        handle.create_dataset("locations_5x_in_20x", data=locs5_in20)
        handle.create_dataset("locations_5x_in_5x", data=locs5_in5)
    os.replace(tmp_h5, output_h5_path)
    return True


def _read_sample_ids(sample_ids: Optional[Union[Sequence[str], str]]) -> Optional[Set[str]]:
    if sample_ids is None:
        return None
    if isinstance(sample_ids, (list, tuple, set)):
        return {str(x).strip() for x in sample_ids if str(x).strip()}
    value = str(sample_ids).strip()
    if os.path.isfile(value) and value.lower().endswith(".txt"):
        with open(value, "r", encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip() and not line.strip().startswith("#")}
    return {value}


def build_multiscale_h5_tree(
    tile_root: str,
    output_root: str,
    input_h5_name: str = "HE.h5",
    output_h5_name: str = "HE_noskip.h5",
    pad_mode: str = "skip",
    compression_opts: int = 4,
    fixed_step_20x: Optional[Tuple[int, int]] = (256, 256),
    sample_ids: Optional[Union[Sequence[str], str]] = None,
    skip_existing: bool = True,
) -> dict:
    """Build multiscale HDF5 files for all samples under a tile root."""
    tile_root_path = Path(tile_root)
    output_root_path = Path(output_root)
    selected = _read_sample_ids(sample_ids)
    sample_dirs = sorted(p for p in tile_root_path.iterdir() if p.is_dir())
    if selected is not None:
        sample_dirs = [p for p in sample_dirs if p.name in selected]

    done = skipped = failed = 0
    for sample_dir in sample_dirs:
        input_h5 = sample_dir / input_h5_name
        if not input_h5.exists():
            print(f"[skip] {sample_dir.name}: missing {input_h5_name}")
            skipped += 1
            continue
        output_h5 = output_root_path / sample_dir.name / output_h5_name
        try:
            changed = build_multiscale_h5_for_sample(
                input_h5=str(input_h5),
                output_h5=str(output_h5),
                pad_mode=pad_mode,
                compression_opts=compression_opts,
                fixed_step_20x=fixed_step_20x,
                skip_existing=skip_existing,
            )
            if changed:
                done += 1
                print(f"[ok] {sample_dir.name} -> {output_h5}")
            else:
                skipped += 1
                print(f"[skip] {sample_dir.name}: existing complete output")
        except Exception as exc:
            failed += 1
            print(f"[fail] {sample_dir.name}: {exc}")

    summary = {"done": done, "skipped": skipped, "failed": failed, "total": len(sample_dirs)}
    print(f"[summary] {summary}")
    return summary
