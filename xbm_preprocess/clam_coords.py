"""Run CLAM tissue segmentation and coordinate extraction."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm


def load_magnification_table(path: Optional[str]) -> Dict[str, float]:
    """Load a SampleID/Magnification table.

    The file may be CSV, TSV, XLSX, XLS, PKL, or PICKLE. If no table is provided,
    all slides are treated as 20x-equivalent inputs.
    """
    if path is None:
        return {}

    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(table_path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(table_path)
    elif suffix == ".tsv":
        df = pd.read_csv(table_path, sep="\t")
    else:
        df = pd.read_csv(table_path)

    required = {"SampleID", "Magnification"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Magnification table is missing columns: {sorted(missing)}")

    df = df[["SampleID", "Magnification"]].copy()
    df["SampleID"] = df["SampleID"].astype(str).str.strip()
    df["Magnification"] = pd.to_numeric(df["Magnification"], errors="coerce")
    df = df[df["Magnification"].isin([20.0, 40.0])]
    return dict(zip(df["SampleID"], df["Magnification"].astype(float)))


def infer_sample_ids(slide_root: str) -> List[str]:
    """Return sample folders under the raw WSI root."""
    root = Path(slide_root)
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _resolve_preset(segment_parameter: str, preset_map: Dict[str, str]) -> str:
    if segment_parameter in preset_map:
        return preset_map[segment_parameter]
    candidate = Path(segment_parameter)
    if candidate.exists():
        return str(candidate)
    raise ValueError(
        f"Unknown segment_parameter={segment_parameter!r}. "
        f"Known presets: {sorted(preset_map)} or pass a preset CSV path."
    )


def run_clam_for_sample(
    sample_id: str,
    slide_root: str,
    output_root: str,
    clam_script: str,
    preset_file: str,
    tile_size: int = 256,
    step_size: int = 256,
    magnification: float = 20.0,
    skip_existing: bool = False,
) -> None:
    """Run CLAM create_patches_fp.py for one sample."""
    slide_root = Path(slide_root)
    output_root = Path(output_root)
    sample_input = slide_root / sample_id

    tile_size_in = int(tile_size)
    step_size_in = int(step_size)
    magnification_dir = "Magnification20"
    if float(magnification) == 40.0:
        tile_size_in *= 2
        step_size_in *= 2
        magnification_dir = "Magnification40"
    elif float(magnification) != 20.0:
        raise ValueError(f"Only 20x and 40x inputs are supported, got {magnification}")

    sample_output = output_root / magnification_dir / sample_id
    expected_coord = sample_output / "patches" / "HE.h5"
    final_h5 = sample_output / "HE.h5"
    if skip_existing and (expected_coord.exists() or final_h5.exists()):
        print(f"[CLAM] skip existing sample: {sample_id}")
        return

    sample_output.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(clam_script),
        "--source",
        str(sample_input),
        "--save_dir",
        str(sample_output),
        "--patch_size",
        str(tile_size_in),
        "--step_size",
        str(step_size_in),
        "--preset",
        str(preset_file),
        "--seg",
        "--patch",
        "--stitch",
    ]
    subprocess.run(cmd, check=True)


def run_clam_for_slides(
    slide_root: str,
    output_root: str,
    clam_script: str,
    preset_map: Dict[str, str],
    segment_parameter: str = "tcga",
    tile_size: int = 256,
    step_size: int = 256,
    magnification_table: Optional[str] = None,
    sample_ids: Optional[Iterable[str]] = None,
    num_workers: int = 8,
    skip_existing: bool = False,
) -> None:
    """Run CLAM coordinate extraction for all sample folders."""
    preset_file = _resolve_preset(segment_parameter, preset_map)
    magnifications = load_magnification_table(magnification_table)
    sample_ids = list(sample_ids) if sample_ids is not None else infer_sample_ids(slide_root)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for sample_id in sample_ids:
            magnification = magnifications.get(str(sample_id), 20.0)
            future = executor.submit(
                run_clam_for_sample,
                sample_id=str(sample_id),
                slide_root=slide_root,
                output_root=output_root,
                clam_script=clam_script,
                preset_file=preset_file,
                tile_size=tile_size,
                step_size=step_size,
                magnification=magnification,
                skip_existing=skip_existing,
            )
            futures[future] = sample_id

        for future in tqdm(as_completed(futures), total=len(futures), desc="CLAM"):
            sample_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[CLAM] failed sample={sample_id}: {exc}")
