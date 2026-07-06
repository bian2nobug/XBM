"""Minimal Prov-GigaPath feature extraction for multiscale HDF5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class H5TileDataset(Dataset):
    """Lazy HDF5 tile dataset supporting NHWC or NCHW image arrays."""

    def __init__(
        self,
        h5_path: str,
        h5_key: str,
        image_size: int = 224,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.h5_path = str(h5_path)
        self.h5_key = h5_key
        self.image_size = int(image_size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        with h5py.File(self.h5_path, "r") as handle:
            self.key = self._resolve_key(handle, h5_key)
            self.length = int(handle[self.key].shape[0])

    @staticmethod
    def _resolve_key(handle: h5py.File, h5_key: str) -> str:
        if h5_key in handle:
            return h5_key
        matches = [key for key in handle.keys() if key.lower() == h5_key.lower()]
        if matches:
            return matches[0]
        raise KeyError(f"HDF5 key {h5_key!r} not found. Available keys: {list(handle.keys())}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        with h5py.File(self.h5_path, "r") as handle:
            tile = torch.as_tensor(handle[self.key][index])

        if tile.ndim != 3:
            raise ValueError(f"Tile must be 3D, got shape={tuple(tile.shape)}")
        if tile.shape[-1] in (1, 3):
            tile = tile.permute(2, 0, 1)
        elif tile.shape[0] not in (1, 3):
            raise ValueError(f"Cannot infer channel dimension from tile shape={tuple(tile.shape)}")

        tile = tile.float()
        if tile.max() > 2.0:
            tile = tile / 255.0
        if tile.shape[0] == 1:
            tile = tile.repeat(3, 1, 1)

        tile = F.interpolate(
            tile.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return (tile - self.mean) / self.std


def load_provgigapath_model(
    checkpoint_path: str,
    device: str = "cuda:0",
    model_name: str = "hf_hub:prov-gigapath/prov-gigapath",
) -> torch.nn.Module:
    """Load Prov-GigaPath with timm from a local checkpoint."""
    from timm import create_model

    model = create_model(model_name, pretrained=False, checkpoint_path=checkpoint_path)
    model.to(device)
    model.eval()
    return model


def extract_features_from_h5(
    h5_path: str,
    output_path: str,
    checkpoint_path: str,
    h5_key: str,
    device: str = "cuda:0",
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = 224,
    model_name: str = "hf_hub:prov-gigapath/prov-gigapath",
) -> np.ndarray:
    """Extract Prov-GigaPath features from one HDF5 file and save as .npy."""
    dataset = H5TileDataset(h5_path=h5_path, h5_key=h5_key, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    model = load_provgigapath_model(checkpoint_path=checkpoint_path, device=device, model_name=model_name)

    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{Path(h5_path).parent.name}/{h5_key}"):
            output = model(batch.to(device, non_blocking=True))
            if isinstance(output, (tuple, list)):
                output = output[0]
            features.append(output.detach().cpu().float().numpy())

    feature_array = np.concatenate(features, axis=0) if features else np.empty((0, 1536), dtype=np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, feature_array)
    print(f"[ok] {Path(h5_path).parent.name}/{h5_key}: {feature_array.shape} -> {output_path}")
    return feature_array


def read_sample_ids(sample_ids: Optional[Union[Sequence[str], str]]) -> Optional[List[str]]:
    if sample_ids is None:
        return None
    if isinstance(sample_ids, (list, tuple, set)):
        return [str(x).strip() for x in sample_ids if str(x).strip()]
    value = str(sample_ids).strip()
    if Path(value).is_file() and value.lower().endswith(".txt"):
        with open(value, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip() and not line.strip().startswith("#")]
    return [value]


def find_h5_files(input_root: str, h5_name: str, sample_ids: Optional[Iterable[str]] = None) -> List[Path]:
    root = Path(input_root)
    selected = set(sample_ids) if sample_ids else None
    files = []
    for sample_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if selected is not None and sample_dir.name not in selected:
            continue
        h5_path = sample_dir / h5_name
        if h5_path.is_file():
            files.append(h5_path)
    return files


def extract_feature_tree(
    input_root: str,
    output_root: str,
    checkpoint_path: str,
    h5_key: str,
    h5_name: str = "HE_noskip.h5",
    output_suffix: str = "prov_gigapath_feature.npy",
    device: str = "cuda:0",
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = 224,
    model_name: str = "hf_hub:prov-gigapath/prov-gigapath",
    sample_ids: Optional[Union[Sequence[str], str]] = None,
    skip_existing: bool = True,
) -> dict:
    """Extract one HDF5 key from all sample folders."""
    selected = read_sample_ids(sample_ids)
    h5_files = find_h5_files(input_root, h5_name=h5_name, sample_ids=selected)
    ok = skipped = failed = 0
    for h5_path in h5_files:
        sample_id = h5_path.parent.name
        output_path = Path(output_root) / sample_id / f"{h5_key}_{output_suffix}"
        if skip_existing and output_path.exists():
            skipped += 1
            print(f"[skip] {sample_id}/{h5_key}: existing {output_path}")
            continue
        try:
            extract_features_from_h5(
                h5_path=str(h5_path),
                output_path=str(output_path),
                checkpoint_path=checkpoint_path,
                h5_key=h5_key,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=image_size,
                model_name=model_name,
            )
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"[fail] {sample_id}/{h5_key}: {exc}")

    summary = {"ok": ok, "skipped": skipped, "failed": failed, "total": len(h5_files), "h5_key": h5_key}
    print(f"[summary] {summary}")
    return summary


def extract_multiscale_feature_roots(
    input_root: str,
    feat5_root: str,
    feat10_root: str,
    feat20_root: str,
    checkpoint_path: str,
    h5_name: str = "HE_noskip.h5",
    output_suffix: str = "prov_gigapath_feature.npy",
    device: str = "cuda:0",
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = 224,
    model_name: str = "hf_hub:prov-gigapath/prov-gigapath",
    sample_ids: Optional[Union[Sequence[str], str]] = None,
    skip_existing: bool = True,
) -> dict:
    """Extract Prov-GigaPath features for 5x, 10x, and 20x HDF5 image keys."""
    outputs = {
        "HE_images_5x": feat5_root,
        "HE_images_10x": feat10_root,
        "HE_images_20x": feat20_root,
    }
    summaries = {}
    for h5_key, output_root in outputs.items():
        summaries[h5_key] = extract_feature_tree(
            input_root=input_root,
            output_root=output_root,
            checkpoint_path=checkpoint_path,
            h5_key=h5_key,
            h5_name=h5_name,
            output_suffix=output_suffix,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            model_name=model_name,
            sample_ids=sample_ids,
            skip_existing=skip_existing,
        )
    return summaries
