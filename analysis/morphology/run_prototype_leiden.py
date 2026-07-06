#!/usr/bin/env python3
"""Prototype-based Leiden clustering of tile-level morphology features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA, prototype K-means, and Leiden clustering on tile-level features."
    )
    parser.add_argument("--features", required=True, help="Torch .pt/.pth feature dictionary.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--h5-root",
        default=None,
        help="Optional root containing SampleID/HE_noskip.h5 for valid tile counts.",
    )
    parser.add_argument(
        "--input-layout",
        choices=["auto", "NDT", "NTD"],
        default="auto",
        help="Feature tensor layout. NDT means (sample, feature_dim, tiles).",
    )
    parser.add_argument("--feature-key", default=None, help="Optional feature key in torch dictionary.")
    parser.add_argument("--sample-id-key", default="sample_ids", help="Sample ID key in torch dictionary.")
    parser.add_argument("--pca-dim", type=int, default=20)
    parser.add_argument("--n-prototypes", type=int, default=1000)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--resolution", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n-representative", type=int, default=30)
    parser.add_argument("--nonzero-eps", type=float, default=1e-8)
    parser.add_argument("--make-umap", action="store_true", help="Write optional UMAP coordinates.")
    return parser.parse_args()


def load_feature_object(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("Feature file must contain a torch dictionary.")
    return obj


def find_feature_key(obj: dict[str, Any], requested_key: str | None) -> str:
    if requested_key:
        if requested_key not in obj:
            raise KeyError(f"Feature key {requested_key} not found.")
        return requested_key
    for key in ("feat_histo_NDT", "feat_histo_NTD", "features", "feat", "pyr_feat"):
        if key in obj:
            return key
    raise KeyError("Could not find feature tensor key. Use --feature-key.")


def to_numpy_ntd(features: Any, key: str, layout: str) -> np.ndarray:
    arr = torch.as_tensor(features).detach().cpu().float().numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {arr.shape}.")

    if layout == "NDT":
        return np.transpose(arr, (0, 2, 1))
    if layout == "NTD":
        return arr

    if key.endswith("_NDT"):
        return np.transpose(arr, (0, 2, 1))
    if key.endswith("_NTD"):
        return arr

    known_dims = {384, 512, 768, 1024, 1280, 1536, 2048, 4096}
    if arr.shape[1] in known_dims and arr.shape[2] not in known_dims:
        return np.transpose(arr, (0, 2, 1))
    if arr.shape[2] in known_dims and arr.shape[1] not in known_dims:
        return arr

    raise ValueError(
        f"Could not infer feature layout from shape {arr.shape}. "
        "Pass --input-layout NDT or --input-layout NTD."
    )


def load_features(path: Path, feature_key: str | None, sample_id_key: str, layout: str) -> tuple[np.ndarray, list[str]]:
    obj = load_feature_object(path)
    key = find_feature_key(obj, feature_key)
    features_ntd = to_numpy_ntd(obj[key], key, layout)
    if sample_id_key not in obj:
        raise KeyError(f"Sample ID key {sample_id_key} not found.")
    sample_ids = [str(x) for x in obj[sample_id_key]]
    if len(sample_ids) != features_ntd.shape[0]:
        raise ValueError(f"Number of sample IDs {len(sample_ids)} != N {features_ntd.shape[0]}.")
    return features_ntd, sample_ids


def valid_lengths_from_h5(h5_root: Path, sample_ids: list[str]) -> list[int]:
    lengths: list[int] = []
    for sample_id in sample_ids:
        h5_path = h5_root / sample_id / "HE_noskip.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing HDF5 file: {h5_path}")
        with h5py.File(h5_path, "r") as handle:
            dataset_name = None
            for name in ("locations_5x_in_20x", "locations", "coords"):
                if name in handle:
                    dataset_name = name
                    break
            if dataset_name is None:
                raise KeyError(f"{h5_path} has no locations_5x_in_20x, locations, or coords dataset.")
            lengths.append(int(handle[dataset_name].shape[0]))
    return lengths


def build_valid_mask(features_ntd: np.ndarray, sample_ids: list[str], h5_root: str | None, eps: float) -> np.ndarray:
    n_samples, n_tiles, _ = features_ntd.shape
    if h5_root:
        lengths = valid_lengths_from_h5(Path(h5_root), sample_ids)
        mask = np.zeros((n_samples, n_tiles), dtype=bool)
        for i, length in enumerate(lengths):
            mask[i, : min(length, n_tiles)] = True
        return mask
    return np.abs(features_ntd).sum(axis=2) > eps


def flatten_valid_tiles(features_ntd: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    sample_index, tile_index = np.where(valid_mask)
    flat = features_ntd[sample_index, tile_index]
    tile_meta = pd.DataFrame({"sample_index": sample_index, "tile_index": tile_index})
    return flat, tile_meta


def run_leiden(prototype_centers: np.ndarray, knn_k: int, resolution: float, seed: int) -> np.ndarray:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise ImportError("Install python-igraph and leidenalg to run Leiden clustering.") from exc

    n_nodes = prototype_centers.shape[0]
    if n_nodes < 2:
        return np.zeros(n_nodes, dtype=int)

    n_neighbors = min(knn_k + 1, n_nodes)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(prototype_centers)
    distances, indices = nn.kneighbors(prototype_centers)

    edges: list[tuple[int, int]] = []
    edge_set: set[tuple[int, int]] = set()
    weights: list[float] = []
    for source in range(n_nodes):
        for dist, target in zip(distances[source, 1:], indices[source, 1:]):
            if source == int(target):
                continue
            edge = (min(source, int(target)), max(source, int(target)))
            if edge in edge_set:
                continue
            edge_set.add(edge)
            edges.append(edge)
            weights.append(1.0 / (float(dist) + 1e-12))

    graph = ig.Graph(n=n_nodes, edges=edges, directed=False)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.asarray(partition.membership, dtype=int)


def save_cluster_outputs(
    out_dir: Path,
    sample_ids: list[str],
    valid_mask: np.ndarray,
    tile_meta: pd.DataFrame,
    tile_cluster: np.ndarray,
    tile_pca: np.ndarray,
    prototype_assign: np.ndarray,
    prototype_labels: np.ndarray,
    prototype_centers: np.ndarray,
    top_n: int,
) -> None:
    n_samples, n_tiles = valid_mask.shape
    cluster_nt = np.full((n_samples, n_tiles), -1, dtype=np.int32)
    cluster_nt[tile_meta["sample_index"].to_numpy(), tile_meta["tile_index"].to_numpy()] = tile_cluster
    np.save(out_dir / "tile_cluster_labels_NT.npy", cluster_nt)
    torch.save({"sample_ids": sample_ids, "tile_cluster_labels_NT": torch.from_numpy(cluster_nt)}, out_dir / "tile_cluster_labels_NT.pth")

    cluster_ids = sorted(np.unique(tile_cluster).tolist())
    rows = []
    for sample_idx, sample_id in enumerate(sample_ids):
        sample_labels = cluster_nt[sample_idx][cluster_nt[sample_idx] >= 0]
        row: dict[str, Any] = {"SampleID": sample_id, "n_valid_tiles": int(sample_labels.size)}
        for cluster_id in cluster_ids:
            count = int(np.sum(sample_labels == cluster_id))
            row[f"C{cluster_id}_count"] = count
            row[f"C{cluster_id}_prop"] = count / sample_labels.size if sample_labels.size else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "tile_cluster_composition_by_sample.csv", index=False)

    pd.DataFrame(
        {
            "Cluster": [f"C{x}" for x in cluster_ids],
            "n_tiles": [int(np.sum(tile_cluster == x)) for x in cluster_ids],
        }
    ).to_csv(out_dir / "tile_cluster_size.csv", index=False)

    pd.DataFrame(
        {
            "prototype_id": np.arange(len(prototype_labels)),
            "cluster": [f"C{x}" for x in prototype_labels],
            "n_assigned_tiles": np.bincount(prototype_assign, minlength=len(prototype_labels)),
        }
    ).to_csv(out_dir / "prototype_leiden_labels.csv", index=False)

    representative_rows = []
    sample_arr = tile_meta["sample_index"].to_numpy()
    tile_arr = tile_meta["tile_index"].to_numpy()
    for cluster_id in cluster_ids:
        idx = np.where(tile_cluster == cluster_id)[0]
        if idx.size == 0:
            continue
        center = tile_pca[idx].mean(axis=0, keepdims=True)
        distances = np.linalg.norm(tile_pca[idx] - center, axis=1)
        selected = idx[np.argsort(distances)[:top_n]]
        for rank, flat_idx in enumerate(selected, start=1):
            representative_rows.append(
                {
                    "Cluster": f"C{cluster_id}",
                    "rank": rank,
                    "SampleID": sample_ids[int(sample_arr[flat_idx])],
                    "sample_index": int(sample_arr[flat_idx]),
                    "tile_index": int(tile_arr[flat_idx]),
                    "prototype_id": int(prototype_assign[flat_idx]),
                }
            )
    pd.DataFrame(representative_rows).to_csv(out_dir / "representative_tiles_by_cluster.csv", index=False)

    np.save(out_dir / "prototype_centers_pca.npy", prototype_centers)


def maybe_write_umap(out_dir: Path, tile_pca: np.ndarray, tile_cluster: np.ndarray, tile_meta: pd.DataFrame, sample_ids: list[str]) -> None:
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install umap-learn or omit --make-umap.") from exc

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="euclidean", random_state=42)
    coords = reducer.fit_transform(tile_pca)
    df = tile_meta.copy()
    df["SampleID"] = [sample_ids[i] for i in df["sample_index"]]
    df["Cluster"] = [f"C{x}" for x in tile_cluster]
    df["UMAP1"] = coords[:, 0]
    df["UMAP2"] = coords[:, 1]
    df.to_csv(out_dir / "tile_umap_coordinates.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_ntd, sample_ids = load_features(
        Path(args.features),
        args.feature_key,
        args.sample_id_key,
        args.input_layout,
    )
    valid_mask = build_valid_mask(features_ntd, sample_ids, args.h5_root, args.nonzero_eps)
    flat_features, tile_meta = flatten_valid_tiles(features_ntd, valid_mask)
    if flat_features.shape[0] < 2:
        raise ValueError("Need at least two valid tiles for clustering.")

    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat_features)
    pca_dim = min(args.pca_dim, flat_scaled.shape[1], flat_scaled.shape[0])
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    tile_pca = pca.fit_transform(flat_scaled)

    n_prototypes = min(args.n_prototypes, tile_pca.shape[0])
    kmeans = MiniBatchKMeans(
        n_clusters=n_prototypes,
        random_state=args.seed,
        batch_size=min(8192, max(1024, n_prototypes * 4)),
        n_init=10,
    )
    prototype_assign = kmeans.fit_predict(tile_pca)
    prototype_labels = run_leiden(kmeans.cluster_centers_, args.knn_k, args.resolution, args.seed)
    tile_cluster = prototype_labels[prototype_assign]

    save_cluster_outputs(
        out_dir,
        sample_ids,
        valid_mask,
        tile_meta,
        tile_cluster,
        tile_pca,
        prototype_assign,
        prototype_labels,
        kmeans.cluster_centers_,
        args.top_n_representative,
    )

    joblib.dump(scaler, out_dir / "feature_scaler.joblib")
    joblib.dump(pca, out_dir / "pca_model.joblib")
    joblib.dump(kmeans, out_dir / "prototype_kmeans.joblib")

    params = vars(args).copy()
    params.update(
        {
            "n_samples": len(sample_ids),
            "feature_shape_NTD": list(features_ntd.shape),
            "n_valid_tiles": int(flat_features.shape[0]),
            "n_clusters": int(len(np.unique(tile_cluster))),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }
    )
    with open(out_dir / "run_parameters.json", "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)

    if args.make_umap:
        maybe_write_umap(out_dir, tile_pca, tile_cluster, tile_meta, sample_ids)

    print(f"Wrote morphology clustering outputs to {out_dir}")


if __name__ == "__main__":
    main()
