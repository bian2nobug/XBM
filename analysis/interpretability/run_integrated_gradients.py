#!/usr/bin/env python3
"""Run one-sample Integrated Gradients for the XBM tensor input."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import create_xbm_model, get_device, load_checkpoint_if_available, load_tensor_sample
from aggregators import MultiScaleMILReducer
from ig_core import IGAnalyzer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute XBM Integrated Gradients for one sample.")
    p.add_argument("--data-path", required=True, help="Tensor file shaped [N, C, M, 21].")
    p.add_argument("--label-path", required=True, help="Label tensor file shaped [N].")
    p.add_argument("--sample-id-path", required=True, help="Text file with one SampleID per line.")
    p.add_argument("--sample-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    p.add_argument("--device", default="auto")
    p.add_argument("--target", type=int, default=None, help="Target class. Defaults to the sample label.")
    p.add_argument("--n-steps", type=int, default=16, help="Number of IG integration steps.")
    p.add_argument("--split-dims", type=int, default=1536)
    p.add_argument("--clin-dim", type=int, default=57)
    p.add_argument("--class-dim", type=int, default=4)
    p.add_argument("--cross-embed-dim", type=int, default=256)
    p.add_argument("--cross-num-layers", type=int, default=2)
    p.add_argument("--cross-num-heads", type=int, default=2)
    p.add_argument("--joint-heads", type=int, default=4)
    p.add_argument("--strict-checkpoint", action="store_true")
    p.add_argument("--return-raw", action="store_true", help="Save full attribution tensor. Large output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    x, y, idx, _ = load_tensor_sample(args.data_path, args.label_path, args.sample_id_path, args.sample_id, device)
    target = int(y.item()) if args.target is None else int(args.target)

    model = create_xbm_model(
        split_dims=args.split_dims,
        clin_dim=args.clin_dim,
        class_dim=args.class_dim,
        cross_num_layers=args.cross_num_layers,
        cross_embed_dim=args.cross_embed_dim,
        cross_num_heads=args.cross_num_heads,
        joint_heads=args.joint_heads,
        gradient=False,
        device=device,
    )
    load_checkpoint_if_available(model, args.checkpoint, strict=args.strict_checkpoint)
    model.eval()

    baseline = x.clone()
    baseline[:, :args.split_dims, :, :] = 0.0

    reducer = MultiScaleMILReducer(
        pathology_dim=args.split_dims,
        clinical_dim=args.clin_dim,
        scale_splits=[1, 5, 21],
        scale_names=["5x", "10x", "20x"],
    )
    analyzer = IGAnalyzer(model=model, reducer=reducer, n_steps=args.n_steps, device=str(device))

    print("sample:", args.sample_id)
    print("input:", tuple(x.shape))
    print("target:", target)
    print("n_steps:", args.n_steps)
    print("running IG...")
    results = analyzer.analyze(x, baseline=baseline, target=target, return_raw=args.return_raw)

    inst = results["instance_attribution"].detach().cpu()
    scale = results["scale_attribution"].detach().cpu()
    feat = results["feature_attribution"].detach().cpu()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.sample_id}_ig.pth"
    save_obj = {
        "sample_id": args.sample_id,
        "sample_index": idx,
        "label": int(y.item()),
        "target": target,
        "n_steps": args.n_steps,
        "instance_attribution": inst[0],
        "scale_attribution": scale[0],
        "scale_names": ["5x", "10x", "20x"],
        "feature_attribution": feat[0],
    }
    if args.return_raw and "full_attribution" in results:
        save_obj["full_attribution"] = results["full_attribution"].detach().cpu()
    torch.save(save_obj, out_path)

    print("instance_attribution:", tuple(inst.shape))
    print("scale_attribution:", tuple(scale.shape), scale.numpy())
    print("feature_attribution:", tuple(feat.shape))
    print("instance min/max:", float(inst.min()), float(inst.max()))
    print("instance abs sum:", float(inst.abs().sum()))
    print("scale abs sum:", float(scale.abs().sum()))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
