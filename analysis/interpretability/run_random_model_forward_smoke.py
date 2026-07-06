#!/usr/bin/env python3
"""Smoke test for XBM forward pass and FC-AttnPooling scale-fusion extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from _common import create_xbm_model, get_device, load_checkpoint_if_available, load_tensor_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one-sample XBM forward and extract scale-fusion weights.")
    p.add_argument("--data-path", required=True, help="Tensor file shaped [N, C, M, 21].")
    p.add_argument("--label-path", required=True, help="Label tensor file shaped [N].")
    p.add_argument("--sample-id-path", required=True, help="Text file with one SampleID per line.")
    p.add_argument("--sample-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    p.add_argument("--device", default="auto")
    p.add_argument("--split-dims", type=int, default=1536)
    p.add_argument("--clin-dim", type=int, default=57)
    p.add_argument("--class-dim", type=int, default=4)
    p.add_argument("--cross-embed-dim", type=int, default=256)
    p.add_argument("--cross-num-layers", type=int, default=2)
    p.add_argument("--cross-num-heads", type=int, default=2)
    p.add_argument("--joint-heads", type=int, default=4)
    p.add_argument("--strict-checkpoint", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    x, y, idx, _ = load_tensor_sample(args.data_path, args.label_path, args.sample_id_path, args.sample_id, device)

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

    captured = {}

    def hook_fn(module, inp, out):
        captured["scale_raw"] = out.detach().cpu()
        captured["scale_softmax"] = F.softmax(out, dim=2).squeeze(-1).detach().cpu()

    if not hasattr(model, "Scale_Fusion") or not hasattr(model.Scale_Fusion, "scorer"):
        raise RuntimeError("model.Scale_Fusion.scorer was not found; cannot extract FC-AttnPooling weights")

    handle = model.Scale_Fusion.scorer.register_forward_hook(hook_fn)
    with torch.no_grad():
        logits = model(x)
    handle.remove()

    w = captured["scale_softmax"][0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.sample_id}_scale_fusion.pth"

    torch.save(
        {
            "sample_id": args.sample_id,
            "sample_index": idx,
            "label": int(y.item()),
            "logits": logits.detach().cpu(),
            "scale_attn_raw": captured["scale_raw"],
            "scale_attn_softmax": captured["scale_softmax"],
            "scale_5x": w[:, 0:1].sum(dim=1),
            "scale_10x": w[:, 1:5].sum(dim=1),
            "scale_20x": w[:, 5:21].sum(dim=1),
        },
        out_path,
    )

    print("sample:", args.sample_id)
    print("input:", tuple(x.shape))
    print("label:", int(y.item()))
    print("logits:", tuple(logits.shape), logits.detach().cpu().numpy())
    print("pred:", int(logits.argmax(dim=1).item()) if logits.shape[-1] > 1 else float(logits.sigmoid().item() >= 0.5))
    print("scale_raw:", tuple(captured["scale_raw"].shape))
    print("scale_softmax:", tuple(captured["scale_softmax"].shape))
    print("scale weight M,N:", tuple(w.shape))
    print("5x mean:", float(w[:, 0:1].sum(dim=1).mean()))
    print("10x mean:", float(w[:, 1:5].sum(dim=1).mean()))
    print("20x mean:", float(w[:, 5:21].sum(dim=1).mean()))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
