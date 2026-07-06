#!/usr/bin/env python3
"""Compute cross-attention Attention x Gradient attribution for one XBM sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import create_xbm_model, get_device, load_checkpoint_if_available, load_tensor_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute cross-attention AxG for one sample.")
    p.add_argument("--data-path", required=True, help="Tensor file shaped [N, C, M, 21].")
    p.add_argument("--label-path", required=True, help="Label tensor file shaped [N].")
    p.add_argument("--sample-id-path", required=True, help="Text file with one SampleID per line.")
    p.add_argument("--sample-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    p.add_argument("--device", default="auto")
    p.add_argument("--target", type=int, default=None, help="Target class. Defaults to sample label.")
    p.add_argument("--split-dims", type=int, default=1536)
    p.add_argument("--clin-dim", type=int, default=57)
    p.add_argument("--class-dim", type=int, default=4)
    p.add_argument("--cross-embed-dim", type=int, default=256)
    p.add_argument("--cross-num-layers", type=int, default=2)
    p.add_argument("--cross-num-heads", type=int, default=2)
    p.add_argument("--joint-heads", type=int, default=4)
    p.add_argument("--strict-checkpoint", action="store_true")
    return p.parse_args()


def _target_score(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] == 1:
        sign = 2 * target.float() - 1
        return (logits.squeeze(-1) * sign).sum()
    return logits.gather(1, target.view(-1, 1)).sum()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    x, y, idx, _ = load_tensor_sample(args.data_path, args.label_path, args.sample_id_path, args.sample_id, device)
    target = y.clone() if args.target is None else torch.tensor([args.target], dtype=torch.long, device=device)

    model = create_xbm_model(
        split_dims=args.split_dims,
        clin_dim=args.clin_dim,
        class_dim=args.class_dim,
        cross_num_layers=args.cross_num_layers,
        cross_embed_dim=args.cross_embed_dim,
        cross_num_heads=args.cross_num_heads,
        joint_heads=args.joint_heads,
        gradient=True,
        device=device,
    )
    load_checkpoint_if_available(model, args.checkpoint, strict=args.strict_checkpoint)
    model.eval()
    model.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(True):
        logits = model(x)
        score = _target_score(logits, target)

    cross_attn = model.attn_weights[-1] if isinstance(model.attn_weights, (list, tuple)) else model.attn_weights
    enc_attn = getattr(model, "enc_attn", None)
    grad_inputs = [cross_attn]
    if enc_attn is not None:
        grad_inputs.append(enc_attn)

    grads = torch.autograd.grad(
        outputs=score,
        inputs=grad_inputs,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    cross_grad = grads[0]
    enc_grad = grads[1] if len(grads) > 1 else None

    if cross_grad is None:
        raise RuntimeError("cross attention gradient is None")

    cross_axg = (cross_attn * cross_grad).detach().cpu()
    instance_axg = cross_axg.abs().mean(dim=(0, 1, 2))
    instance_axg = instance_axg / max(float(instance_axg.max()), 1e-8)

    save_obj = {
        "sample_id": args.sample_id,
        "sample_index": idx,
        "label": int(y.item()),
        "target": int(target.item()),
        "logits": logits.detach().cpu(),
        "cross_attention_weights": cross_attn.detach().cpu(),
        "cross_attention_gradients": cross_grad.detach().cpu(),
        "cross_axg": cross_axg,
        "instance_axg": instance_axg,
    }
    if enc_attn is not None:
        save_obj["enc_attention_weights"] = enc_attn.detach().cpu()
    if enc_grad is not None:
        enc_axg = (enc_attn * enc_grad).detach().cpu()
        save_obj["enc_attention_gradients"] = enc_grad.detach().cpu()
        save_obj["enc_axg"] = enc_axg

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.sample_id}_cross_axg.pth"
    torch.save(save_obj, out_path)

    print("sample:", args.sample_id)
    print("input:", tuple(x.shape))
    print("label:", int(y.item()))
    print("target:", int(target.item()))
    print("logits:", tuple(logits.shape), logits.detach().cpu().numpy())
    print("pred:", int(logits.argmax(dim=1).item()) if logits.shape[-1] > 1 else float(logits.sigmoid().item() >= 0.5))
    print("cross_attn:", tuple(cross_attn.shape), "requires_grad=", cross_attn.requires_grad)
    print("cross_grad:", tuple(cross_grad.shape))
    if enc_attn is not None:
        print("enc_attn:", tuple(enc_attn.shape), "requires_grad=", enc_attn.requires_grad)
        if enc_grad is not None:
            print("enc_grad:", tuple(enc_grad.shape))
    print("cross_axg:", tuple(cross_axg.shape), "min/max:", float(cross_axg.min()), float(cross_axg.max()))
    print("instance_axg:", tuple(instance_axg.shape), "min/max:", float(instance_axg.min()), float(instance_axg.max()))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
