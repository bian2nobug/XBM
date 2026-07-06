from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_datasets
from .metrics import binary_metrics, multiclass_metrics, regression_metrics
from .modeling import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _auto_class_weights(labels: Sequence[int], device: torch.device) -> Optional[torch.Tensor]:
    if not labels:
        return None
    arr = np.asarray(labels, dtype=np.int64)
    n_classes = int(arr.max()) + 1
    counts = np.bincount(arr, minlength=n_classes).astype(np.float32)
    if np.any(counts == 0):
        return None
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _criterion(
    task_type: str,
    train_cfg: Optional[dict] = None,
    train_dataset=None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    train_cfg = train_cfg or {}
    if task_type == "binary":
        return nn.BCEWithLogitsLoss()
    if task_type == "multiclass":
        label_smoothing = float(train_cfg.get("label_smoothing", 0.0) or 0.0)
        class_weights = train_cfg.get("class_weights")
        weight_tensor = None
        if class_weights == "auto" and train_dataset is not None and device is not None:
            weight_tensor = _auto_class_weights([int(r.label) for r in train_dataset.records], device)
        elif isinstance(class_weights, (list, tuple)) and device is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    if task_type == "regression":
        return nn.MSELoss()
    raise ValueError(f"Unsupported task_type: {task_type}")


def _prepare_loss(logits: torch.Tensor, labels: torch.Tensor, task_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if task_type == "binary":
        return logits.view(-1), labels.float().view(-1)
    if task_type == "multiclass":
        return logits, labels.long().view(-1)
    return logits.view(-1), labels.float().view(-1)


def train_one_epoch(model, loader, optimizer, criterion, device, task_type: str) -> float:
    model.train()
    losses = []
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss_input, target = _prepare_loss(logits, y, task_type)
        loss = criterion(loss_input, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model, loader, criterion, device, task_type: str) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    model.eval()
    losses = []
    logits_all = []
    labels_all = []
    sample_ids = []
    for x, y, sid in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_input, target = _prepare_loss(logits, y, task_type)
        loss = criterion(loss_input, target)
        losses.append(float(loss.detach().cpu()))
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())
        sample_ids.extend(list(sid))

    logits_cat = torch.cat(logits_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    avg_loss = float(np.mean(losses)) if losses else float("nan")
    if task_type == "binary":
        metrics, scores = binary_metrics(logits_cat, labels_cat)
        pred_df = pd.DataFrame({"SampleID": sample_ids, "label": labels_cat.numpy().reshape(-1), "prediction": scores})
    elif task_type == "multiclass":
        metrics, probs = multiclass_metrics(logits_cat, labels_cat)
        pred_df = pd.DataFrame({"SampleID": sample_ids, "label": labels_cat.numpy().reshape(-1)})
        for i in range(probs.shape[1]):
            pred_df[f"prob_class_{i}"] = probs[:, i]
        pred_df["prediction"] = probs.argmax(axis=1)
    else:
        metrics = regression_metrics(logits_cat, labels_cat)
        pred_df = pd.DataFrame({
            "SampleID": sample_ids,
            "label": labels_cat.numpy().reshape(-1),
            "prediction": logits_cat.numpy().reshape(-1),
        })
    metrics["loss"] = avg_loss
    return avg_loss, metrics, pred_df


def run_training(cfg: dict) -> Dict[str, object]:
    train_cfg = cfg.get("training", {})
    task_type = train_cfg.get("task_type", "binary")
    seed = int(train_cfg.get("seed", cfg.get("data", {}).get("seed", 8766)))
    set_seed(seed)

    output_dir = Path(train_cfg.get("output_dir", "runs/xbm_task"))
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets, meta = build_datasets(cfg)
    if "train" not in datasets:
        raise ValueError("A train split is required")
    if "val" not in datasets:
        raise ValueError("A val split is required")

    device = torch.device(train_cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.get("model", {})).to(device)
    criterion = _criterion(task_type, train_cfg=train_cfg, train_dataset=datasets["train"], device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = _make_loader(datasets["train"], batch_size, True, num_workers)
    val_loader = _make_loader(datasets["val"], batch_size, False, num_workers)
    test_loader = _make_loader(datasets["test"], batch_size, False, num_workers) if "test" in datasets else None

    best_metric = -float("inf")
    best_epoch = -1
    history = []
    monitor = train_cfg.get("monitor")
    if monitor is None:
        monitor = "roc_auc" if task_type in ("binary", "multiclass") else "-loss"

    for epoch in range(1, int(train_cfg.get("epochs", 50)) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, task_type)
        val_loss, val_metrics, val_pred = evaluate(model, val_loader, criterion, device, task_type)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        val_pred.to_csv(output_dir / "predictions_val.csv", index=False)

        score = -val_loss if monitor == "-loss" else float(val_metrics.get(monitor, -float("inf")))
        if not np.isfinite(score):
            score = -val_loss
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "meta": meta,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                output_dir / "best_model.pt",
            )

    torch.save({"model_state_dict": model.state_dict(), "config": cfg, "meta": meta}, output_dir / "last_model.pt")

    results = {"best_epoch": best_epoch, "best_score": best_metric, "meta": meta}
    if test_loader is not None:
        checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, test_metrics, test_pred = evaluate(model, test_loader, criterion, device, task_type)
        test_pred.to_csv(output_dir / "predictions_test.csv", index=False)
        results["test_metrics"] = test_metrics

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def run_evaluation(cfg: dict, checkpoint_path: str, split: str = "test") -> Dict[str, float]:
    train_cfg = cfg.get("training", {})
    task_type = train_cfg.get("task_type", "binary")
    output_dir = Path(train_cfg.get("output_dir", "runs/xbm_task"))
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets, _ = build_datasets(cfg)
    if split not in datasets:
        raise ValueError(f"Split {split!r} was not found in the configured data")
    device = torch.device(train_cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.get("model", {})).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)

    loader = _make_loader(
        datasets[split],
        int(train_cfg.get("batch_size", 1)),
        False,
        int(train_cfg.get("num_workers", 0)),
    )
    _, metrics, pred_df = evaluate(model, loader, _criterion(task_type, train_cfg=train_cfg, device=device), device, task_type)
    pred_df.to_csv(output_dir / f"predictions_{split}.csv", index=False)
    with open(output_dir / f"metrics_{split}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics
