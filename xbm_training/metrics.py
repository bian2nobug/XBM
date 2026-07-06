from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def binary_metrics(logits, labels, threshold: float = 0.5) -> Tuple[Dict[str, float], np.ndarray]:
    y_true = _to_numpy(labels).reshape(-1).astype(int)
    scores = 1.0 / (1.0 + np.exp(-_to_numpy(logits).reshape(-1)))
    y_pred = (scores >= threshold).astype(int)

    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        fpr, tpr, th = roc_curve(y_true, scores)
        best = int(np.argmax(tpr - fpr))
        best_thr = float(th[best])
        best_pred = (scores >= best_thr).astype(int)
        metrics.update({
            "best_threshold": best_thr,
            "acc_best": float(accuracy_score(y_true, best_pred)),
            "precision_best": float(precision_score(y_true, best_pred, zero_division=0)),
            "recall_best": float(recall_score(y_true, best_pred, zero_division=0)),
            "f1_best": float(f1_score(y_true, best_pred, zero_division=0)),
        })
    else:
        metrics["roc_auc"] = float("nan")
        metrics["best_threshold"] = float("nan")
    return metrics, scores


def multiclass_metrics(logits, labels) -> Tuple[Dict[str, float], np.ndarray]:
    y_true = _to_numpy(labels).reshape(-1).astype(int)
    z = _to_numpy(logits)
    z = z - z.max(axis=1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)
    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        macro_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float("nan")
    try:
        micro_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="micro"))
    except ValueError:
        micro_auc = float("nan")
    metrics["roc_auc_ovr_macro"] = macro_auc
    metrics["roc_auc_ovr_micro"] = micro_auc
    metrics["macro_auc"] = macro_auc
    metrics["micro_auc"] = micro_auc
    return metrics, probs


def regression_metrics(preds, labels) -> Dict[str, float]:
    y_true = _to_numpy(labels).reshape(-1).astype(float)
    y_pred = _to_numpy(preds).reshape(-1).astype(float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
