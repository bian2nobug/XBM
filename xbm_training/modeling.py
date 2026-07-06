from __future__ import annotations

import importlib
from typing import Any, Dict, Type

import torch.nn as nn


def import_class(class_path: str) -> Type[nn.Module]:
    """Import a class from 'package.module:ClassName' or 'package.module.ClassName'."""
    if ":" in class_path:
        module_name, class_name = class_path.split(":", 1)
    else:
        module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not issubclass(cls, nn.Module):
        raise TypeError(f"{class_path} is not a torch.nn.Module subclass")
    return cls


def build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    class_path = model_cfg.get("class_path")
    if not class_path:
        raise ValueError("model.class_path is required")
    params = model_cfg.get("params", {}) or {}
    return import_class(class_path)(**params)
