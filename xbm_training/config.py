from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return cfg


def require_section(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = cfg.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing required config section: {name}")
    return section
