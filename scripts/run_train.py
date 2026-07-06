#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xbm_training.config import load_config
from xbm_training.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XBM downstream prediction model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    results = run_training(load_config(args.config))
    print(results)


if __name__ == "__main__":
    main()
