#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xbm_training.config import load_config
from xbm_training.trainer import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained XBM checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt or a state_dict.")
    parser.add_argument("--split", default="test", help="Split name in the config data table.")
    args = parser.parse_args()

    metrics = run_evaluation(load_config(args.config), args.checkpoint, split=args.split)
    print(metrics)


if __name__ == "__main__":
    main()
