#!/usr/bin/env python
"""Run the XBM WSI preprocessing pipeline from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XBM WSI preprocessing.")
    parser.add_argument("--config", required=True, help="Path to preprocess YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from xbm_preprocess.pipeline import run_preprocess_pipeline

    run_preprocess_pipeline(args.config)


if __name__ == "__main__":
    main()
