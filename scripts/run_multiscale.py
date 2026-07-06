#!/usr/bin/env python
"""Run XBM multiscale feature construction from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XBM multiscale construction.")
    parser.add_argument("--config", required=True, help="Path to multiscale YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned paths without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from xbm_multiscale.pipeline import run_multiscale_pipeline

    run_multiscale_pipeline(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
