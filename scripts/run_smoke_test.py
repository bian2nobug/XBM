#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate toy data, train a small smoke model, and evaluate it.")
    p.add_argument("--config", default="configs/smoke_test.yaml", help="Smoke-test training config.")
    p.add_argument("--checkpoint", default="/tmp/xbm_train_smoke/run/best_model.pt", help="Checkpoint path produced by training.")
    p.add_argument("--split", default="test", help="Split to evaluate after training.")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    args = parse_args()
    run([sys.executable, "scripts/make_smoke_data.py"])
    run([sys.executable, "scripts/run_train.py", "--config", args.config])
    run([
        sys.executable,
        "scripts/run_evaluate.py",
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--split",
        args.split,
    ])
    print("Smoke test finished. Results: /tmp/xbm_train_smoke/run")


if __name__ == "__main__":
    main()
