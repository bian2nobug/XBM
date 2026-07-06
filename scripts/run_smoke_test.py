#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    run([sys.executable, "scripts/make_smoke_data.py"])
    run([sys.executable, "scripts/run_train.py", "--config", "configs/smoke_test.yaml"])
    run([
        sys.executable,
        "scripts/run_evaluate.py",
        "--config",
        "configs/smoke_test.yaml",
        "--checkpoint",
        "/tmp/xbm_train_smoke/run/best_model.pt",
        "--split",
        "test",
    ])
    print("Smoke test finished. Results: /tmp/xbm_train_smoke/run")


if __name__ == "__main__":
    main()
