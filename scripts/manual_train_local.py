#!/usr/bin/env python3
"""Run one manual local LoRA training pass.

Usage:
  python scripts/manual_train_local.py --force

The trainer reads backend/.env, consumes conversation/training_examples from
MEMORY_DB_PATH, and writes the latest adapter under LOCAL_LLM_ADAPTER_DIR.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Agentic Hive local LoRA training pass")
    parser.add_argument("--force", action="store_true", help="Train even if fewer than LOCAL_TRAINING_MIN_NEW_PAIRS are available")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    backend = repo_root / "backend"
    sys.path.insert(0, str(backend))

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.force:
        os.environ["FORCE_TRAIN"] = "1"

    from app.training.worker import main as worker_main

    return int(worker_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
