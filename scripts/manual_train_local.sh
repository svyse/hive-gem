#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/manual_train_local.py "$@"
