#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/check_local_cuda.py
