#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../backend"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="0"
if [[ "${1:-}" == "--force" || "${1:-}" == "-Force" ]]; then
  export FORCE_TRAIN=1
fi
python -m app.training.worker
