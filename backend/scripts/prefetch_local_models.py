from __future__ import annotations

import os
from pathlib import Path

# Keep cache consistent with backend .env.
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    # Keep this commented until you want to prefetch the code/training model too.
    # "Qwen/Qwen2.5-Coder-1.5B-Instruct",
]

for model_id in MODELS:
    print(f"Downloading {model_id} into HF cache...")
    path = snapshot_download(
        repo_id=model_id,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Downloaded {model_id} -> {path}")

print("Done. Restart FastAPI, then ask Q&A again.")
