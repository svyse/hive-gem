from __future__ import annotations

import re
from pathlib import Path


def model_slug(model_id: str | None) -> str:
    raw = str(model_id or "default").strip().lower()
    raw = raw.replace("\\", "/")
    raw = re.sub(r"[^a-z0-9._/-]+", "-", raw)
    raw = raw.replace("/", "-")
    raw = raw.replace("_", "-")
    raw = re.sub(r"-+", "-", raw).strip("-.")
    return raw or "default"


def resolve_model_adapter_dir(configured_dir: str | Path | None, model_id: str | None) -> Path:
    base = Path(configured_dir or "").expanduser()
    if not str(base):
        return base

    # If the user explicitly points at a non-"latest" directory, respect it.
    if base.name and base.name != "latest":
        return base

    slug = model_slug(model_id)
    parent = base.parent if base.name == "latest" else base
    return parent / slug / "latest"


def resolve_model_adapter_root(configured_dir: str | Path | None, model_id: str | None) -> Path:
    return resolve_model_adapter_dir(configured_dir, model_id).parent


def resolve_training_state_file(configured_dir: str | Path | None, model_id: str | None) -> Path:
    return resolve_model_adapter_root(configured_dir, model_id) / "training_state.json"
