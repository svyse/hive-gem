from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from app.core.config import settings
    from app.llm.adapter_paths import resolve_model_adapter_dir, model_slug
except Exception:  # allow running from repo root before PYTHONPATH is set
    backend_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(backend_dir))
    from app.core.config import settings  # type: ignore
    from app.llm.adapter_paths import resolve_model_adapter_dir, model_slug  # type: ignore


def _normalise(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    try:
        p = Path(text).expanduser()
        if p.exists():
            text = str(p.resolve()).replace("\\", "/")
    except Exception:
        pass
    return text.rstrip("/").lower()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _base_model(adapter_dir: Path) -> str:
    for name in ("adapter_config.json", "adapter_metadata.json"):
        data = _read_json(adapter_dir / name)
        value = data.get("base_model_name_or_path") or data.get("base_model_name") or data.get("model_id")
        if value:
            return str(value)
    return ""


def main() -> int:
    model = str(os.getenv("LOCAL_LLM_MODEL") or getattr(settings, "local_llm_model", "") or "Qwen/Qwen2.5-Coder-1.5B-Instruct").strip()
    configured = Path(os.getenv("LOCAL_LLM_ADAPTER_DIR") or str(getattr(settings, "local_llm_adapter_dir", "") or "~/.agentic_hive_studio/local_llm/adapters/latest")).expanduser()
    resolved = resolve_model_adapter_dir(configured, model)

    candidates: List[Path] = []
    for p in (
        resolved,
        configured,
        configured.parent / model_slug(model) / "latest" if configured.name == "latest" else configured / model_slug(model) / "latest",
    ):
        if p not in candidates:
            candidates.append(p)

    print("model:", model)
    print("configured adapter dir:", configured)
    print("resolved model-specific adapter dir:", resolved)
    print("disable adapter env:", os.getenv("LOCAL_LLM_DISABLE_ADAPTER"))
    print("enable adapter env:", os.getenv("LOCAL_LLM_ENABLE_ADAPTER"))

    found = False
    compatible = False
    for p in candidates:
        cfg = p / "adapter_config.json"
        meta = p / "adapter_metadata.json"
        print("candidate:", p)
        print("  adapter_config.json:", cfg.exists())
        print("  adapter_metadata.json:", meta.exists())
        if not cfg.exists() and not meta.exists():
            continue
        found = True
        base = _base_model(p)
        print("  base_model_name_or_path:", base or "<not recorded>")
        if not base:
            print("  compatibility: unknown (legacy adapter; PEFT will validate tensor shapes on load)")
            compatible = True
        elif _normalise(base) == _normalise(model):
            print("  compatibility: OK")
            compatible = True
        else:
            print("  compatibility: MISMATCH")

    if not found:
        print("No adapter found. This is safe: the trainer will create a fresh model-specific adapter after training data is available.")
        return 0
    if not compatible:
        print("No compatible adapter found for the active model. Keep LOCAL_LLM_DISABLE_ADAPTER=true or train a fresh adapter for this model.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
