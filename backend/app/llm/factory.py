from __future__ import annotations

import gc
import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

from app.core.config import settings

log = logging.getLogger(__name__)

# Startup-only provider selection.
# Edit backend/.env LLM_BACKEND=local|gemini|openai|claude and restart FastAPI.
# There is intentionally no dashboard/runtime hot-swap path in this build.
_LLM: Optional[Any] = None
_LLM_LOCK = threading.RLock()
_LLM_EPOCH = 0
_ALLOWED_BACKENDS = {"local", "openai", "gemini", "claude"}
_DASHBOARD_BACKENDS = ("local", "openai", "gemini")


def _normalize_backend(name: str | None) -> str:
    backend = (name or "").strip().lower()
    if backend not in _ALLOWED_BACKENDS:
        return "local"
    return backend


def _startup_backend() -> str:
    return _normalize_backend(getattr(settings, "llm_backend", "local"))


def _model_name(name: str) -> str:
    backend = _normalize_backend(name)
    if backend == "openai":
        return str(getattr(settings, "openai_model", "") or "")
    if backend == "gemini":
        return str(getattr(settings, "gemini_model", "") or "")
    if backend == "claude":
        return str(getattr(settings, "claude_model", "") or "")
    return str(getattr(settings, "local_llm_model", "") or "")


def _lightweight_available(name: str) -> bool:
    """Cheap availability check: never import torch, probe CUDA, or call networks."""
    backend = _normalize_backend(name)
    if backend == "gemini":
        return bool((getattr(settings, "gemini_api_key", None) or "").strip())
    if backend == "openai":
        return bool((getattr(settings, "openai_api_key", None) or "").strip())
    if backend == "claude":
        return bool((getattr(settings, "anthropic_api_key", None) or "").strip())
    if backend == "local":
        return True
    return False


def _status_for_backend(backend: str) -> Dict[str, Any]:
    b = _normalize_backend(backend)
    allowed = list(_DASHBOARD_BACKENDS)
    if b not in allowed and b == "claude":
        allowed.append("claude")
    return {
        "active_backend": b,
        "allowed_backends": allowed,
        "available": {name: _lightweight_available(name) for name in allowed},
        "models": {name: _model_name(name) for name in allowed},
        "fallback_enabled": bool(getattr(settings, "llm_fallback_to_openai", False)),
        "fallback_backends": list(getattr(settings, "llm_fallback_backends", []) or []),
        "message": (
            "Startup-only provider selection is active. Edit backend/.env LLM_BACKEND "
            "and restart FastAPI to change providers. Dashboard hot-swap is disabled."
        ),
    }


@contextmanager
def use_llm_backend(_backend: str | None):
    """Compatibility no-op.

    Older routes/run-manager code may pass a per-request backend. In this build
    those overrides are intentionally ignored so Q&A/code always use the single
    startup provider selected by LLM_BACKEND in backend/.env.
    """
    yield


def request_backend() -> Optional[str]:
    return None


def active_backend() -> str:
    return _startup_backend()


def get_backend_status() -> Dict[str, Any]:
    return _status_for_backend(active_backend())


def set_active_backend(name: str) -> Dict[str, Any]:
    requested = _normalize_backend(name)
    current = active_backend()
    log.info(
        "Runtime backend switch ignored: requested=%s current=%s. Startup-only mode is enabled.",
        requested,
        current,
    )
    status = get_backend_status()
    status["message"] = (
        f"Runtime switching is disabled. Current startup backend is '{current}'. "
        f"To use '{requested}', set LLM_BACKEND={requested} in backend/.env and restart FastAPI."
    )
    return status


def _build_client(name: str) -> Any:
    backend = _normalize_backend(name)
    if backend == "local":
        from app.llm.local_hf_client import LocalHFClient

        return LocalHFClient()
    if backend == "openai":
        from app.llm.openai_client import OpenAIClient

        return OpenAIClient()
    if backend == "gemini":
        from app.llm.gemini_client import GeminiClient

        return GeminiClient()
    if backend == "claude":
        from app.llm.claude_client import ClaudeClient

        return ClaudeClient()
    raise ValueError(f"Unsupported backend: {name}")


class LazyLLMClient:
    """Delay fallback provider construction until it is actually invoked."""

    def __init__(self, backend: str, builder: Callable[[str], Any] = _build_client) -> None:
        self.backend = _normalize_backend(backend)
        self._builder = builder
        self._client: Any = None
        self._lock = threading.RLock()

    def _get(self) -> Any:
        with self._lock:
            if self._client is None:
                log.info("Building fallback LLM backend lazily: %s", self.backend)
                self._client = self._builder(self.backend)
            return self._client

    def is_available(self) -> bool:
        if self._client is None:
            return _lightweight_available(self.backend)
        return _client_available(self._client)

    def runtime_status(self) -> Dict[str, Any]:
        return self._get().runtime_status()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._get(), attr)


def _fallback_order(primary_backend: str) -> List[str]:
    order: List[str] = []
    seen = {primary_backend}
    raw = list(getattr(settings, "llm_fallback_backends", []) or [])
    if not raw:
        raw = ["openai", "gemini", "claude", "local"]
    for item in raw:
        name = _normalize_backend(str(item))
        if name in seen:
            continue
        seen.add(name)
        order.append(name)
    return order


def _client_available(client: Any) -> bool:
    try:
        fn = getattr(client, "is_available", None)
        if callable(fn):
            return bool(fn())
    except Exception:
        return False
    return True


def _build_selected_client(backend: str) -> Any:
    primary_backend = _normalize_backend(backend)
    primary = _build_client(primary_backend)

    if bool(getattr(settings, "llm_fallback_to_openai", False)):
        fallback_names = _fallback_order(primary_backend)
        fallbacks = [LazyLLMClient(name) for name in fallback_names]
        if fallbacks:
            from app.llm.fallback_client import FallbackLLMClient

            chain = [primary_backend, *fallback_names]
            log.info("Using startup LLM fallback chain: %s", " -> ".join(chain))
            return FallbackLLMClient(primary=primary, fallbacks=fallbacks)

    return primary


def get_llm_client() -> Any:
    """Return the single startup-selected LLM client."""
    global _LLM
    backend = active_backend()
    with _LLM_LOCK:
        if _LLM is not None and getattr(_LLM, "_factory_backend", backend) == backend:
            return _LLM

    client = _build_selected_client(backend)
    try:
        setattr(client, "_factory_backend", backend)
    except Exception:
        pass

    with _LLM_LOCK:
        _LLM = client
        return _LLM


def _drop_cached_client(*, eager_gc: bool = False) -> None:
    global _LLM, _LLM_EPOCH
    with _LLM_LOCK:
        _LLM_EPOCH += 1
        _LLM = None
    if eager_gc:
        try:
            gc.collect()
        except Exception:
            pass


def reload_llm_client(*, eager: bool = True) -> Any:
    _drop_cached_client(eager_gc=True)
    if eager:
        return get_llm_client()
    return None


def get_local_runtime_status(*, deep: bool = False) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "backend": "local",
        "active_backend": active_backend(),
        "model_id": str(getattr(settings, "local_llm_model", "") or ""),
        "configured_device": str(getattr(settings, "local_llm_device", "cuda:0") or "cuda:0"),
        "visible_cuda_devices": str(getattr(settings, "cuda_visible_devices", "0") or "0"),
        "resolved_device": "not_checked",
        "loaded": False,
        "deep": bool(deep),
        "cuda_available": None,
        "cuda_devices": [],
        "cuda_check": "skipped" if not deep else "requested",
        "message": "CUDA/model diagnostics skipped. Use deep=true or the dashboard Check CUDA button.",
        "training": {
            "enabled": bool(getattr(settings, "local_training_enabled", False)),
            "device_config": str(getattr(settings, "local_training_device", "cuda:0") or "cuda:0"),
            "model_id": str(getattr(settings, "local_training_model", None) or getattr(settings, "local_llm_model", "")),
            "max_gpu_mb": int(getattr(settings, "local_training_max_gpu_mb", 0) or 0),
            "feedback_training_enabled": bool(getattr(settings, "feedback_training_enabled", True)),
            "feedback_weight": int(getattr(settings, "local_training_feedback_weight", 4) or 4),
            "worker_entrypoint": "python -m app.training.worker",
        },
    }
    if not deep:
        return base

    try:
        from app.llm.local_hf_client import LocalHFClient

        client = None
        with _LLM_LOCK:
            if getattr(_LLM, "_factory_backend", None) == "local":
                client = _LLM
        if client is None:
            client = LocalHFClient()
        status = dict(client.runtime_status())
    except Exception as e:
        status = {"backend": "local", "error": f"{type(e).__name__}: {e}"}

    merged = {**base, **status}
    merged["active_backend"] = active_backend()
    merged["deep"] = True
    merged["cuda_check"] = "completed" if "error" not in merged else "failed"
    merged["training"] = base["training"]
    return merged
