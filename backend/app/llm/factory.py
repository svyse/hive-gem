from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.llm.claude_client import ClaudeClient
from app.llm.gemini_client import GeminiClient
from app.llm.local_hf_client import LocalHFClient
from app.llm.openai_client import OpenAIClient


log = logging.getLogger(__name__)


_LLM: Optional[Any] = None
_ALLOWED_BACKENDS = {"local", "openai", "gemini", "claude"}
_DASHBOARD_BACKENDS = ("local", "openai", "gemini")


def _normalize_backend(name: str | None) -> str:
    backend = (name or "").strip().lower()
    if backend not in _ALLOWED_BACKENDS:
        return "local"
    return backend


def _build_clients() -> Dict[str, Any]:
    return {
        "local": LocalHFClient(),
        "openai": OpenAIClient(),
        "gemini": GeminiClient(),
        "claude": ClaudeClient(),
    }


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


def active_backend() -> str:
    return _normalize_backend(getattr(settings, "llm_backend", "local"))


def get_backend_status() -> Dict[str, Any]:
    """Return a lightweight provider status without forcing model weights to load."""

    clients = _build_clients()
    backend = active_backend()

    available = {}
    models = {}
    for name, client in clients.items():
        if name not in _DASHBOARD_BACKENDS:
            continue
        available[name] = _client_available(client)
        model = None
        for attr in ("model", "model_id"):
            try:
                value = getattr(client, attr, None)
                if value:
                    model = str(value)
                    break
            except Exception:
                pass
        if model is None:
            if name == "openai":
                model = str(getattr(settings, "openai_model", "") or "")
            elif name == "gemini":
                model = str(getattr(settings, "gemini_model", "") or "")
            elif name == "local":
                model = str(getattr(settings, "local_llm_model", "") or "")
        models[name] = model

    return {
        "active_backend": backend if backend in _DASHBOARD_BACKENDS else "local",
        "allowed_backends": list(_DASHBOARD_BACKENDS),
        "available": available,
        "models": models,
        "fallback_enabled": bool(getattr(settings, "llm_fallback_to_openai", False)),
        "fallback_backends": list(getattr(settings, "llm_fallback_backends", []) or []),
    }


def set_active_backend(name: str) -> Dict[str, Any]:
    """Hot-swap the default backend for all new agent calls in this process."""

    backend = _normalize_backend(name)
    if backend not in _DASHBOARD_BACKENDS:
        raise ValueError("Backend must be one of: local, openai, gemini")

    old_backend = active_backend()
    setattr(settings, "llm_backend", backend)
    reload_llm_client()
    log.info("LLM backend switched: %s -> %s", old_backend, backend)
    status = get_backend_status()
    status["message"] = f"Switched LLM backend from {old_backend} to {backend}. New requests will use {backend}."
    return status


def get_llm_client() -> Any:
    """Return a process-wide LLM client.

    The hive can spin up many agents per request. Loading a local transformer
    model per agent would explode VRAM. So we keep a single shared client.

    Selection rules:
      - LLM_BACKEND can be local, openai, gemini, or claude.
      - The dashboard can switch local/openai/gemini at runtime.
      - If LLM_FALLBACK_TO_OPENAI=true, the selected provider becomes the
        primary and the app walks the configured fallback chain.
      - Local remains the final safety net when included in
        LLM_FALLBACK_BACKENDS.
    """

    global _LLM
    if _LLM is not None:
        return _LLM

    backend = active_backend()
    clients = _build_clients()
    primary = clients[backend]

    if bool(getattr(settings, "llm_fallback_to_openai", False)):
        fallback_names = _fallback_order(backend)
        fallbacks = [clients[name] for name in fallback_names]
        if fallbacks:
            from app.llm.fallback_client import FallbackLLMClient

            chain = [backend, *fallback_names]
            log.info("Using LLM fallback chain: %s", " -> ".join(chain))
            _LLM = FallbackLLMClient(primary=primary, fallbacks=fallbacks)
            return _LLM

    _LLM = primary
    return _LLM


def reload_llm_client() -> Any:
    """Drop the cached client and rebuild it."""

    global _LLM
    _LLM = None
    return get_llm_client()
