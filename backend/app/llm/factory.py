from __future__ import annotations

import gc
import json
import logging
import threading
import queue
import contextvars
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.core.config import settings
log = logging.getLogger(__name__)


_LLM: Optional[Any] = None
_LLM_LOCK = threading.RLock()
_LLM_EPOCH = 0
_ACTIVE_BACKEND_OVERRIDE: Optional[str] = None
_ALLOWED_BACKENDS = {"local", "openai", "gemini", "claude"}
_DASHBOARD_BACKENDS = ("local", "openai", "gemini")
_REQUEST_BACKEND: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("agentic_hive_request_backend", default=None)

_PERSIST_QUEUE: "queue.Queue[str]" = queue.Queue(maxsize=4)
_PERSIST_WORKER_STARTED = False
_PERSIST_WORKER_LOCK = threading.Lock()
_PERSISTED_BACKEND_LOADED = False


def _status_for_backend(backend: str) -> Dict[str, Any]:
    """Pure in-memory status helper used by hot-swap routes.

    This must remain free of local model/CUDA imports and filesystem reads/writes.
    The dashboard switch endpoint uses it so a busy local Torch runtime cannot
    block switching to Gemini/OpenAI.
    """

    b = _normalize_backend(backend)
    if b not in _DASHBOARD_BACKENDS:
        b = "local"
    return {
        "active_backend": b,
        "allowed_backends": list(_DASHBOARD_BACKENDS),
        "available": {name: _lightweight_available(name) for name in _DASHBOARD_BACKENDS},
        "models": {name: _model_name(name) for name in _DASHBOARD_BACKENDS},
        "fallback_enabled": bool(getattr(settings, "llm_fallback_to_openai", False)),
        "fallback_backends": list(getattr(settings, "llm_fallback_backends", []) or []),
    }


def _persist_worker() -> None:
    while True:
        backend = _PERSIST_QUEUE.get()
        try:
            _write_persisted_backend(str(backend))
        except Exception as e:
            log.warning("Could not persist runtime backend selection in background: %s", e)
        finally:
            try:
                _PERSIST_QUEUE.task_done()
            except Exception:
                pass


def _persist_backend_async(backend: str) -> None:
    global _PERSIST_WORKER_STARTED
    try:
        with _PERSIST_WORKER_LOCK:
            if not _PERSIST_WORKER_STARTED:
                t = threading.Thread(target=_persist_worker, name="backend-selection-persist", daemon=True)
                t.start()
                _PERSIST_WORKER_STARTED = True
        try:
            while _PERSIST_QUEUE.full():
                _PERSIST_QUEUE.get_nowait()
                _PERSIST_QUEUE.task_done()
        except Exception:
            pass
        _PERSIST_QUEUE.put_nowait(backend)
    except Exception as e:
        log.warning("Could not queue runtime backend persistence: %s", e)



def _normalize_backend(name: str | None) -> str:
    backend = (name or "").strip().lower()
    if backend not in _ALLOWED_BACKENDS:
        return "local"
    return backend


def _runtime_state_file() -> Path:
    """Small runtime state file used to persist dashboard hot-swaps across reloads.

    Uvicorn --reload restarts the backend process. If the selected provider lives
    only in memory, the dashboard appears to snap back to the .env value after a
    reload. Store the runtime selection beside the memory DB, which is outside the
    backend source tree in normal setups, so file writes do not trigger reloads.
    """

    try:
        base = Path(settings.memory_db_path).expanduser().resolve().parent
    except Exception:
        base = Path.home() / ".agentic_hive_studio"
    return base / "runtime_backend.json"


def _read_persisted_backend() -> Optional[str]:
    try:
        p = _runtime_state_file()
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8") or "{}")
        backend = _normalize_backend(str(data.get("active_backend") or ""))
        return backend if backend in _DASHBOARD_BACKENDS else None
    except Exception:
        return None


def _write_persisted_backend(backend: str) -> None:
    try:
        p = _runtime_state_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"active_backend": backend}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("Could not persist runtime backend selection: %s", e)


def _build_client(name: str) -> Any:
    """Build exactly one backend client.

    Do not instantiate every provider here. Constructing LocalHFClient imports
    torch and probes CUDA. On Windows/CUDA systems that can block if the GPU is
    busy with a prior local generation/load. Hot-swapping from local to Gemini
    must therefore avoid touching the local backend unless local is actually the
    selected backend or a fallback is actually invoked.
    """

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
    """Proxy that delays provider construction until a real LLM call is made.

    This is especially important for fallback chains. If Gemini is primary and
    local is only a fallback, switching to Gemini must not import torch/probe CUDA
    simply because local is listed later in LLM_FALLBACK_BACKENDS.
    """

    def __init__(self, backend: str, builder: Callable[[str], Any] = _build_client) -> None:
        self.backend = _normalize_backend(backend)
        self._builder = builder
        self._client: Any = None
        self._lock = threading.RLock()

    def _get(self) -> Any:
        with self._lock:
            if self._client is None:
                log.info("Building LLM backend lazily: %s", self.backend)
                self._client = self._builder(self.backend)
            return self._client

    def is_available(self) -> bool:
        # Cheap dashboard/fallback availability check; do not instantiate local.
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


def _lightweight_available(name: str) -> bool:
    """Return dashboard availability without network/model/GPU work.

    This function must never import torch, inspect the local HF client, acquire
    model-load locks, call provider SDKs, or probe CUDA. It is used by the
    hot-swap endpoint, so it has to remain fast even while local generation is
    loading/running in another thread.
    """

    backend = _normalize_backend(name)
    if backend == "gemini":
        return bool((getattr(settings, "gemini_api_key", None) or "").strip())
    if backend == "openai":
        return bool((getattr(settings, "openai_api_key", None) or "").strip())
    if backend == "local":
        return True
    return False


def _model_name(name: str) -> str:
    backend = _normalize_backend(name)
    if backend == "openai":
        return str(getattr(settings, "openai_model", "") or "")
    if backend == "gemini":
        return str(getattr(settings, "gemini_model", "") or "")
    if backend == "local":
        return str(getattr(settings, "local_llm_model", "") or "")
    if backend == "claude":
        return str(getattr(settings, "claude_model", "") or "")
    return ""


@contextmanager
def use_llm_backend(backend: str | None):
    """Temporarily force an LLM backend for the current request/task.

    This lets Q&A/code requests use the dashboard-selected provider even if the
    global hot-swap status request was delayed by a busy local CUDA runtime.
    ContextVars propagate through normal awaits and asyncio.create_task() on
    Python 3.11+, and asyncio.to_thread() copies the current context too.
    """

    normalized = _normalize_backend(backend) if backend else None
    if normalized not in _DASHBOARD_BACKENDS:
        normalized = None
    token = _REQUEST_BACKEND.set(normalized)
    try:
        yield
    finally:
        _REQUEST_BACKEND.reset(token)


def request_backend() -> Optional[str]:
    backend = _REQUEST_BACKEND.get()
    return backend if backend in _DASHBOARD_BACKENDS else None


def active_backend() -> str:
    global _ACTIVE_BACKEND_OVERRIDE, _PERSISTED_BACKEND_LOADED

    scoped = request_backend()
    if scoped:
        return scoped

    if _ACTIVE_BACKEND_OVERRIDE:
        return _normalize_backend(_ACTIVE_BACKEND_OVERRIDE)

    # Read persisted dashboard selection at most once per process. Repeated
    # backend-status polls must never repeatedly touch disk while a local CUDA
    # generation is running or while Windows file watchers are busy.
    if not _PERSISTED_BACKEND_LOADED:
        _PERSISTED_BACKEND_LOADED = True
        persisted = _read_persisted_backend()
        if persisted:
            _ACTIVE_BACKEND_OVERRIDE = persisted
            return persisted

    return _normalize_backend(getattr(settings, "llm_backend", "local"))


def get_backend_status() -> Dict[str, Any]:
    """Return a lightweight provider status without forcing model weights/CUDA."""

    return _status_for_backend(active_backend())


def _drop_cached_client(*, eager_gc: bool = False) -> None:
    """Forget the shared client without blocking on local model work."""

    global _LLM, _LLM_EPOCH
    with _LLM_LOCK:
        _LLM_EPOCH += 1
        _LLM = None
    if eager_gc:
        try:
            gc.collect()
        except Exception:
            pass


def set_active_backend(name: str) -> Dict[str, Any]:
    """Hot-swap the default backend for all new agent calls in this process.

    The switch path is deliberately pure in-memory and returns immediately. It
    does not build clients, import torch, probe CUDA, wait for a local generation
    lock, or synchronously write the runtime state file. Persistence happens on a
    daemon thread; Q&A/code requests also carry an explicit backend in their
    request body, so the frontend-selected backend works even if persistence is
    delayed.
    """

    global _ACTIVE_BACKEND_OVERRIDE, _LLM, _LLM_EPOCH, _PERSISTED_BACKEND_LOADED
    backend = _normalize_backend(name)
    if backend not in _DASHBOARD_BACKENDS:
        raise ValueError("Backend must be one of: local, openai, gemini")

    old_backend = _normalize_backend(_ACTIVE_BACKEND_OVERRIDE or getattr(settings, "llm_backend", "local"))

    setattr(settings, "llm_backend", backend)
    _ACTIVE_BACKEND_OVERRIDE = backend
    _PERSISTED_BACKEND_LOADED = True
    _LLM_EPOCH += 1
    _LLM = None

    _persist_backend_async(backend)

    log.info("LLM backend switched: %s -> %s (pure in-memory; clients stay lazy)", old_backend, backend)
    status = _status_for_backend(backend)
    status["message"] = f"Switched LLM backend from {old_backend} to {backend}. New requests will use {backend}."
    return status


def _build_selected_client(backend: str) -> Any:
    primary = _build_client(backend)

    if bool(getattr(settings, "llm_fallback_to_openai", False)):
        fallback_names = _fallback_order(backend)
        # Fallbacks must be lazy; do not instantiate local/GPU until truly needed.
        fallbacks = [LazyLLMClient(name) for name in fallback_names]
        if fallbacks:
            from app.llm.fallback_client import FallbackLLMClient

            chain = [backend, *fallback_names]
            log.info("Using LLM fallback chain: %s", " -> ".join(chain))
            return FallbackLLMClient(primary=primary, fallbacks=fallbacks)

    return primary


def get_llm_client() -> Any:
    """Return a process-wide LLM client without blocking dashboard hot-swap.

    We intentionally build clients outside _LLM_LOCK. If local torch/CUDA import is
    slow, the model switch endpoint can still acquire the lock, change the active
    backend, and invalidate the build epoch. A stale client built before a switch
    can finish its in-flight request, but it is not cached for future requests.
    """

    global _LLM

    with _LLM_LOCK:
        backend = active_backend()
        epoch = _LLM_EPOCH
        if _LLM is not None and getattr(_LLM, "_factory_backend", backend) == backend:
            return _LLM

    client = _build_selected_client(backend)
    try:
        setattr(client, "_factory_backend", backend)
    except Exception:
        pass

    with _LLM_LOCK:
        if epoch == _LLM_EPOCH and active_backend() == backend:
            _LLM = client
            return _LLM

    log.info("Discarding stale LLM client built for %s because backend changed during construction.", backend)
    return client


def get_local_runtime_status(*, deep: bool = False) -> Dict[str, Any]:
    """Return local runtime status.

    By default this is intentionally shallow and never imports torch/CUDA. The
    dashboard can call deep=true only when the user explicitly asks to check GPU
    status. This prevents an automatic local-status poll from blocking backend
    hot-swap on Windows/CUDA while torch is importing or the GPU is busy.
    """

    base: Dict[str, Any] = {
        "backend": "local",
        "active_backend": active_backend(),
        "model_id": str(getattr(settings, "local_llm_model", "") or ""),
        "configured_device": str(getattr(settings, "local_llm_device", "auto") or "auto"),
        "resolved_device": "not_checked",
        "loaded": False,
        "deep": bool(deep),
        "cuda_available": None,
        "cuda_devices": [],
        "cuda_check": "skipped" if not deep else "requested",
        "message": "CUDA/model diagnostics skipped. Use deep=true or the dashboard Check CUDA button to probe torch/GPU.",
        "training": {
            "enabled": bool(getattr(settings, "local_training_enabled", False)),
            "device_config": str(getattr(settings, "local_training_device", "auto") or "auto"),
            "model_id": str(getattr(settings, "local_training_model", None) or getattr(settings, "local_llm_model", "")),
            "max_gpu_mb": int(getattr(settings, "local_training_max_gpu_mb", 0) or 0),
            "feedback_training_enabled": bool(getattr(settings, "feedback_training_enabled", True)),
            "feedback_weight": int(getattr(settings, "local_training_feedback_weight", 4) or 4),
        },
    }

    if not deep:
        return base

    client: Any = None
    global _LLM
    try:
        acquired = _LLM_LOCK.acquire(blocking=False)
        try:
            if acquired and getattr(_LLM, "backend", None) == "local":
                client = _LLM
        finally:
            if acquired:
                _LLM_LOCK.release()
    except Exception:
        client = None

    if client is None:
        from app.llm.local_hf_client import LocalHFClient

        client = LocalHFClient()

    try:
        status = dict(client.runtime_status())
    except Exception as e:
        status = {"backend": "local", "error": f"{type(e).__name__}: {e}"}

    merged = {**base, **status}
    merged["active_backend"] = active_backend()
    merged["deep"] = True
    merged["cuda_check"] = "completed" if "error" not in merged else "failed"
    merged["training"] = base["training"]
    return merged


def reload_llm_client(*, eager: bool = True) -> Any:
    """Drop the cached client and optionally rebuild it."""

    _drop_cached_client(eager_gc=False)
    if eager:
        return get_llm_client()
    return None
