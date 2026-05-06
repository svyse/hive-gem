from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.llm.common import LLMUnavailable, safe_json_loads
from app.llm.adapter_paths import resolve_model_adapter_dir


log = logging.getLogger(__name__)

# Reduce CUDA fragmentation on 12 GB cards such as the RTX 3060.
# Respect an explicit user setting if they already provided one.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _from_pretrained_with_dtype(loader: Any, model_id: str, *, dtype: Any, **kwargs: Any) -> Any:
    """Load a HF model using the modern `dtype=` kwarg when available.

    Newer Transformers versions warn that `torch_dtype` is deprecated in this path.
    Older versions may still only accept `torch_dtype`, so we probe the signature and
    fall back safely for compatibility.
    """

    try:
        sig = inspect.signature(loader.from_pretrained)
        if "dtype" in sig.parameters:
            return loader.from_pretrained(model_id, dtype=dtype, **kwargs)
    except Exception:
        pass

    try:
        return loader.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError as e:
        msg = str(e)
        if "dtype" not in msg and "unexpected keyword" not in msg:
            raise
    return loader.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


def _pick_largest_cuda_device() -> str | None:
    """Pick the CUDA device with the most VRAM.

    This helps on multi-GPU machines where cuda:0 may be a smaller card.
    """

    try:
        import torch
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    best_i = 0
    best_mem = 0
    for i in range(int(torch.cuda.device_count() or 0)):
        try:
            mem = int(torch.cuda.get_device_properties(i).total_memory)
        except Exception:
            mem = 0
        if mem > best_mem:
            best_i = i
            best_mem = mem

    return f"cuda:{best_i}"


def _resolve_device(device: str | None) -> str:
    d = (device or "auto").strip().lower()
    try:
        import torch
    except Exception:
        # Transformers requires torch in practice; fall back to CPU.
        return "cpu"

    # Auto: pick the largest VRAM GPU if available (configurable), otherwise CPU.
    if d in {"auto", ""}:
        if torch.cuda.is_available():
            if bool(getattr(settings, "local_llm_prefer_largest_gpu", True)):
                picked = _pick_largest_cuda_device()
                return picked or "cuda:0"
            return "cuda:0"
        return "cpu"

    # Common aliases
    if d in {"cuda", "gpu"}:
        picked = _pick_largest_cuda_device()
        return picked or "cuda:0"

    return device or "cpu"




def _has_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _has_peft() -> bool:
    try:
        import peft  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class LocalHFClient:
    """Local Hugging Face Transformers client.

    Provides the same high-level API used by agents (chat_text/chat_json + async variants),
    but runs a local model via `transformers`.

    Notes:
    - Model weights are loaded lazily on first use.
    - Concurrency is limited (default: 1) to avoid GPU OOM.
    - Optional LoRA adapters (PEFT) are loaded if present.
    """

    model_id: str = settings.local_llm_model
    device: str = settings.local_llm_device

    def __post_init__(self) -> None:
        self.backend = "local"
        self._device = _resolve_device(self.device)
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_error: Optional[str] = None
        self._load_guard = threading.Lock()
        self._lock = threading.Semaphore(max(1, int(getattr(settings, "local_llm_concurrency", 1) or 1)))
        self._adapter_mtime: Optional[float] = None

        # Optional HF hub token / cache dir
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if settings.hf_home:
            os.environ.setdefault("HF_HOME", str(Path(settings.hf_home).expanduser()))
        if settings.hf_hub_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", settings.hf_hub_token)

    # ------------------------------
    # Availability
    # ------------------------------

    def is_available(self) -> bool:
        # Consider local backend "available" even before weights are downloaded.
        # We only return False if we already tried loading and failed.
        return self._load_error is None

    def runtime_status(self) -> Dict[str, Any]:
        """Return local model/GPU status for the dashboard and diagnostics."""
        info: Dict[str, Any] = {
            "backend": "local",
            "model_id": self.model_id,
            "configured_device": self.device,
            "resolved_device": self._device,
            "loaded": self._model is not None and self._tokenizer is not None,
            "load_error": self._load_error,
            "bitsandbytes_available": _has_bitsandbytes(),
            "peft_available": _has_peft(),
            "adapter_path": str(self._adapter_path()),
            "adapter_loaded_mtime": self._adapter_mtime,
        }

        try:
            import torch

            info["cuda_available"] = bool(torch.cuda.is_available())
            info["cuda_device_count"] = int(torch.cuda.device_count() if torch.cuda.is_available() else 0)
            devices: List[Dict[str, Any]] = []
            if torch.cuda.is_available():
                for i in range(int(torch.cuda.device_count())):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        total_mb = int(getattr(props, "total_memory", 0) or 0) // (1024 * 1024)
                        devices.append(
                            {
                                "index": i,
                                "name": str(getattr(props, "name", f"cuda:{i}")),
                                "total_memory_mb": total_mb,
                                "allocated_mb": int(torch.cuda.memory_allocated(i)) // (1024 * 1024),
                                "reserved_mb": int(torch.cuda.memory_reserved(i)) // (1024 * 1024),
                            }
                        )
                    except Exception as e:
                        devices.append({"index": i, "error": str(e)})
            info["cuda_devices"] = devices
            try:
                info["cuda_current_device"] = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
            except Exception:
                info["cuda_current_device"] = None
        except Exception as e:
            info["cuda_available"] = False
            info["torch_error"] = str(e)

        if self._model is not None:
            try:
                info["hf_device_map"] = getattr(self._model, "hf_device_map", None)
            except Exception:
                info["hf_device_map"] = None
            try:
                first_param = next(self._model.parameters())
                info["first_parameter_device"] = str(first_param.device)
                info["first_parameter_dtype"] = str(first_param.dtype).replace("torch.", "")
            except Exception as e:
                info["first_parameter_error"] = str(e)
        return info

    # ------------------------------
    # Loading
    # ------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        with self._load_guard:
            if self._model is not None and self._tokenizer is not None:
                return

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from transformers import BitsAndBytesConfig
            except Exception as e:
                self._load_error = str(e)
                raise LLMUnavailable(f"Local HF backend unavailable (missing deps): {e}") from e

            model_id = self.model_id
            use_cuda = self._device.startswith("cuda") and torch.cuda.is_available()

            if use_cuda:
                try:
                    idx = int(self._device.split(":", 1)[1]) if ":" in self._device else int(torch.cuda.current_device())
                    props = torch.cuda.get_device_properties(idx)
                    log.info(
                        "Local CUDA selected: %s | %s | total_vram=%s MiB",
                        self._device,
                        getattr(props, "name", "CUDA GPU"),
                        int(getattr(props, "total_memory", 0) or 0) // (1024 * 1024),
                    )
                    torch.cuda.set_device(idx)
                    torch.cuda.empty_cache()
                except Exception as e:
                    log.warning("CUDA was selected but diagnostics/setup failed for %s: %s", self._device, e)
            else:
                log.warning(
                    "Local model will run on CPU (configured=%s, resolved=%s, cuda_available=%s).",
                    self.device,
                    self._device,
                    bool(torch.cuda.is_available()),
                )

            log.info("Loading local model: %s (device=%s)", model_id, self._device)

            trust_remote = bool(getattr(settings, "local_llm_trust_remote_code", False))

            # Tokenizer
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            # Optional 4-bit load (best-effort)
            quant_cfg = None
            load_4bit = bool(getattr(settings, "local_llm_load_in_4bit", False))
            if load_4bit and use_cuda and _has_bitsandbytes():
                try:
                    quant_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                except Exception as e:
                    log.warning("4-bit requested but BitsAndBytesConfig failed: %s", e)
                    quant_cfg = None
            elif load_4bit and use_cuda:
                log.warning("LOCAL_LLM_LOAD_IN_4BIT=true but bitsandbytes is unavailable; falling back to FP16/offload.")

            torch_dtype = torch.float16 if use_cuda else torch.float32

            def _cuda_index(dev: str) -> int:
                try:
                    if ":" in dev:
                        return int(dev.split(":", 1)[1])
                except Exception:
                    pass
                return 0

            def _load_model(*, offload: bool) -> Any:
                # Offload path: allow some weights on CPU to keep VRAM under the configured limit.
                kwargs: Dict[str, Any] = {}
                if offload and use_cuda:
                    gpu_mb = int(getattr(settings, "local_llm_max_gpu_mb", 11000) or 11000)
                    cpu_mb = int(getattr(settings, "local_llm_max_cpu_mb", 24000) or 24000)
                    idx = _cuda_index(self._device)
                    kwargs.update(
                        {
                            "device_map": "auto",
                            "max_memory": {idx: f"{gpu_mb}MiB", "cpu": f"{cpu_mb}MiB"},
                            "offload_folder": str(Path(settings.local_llm_offload_folder).expanduser()),
                            "offload_state_dict": True,
                            "low_cpu_mem_usage": True,
                        }
                    )
                elif use_cuda:
                    # Regular GPU load on the chosen device
                    kwargs.update({"device_map": {"": self._device}})

                if quant_cfg is not None:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        trust_remote_code=trust_remote,
                        quantization_config=quant_cfg,
                        **kwargs,
                    )
                else:
                    model = _from_pretrained_with_dtype(
                        AutoModelForCausalLM,
                        model_id,
                        trust_remote_code=trust_remote,
                        dtype=torch_dtype,
                        **kwargs,
                    )
                    # If not using device_map, we may need to move to device explicitly.
                    if use_cuda and not offload and not kwargs.get("device_map"):
                        model.to(self._device)

                try:
                    if getattr(model, "config", None) is not None:
                        model.config.use_cache = True
                except Exception:
                    pass
                model.eval()
                return model

            try:
                model = _load_model(offload=False)
            except Exception as e:
                msg = str(e)
                is_oom = use_cuda and ("out of memory" in msg.lower() or "cuda" in msg.lower() and "oom" in msg.lower())
                if is_oom and bool(getattr(settings, "local_llm_offload_if_oom", True)):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    log.warning(
                        "CUDA OOM while loading %s on %s; retrying with CPU offload (device_map=auto).",
                        model_id,
                        self._device,
                    )
                    try:
                        model = _load_model(offload=True)
                    except Exception as e2:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        self._load_error = str(e2)
                        raise LLMUnavailable(f"Failed to load local model '{model_id}' (even with offload): {e2}") from e2
                else:
                    self._load_error = msg
                    raise LLMUnavailable(f"Failed to load local model '{model_id}': {e}") from e

            self._tokenizer = tok
            self._model = model

            try:
                first_param = next(model.parameters())
                log.info(
                    "Local model loaded: model=%s | device_map=%s | first_parameter_device=%s | dtype=%s",
                    model_id,
                    getattr(model, "hf_device_map", None),
                    str(first_param.device),
                    str(first_param.dtype).replace("torch.", ""),
                )
            except Exception:
                log.info("Local model loaded: model=%s | device_map=%s", model_id, getattr(model, "hf_device_map", None))

            # Optional LoRA adapter
            self._maybe_load_adapter(force=True)

    def _adapter_path(self) -> Path:
        return resolve_model_adapter_dir(getattr(settings, "local_llm_adapter_dir", "") or "", self.model_id)

    def _adapter_base_model_name(self, adapter_dir: Path) -> str | None:
        cfg = adapter_dir / "adapter_config.json"
        if not cfg.exists():
            return None
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            name = data.get("base_model_name_or_path") or data.get("base_model_name")
            return str(name).strip() if name else None
        except Exception:
            return None

    def _adapter_is_compatible(self, adapter_dir: Path) -> bool:
        base_name = self._adapter_base_model_name(adapter_dir)
        if not base_name:
            return True

        want = str(self.model_id or "").strip().lower()
        have = str(base_name or "").strip().lower()
        if want == have:
            return True

        log.info(
            "Skipping LoRA adapter at %s because it targets base model '%s' but live model is '%s'.",
            adapter_dir,
            base_name,
            self.model_id,
        )
        return False

    def _maybe_load_adapter(self, *, force: bool = False) -> None:
        if not _has_peft():
            return

        adapter_dir = self._adapter_path()
        if not adapter_dir:
            return

        cfg = adapter_dir / "adapter_config.json"
        if not cfg.exists():
            return
        if not self._adapter_is_compatible(adapter_dir):
            return

        try:
            mtime = cfg.stat().st_mtime
        except Exception:
            mtime = None

        if (not force) and (self._adapter_mtime is not None) and (mtime is not None) and (mtime <= self._adapter_mtime):
            return

        try:
            from peft import PeftModel  # type: ignore

            log.info("Loading LoRA adapter from: %s", adapter_dir)
            self._model = PeftModel.from_pretrained(self._model, str(adapter_dir), is_trainable=False)
            self._adapter_mtime = mtime
        except Exception as e:
            log.warning("Failed to load adapter at %s: %s", adapter_dir, e)

    # ------------------------------
    # Prompt formatting
    # ------------------------------

    def _format_prompt(self, *, system: str, messages: List[Dict[str, str]]) -> str:
        self._ensure_loaded()
        tok = self._tokenizer

        chat = [{"role": "system", "content": system}] + (messages or [])

        # Use chat templates if the tokenizer provides them.
        try:
            if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

        # Fallback: very simple plain-text formatting
        parts: List[str] = []
        if system:
            parts.append(f"System: {system}")
        for m in chat:
            if m.get("role") == "system":
                continue
            role = (m.get("role") or "user").capitalize()
            parts.append(f"{role}: {m.get('content') or ''}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _effective_timeout(self, timeout_s: Optional[float]) -> float:
        regular = float(
            getattr(settings, "local_llm_timeout_s", None)
            or getattr(settings, "llm_timeout_s", 60.0)
            or 60.0
        )
        initial = float(
            getattr(settings, "local_llm_initial_timeout_s", None)
            or max(regular, 180.0)
        )

        requested = float(timeout_s) if timeout_s is not None else regular
        if self._model is None or self._tokenizer is None:
            return max(requested, initial)
        return requested

    def _max_new_tokens_for_purpose(self, purpose: Optional[str], ctx_len: int) -> int:
        p = str(purpose or "").strip().lower()
        default = int(getattr(settings, "local_llm_max_new_tokens", 384) or 384)
        if "json" in p:
            value = int(getattr(settings, "local_llm_json_max_new_tokens", default) or default)
        elif any(k in p for k in ("code", "module", "plan", "test", "compile", "orchestr")):
            value = int(getattr(settings, "local_llm_code_max_new_tokens", default) or default)
        elif any(k in p for k in ("qa", "answer", "chat")):
            value = int(getattr(settings, "local_llm_qa_max_new_tokens", default) or default)
        else:
            value = default

        if value >= ctx_len:
            value = max(64, ctx_len // 4)
        return max(64, min(value, max(64, ctx_len - 128)))

    def _json_retry_messages(self, *, messages: List[Dict[str, str]], raw_text: str) -> List[Dict[str, str]]:
        repaired = list(messages or [])
        repaired.append({"role": "assistant", "content": raw_text})
        repaired.append(
            {
                "role": "user",
                "content": (
                    "Re-emit the previous answer as strict valid JSON only. "
                    "Do not add markdown fences, commentary, or any text before or after the JSON."
                ),
            }
        )
        return repaired

    # ------------------------------
    # Chat API
    # ------------------------------

    def _infer_purpose(self, *, system: str, messages: List[Dict[str, str]], default: str = "qa") -> str:
        tail = "\n".join(str((m or {}).get("content") or "") for m in (messages or [])[-2:])
        blob = (str(system or "") + "\n" + tail).lower()
        if any(k in blob for k in ("code", "file", "project", "pytest", "compile", "module", "function", "class", "workspace")):
            return "code-json"
        if "json" in blob:
            return "json"
        return default

    def chat_text(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        purpose: Optional[str] = None,
    ) -> str:
        # Ensure model is ready.
        self._ensure_loaded()
        # Hot-reload adapter if training updated it.
        self._maybe_load_adapter(force=False)

        prompt = self._format_prompt(system=system, messages=messages)

        import torch

        with self._lock:
            tok = self._tokenizer
            model = self._model

            # ------------------------------
            # Tokenization with truncation
            # ------------------------------
            # Many small local models (e.g. TinyLlama) have a ~2k context.
            # Some orchestrator contexts (web results, memories, uploads) can
            # easily exceed that and cause "indexing errors".
            #
            # We keep the *most recent* part of the prompt by truncating from
            # the left, leaving room for generation tokens.

            def _context_window() -> int:
                # Prefer model config; fall back to tokenizer; last resort 2048.
                try:
                    cfg = getattr(model, "config", None)
                    for attr in (
                        "max_position_embeddings",
                        "n_positions",
                        "max_seq_len",
                        "max_sequence_length",
                        "seq_length",
                    ):
                        v = getattr(cfg, attr, None)
                        if isinstance(v, int) and 256 <= v <= 100_000:
                            return int(v)
                except Exception:
                    pass
                try:
                    v = int(getattr(tok, "model_max_length", 0) or 0)
                    if 256 <= v <= 100_000:
                        return int(v)
                except Exception:
                    pass
                return 2048

            ctx_len = _context_window()
            # Clamp max_new_tokens so input+output stays within ctx_len.
            max_new_tokens = self._max_new_tokens_for_purpose(purpose, ctx_len)

            # Leave a small buffer for special tokens and cap input size to avoid OOM.
            max_input_tokens = max(128, ctx_len - max_new_tokens - 8)
            configured_input_cap = int(getattr(settings, "local_llm_max_input_tokens", 3072) or 0)
            if configured_input_cap > 0:
                max_input_tokens = max(128, min(max_input_tokens, configured_input_cap))

            old_side = getattr(tok, "truncation_side", "right")
            try:
                tok.truncation_side = "left"
            except Exception:
                pass

            try:
                inputs = tok(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_tokens,
                    add_special_tokens=False,
                )
            finally:
                try:
                    tok.truncation_side = old_side
                except Exception:
                    pass
            # Move inputs to the primary execution device.
            # Using `next(model.parameters()).device` is unreliable when the
            # model is sharded/offloaded (device_map="auto"). In that case the
            # first parameter can live on CPU even though the embedding layer is
            # on GPU, which causes device-mismatch errors during generation.
            try:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            except Exception:
                pass

            top_p = float(getattr(settings, "local_llm_top_p", 0.95) or 0.95)
            top_k = int(getattr(settings, "local_llm_top_k", 50) or 50)
            rep_pen = float(getattr(settings, "local_llm_repetition_penalty", 1.05) or 1.05)

            do_sample = float(temperature or 0) > 0
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "repetition_penalty": rep_pen,
                "pad_token_id": tok.pad_token_id,
                "eos_token_id": tok.eos_token_id,
                "use_cache": True,
            }
            if do_sample:
                gen_kwargs.update(
                    {
                        "temperature": float(temperature or 0.2),
                        "top_p": top_p,
                        "top_k": top_k,
                    }
                )

            try:
                prompt_tokens = int(inputs["input_ids"].shape[-1])
            except Exception:
                prompt_tokens = -1
            log.info(
                "Local generation start (device=%s, max_new_tokens=%s, prompt_tokens=%s, purpose=%s)",
                self._device,
                max_new_tokens,
                prompt_tokens,
                purpose or "qa",
            )
            try:
                with torch.inference_mode():
                    out = model.generate(**inputs, **gen_kwargs)
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" not in msg and "cuda" not in msg:
                    raise
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                retry_tokens = max(64, int(max_new_tokens // 2))
                gen_kwargs["max_new_tokens"] = retry_tokens
                log.warning(
                    "CUDA OOM during local generation; retrying with max_new_tokens=%s (was %s).",
                    retry_tokens,
                    max_new_tokens,
                )
                with torch.inference_mode():
                    out = model.generate(**inputs, **gen_kwargs)
            finally:
                if bool(getattr(settings, "local_llm_empty_cache_after_generate", False)) and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            try:
                generated_tokens = int(out[0].shape[-1] - inputs["input_ids"].shape[-1])
            except Exception:
                generated_tokens = -1
            log.info(
                "Local generation finished (device=%s, generated_tokens=%s, purpose=%s)",
                self._device,
                generated_tokens,
                purpose or "qa",
            )

            # Strip prompt from output
            prompt_len = inputs["input_ids"].shape[-1]
            gen_ids = out[0][prompt_len:]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            return (text or "").strip()

    def _repair_json_output(self, *, system: str, messages: List[Dict[str, str]], raw_text: str) -> Any:
        repair_system = (
            "You convert an assistant answer into VALID JSON.\n"
            "Preserve the original meaning and schema requested by the original system prompt.\n"
            "Return JSON only. Do not use markdown fences."
        )
        repair_messages = [
            {
                "role": "user",
                "content": (
                    "Original system prompt:\n"
                    f"{system}\n\n"
                    "Original user messages:\n"
                    f"{messages}\n\n"
                    "Assistant output to repair into valid JSON:\n"
                    f"{raw_text}"
                ),
            }
        ]
        repaired = self.chat_text(system=repair_system, messages=repair_messages, temperature=0.0, purpose="json-repair")
        return safe_json_loads(repaired)

    def chat_json(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        purpose: Optional[str] = None,
    ) -> Any:
        effective_purpose = purpose or self._infer_purpose(system=system, messages=messages, default="json")
        text = self.chat_text(system=system, messages=messages, temperature=temperature, purpose=effective_purpose)
        try:
            return safe_json_loads(text)
        except Exception:
            return self._repair_json_output(system=system, messages=messages, raw_text=text)

    def _async_timeout(self, timeout_s: Optional[float]) -> float:
        return self._effective_timeout(timeout_s)

    async def chat_text_async(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        timeout_s: Optional[float] = None,
        purpose: Optional[str] = None,
    ) -> str:
        # Local generation is compute-bound; run it in a thread to keep FastAPI responsive.
        # IMPORTANT: respect a per-call timeout so callers can fall back to OpenAI
        # if the local model is still downloading or takes too long to generate.
        t = self._async_timeout(timeout_s)
        return await asyncio.wait_for(
            asyncio.to_thread(self.chat_text, system=system, messages=messages, temperature=temperature, purpose=purpose),
            timeout=t,
        )

    async def chat_json_async(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        timeout_s: Optional[float] = None,
        purpose: Optional[str] = None,
    ) -> Any:
        t = self._async_timeout(timeout_s)
        return await asyncio.wait_for(
            asyncio.to_thread(self.chat_json, system=system, messages=messages, temperature=temperature, purpose=purpose),
            timeout=t,
        )
