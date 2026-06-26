from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class ScopedLLMClient:
    """Pins an LLM purpose for an agent and mirrors LLM calls into run logs."""

    def __init__(self, base: Any, *, purpose: str, run_logger: Optional[Callable[[str], None]] = None) -> None:
        self._base = base
        self._purpose = purpose
        self._run_logger = run_logger
        self.backend = getattr(base, "backend", "llm")

    def _log(self, message: str) -> None:
        if not callable(self._run_logger):
            return
        try:
            self._run_logger(message)
        except Exception:
            pass

    def _start(self, method: str, purpose: str) -> float:
        self._log(f"LLM {self.backend}/{purpose} {method} started")
        return time.monotonic()

    def _finish(self, method: str, purpose: str, started: float, result: Any = None) -> None:
        elapsed = time.monotonic() - started
        extra = ""
        if isinstance(result, str):
            extra = f", chars={len(result)}"
        elif isinstance(result, (dict, list)):
            extra = f", type={type(result).__name__}"
        self._log(f"LLM {self.backend}/{purpose} {method} finished in {elapsed:.1f}s{extra}")

    def _fail(self, method: str, purpose: str, started: float, err: BaseException) -> None:
        elapsed = time.monotonic() - started
        msg = str(err or "").replace("\n", " ")
        if len(msg) > 300:
            msg = msg[:297] + "..."
        self._log(f"LLM {self.backend}/{purpose} {method} failed after {elapsed:.1f}s: {type(err).__name__}: {msg}")

    def is_available(self) -> bool:
        fn = getattr(self._base, "is_available", None)
        return bool(fn()) if callable(fn) else True

    def recommended_timeout_s(self) -> float:
        fn = getattr(self._base, "recommended_timeout_s", None)
        if callable(fn):
            return float(fn())
        return 0.0

    def chat_text(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        purpose: Optional[str] = None,
    ) -> str:
        effective_purpose = purpose or self._purpose
        started = self._start("chat_text", effective_purpose)
        try:
            try:
                result = self._base.chat_text(system=system, messages=messages, temperature=temperature, purpose=effective_purpose)
            except TypeError:
                result = self._base.chat_text(system=system, messages=messages, temperature=temperature)
            self._finish("chat_text", effective_purpose, started, result)
            return result
        except Exception as e:
            self._fail("chat_text", effective_purpose, started, e)
            raise

    def chat_json(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        purpose: Optional[str] = None,
    ) -> Any:
        effective_purpose = purpose or self._purpose
        started = self._start("chat_json", effective_purpose)
        try:
            try:
                result = self._base.chat_json(system=system, messages=messages, temperature=temperature, purpose=effective_purpose)
            except TypeError:
                result = self._base.chat_json(system=system, messages=messages, temperature=temperature)
            self._finish("chat_json", effective_purpose, started, result)
            return result
        except Exception as e:
            self._fail("chat_json", effective_purpose, started, e)
            raise

    async def chat_text_async(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        timeout_s: Optional[float] = None,
        purpose: Optional[str] = None,
    ) -> str:
        effective_purpose = purpose or self._purpose
        started = self._start("chat_text_async", effective_purpose)
        try:
            try:
                result = await self._base.chat_text_async(
                    system=system,
                    messages=messages,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    purpose=effective_purpose,
                )
            except TypeError:
                result = await self._base.chat_text_async(
                    system=system,
                    messages=messages,
                    temperature=temperature,
                    timeout_s=timeout_s,
                )
            self._finish("chat_text_async", effective_purpose, started, result)
            return result
        except Exception as e:
            self._fail("chat_text_async", effective_purpose, started, e)
            raise

    async def chat_json_async(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        timeout_s: Optional[float] = None,
        purpose: Optional[str] = None,
    ) -> Any:
        effective_purpose = purpose or self._purpose
        started = self._start("chat_json_async", effective_purpose)
        try:
            try:
                result = await self._base.chat_json_async(
                    system=system,
                    messages=messages,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    purpose=effective_purpose,
                )
            except TypeError:
                result = await self._base.chat_json_async(
                    system=system,
                    messages=messages,
                    temperature=temperature,
                    timeout_s=timeout_s,
                )
            self._finish("chat_json_async", effective_purpose, started, result)
            return result
        except Exception as e:
            self._fail("chat_json_async", effective_purpose, started, e)
            raise
