from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.llm.common import LLMUnavailable


log = logging.getLogger(__name__)


def _looks_like_no_answer(text: str) -> bool:
    """Heuristic: detect when a model likely failed to answer."""

    t = (text or "").strip()
    if not t:
        return True

    tl = t.lower()
    phrases = [p.strip().lower() for p in (settings.llm_fallback_phrases or []) if str(p).strip()]
    for p in phrases:
        if p and p in tl:
            return True

    min_chars = int(getattr(settings, "llm_fallback_min_chars", 0) or 0)
    if min_chars > 0 and len(t) < min_chars:
        return True

    return False


@dataclass
class FallbackLLMClient:
    """LLM client wrapper that can walk across a provider fallback chain.

    Examples:
      - OpenAI -> Gemini -> Claude -> local
      - local -> OpenAI -> Gemini -> Claude

    The legacy two-hop constructor still works:
      FallbackLLMClient(primary=openai, fallback=local)
    """

    primary: Any
    fallback: Any | None = None
    fallbacks: Optional[List[Any]] = None

    def __post_init__(self) -> None:
        self._clients: List[Any] = []
        seen: set[str] = set()
        for client in [self.primary, self.fallback, *(self.fallbacks or [])]:
            if client is None:
                continue
            name = str(getattr(client, "backend", client.__class__.__name__)).strip() or client.__class__.__name__
            if name in seen:
                continue
            seen.add(name)
            self._clients.append(client)

        chain = "->".join(getattr(c, "backend", c.__class__.__name__) for c in self._clients) or "llm"
        self.backend = f"fallback({chain})"

    def _fallbacks_enabled(self) -> bool:
        return bool(getattr(settings, "llm_fallback_to_openai", False))

    def _client_name(self, client: Any) -> str:
        return str(getattr(client, "backend", client.__class__.__name__))

    def _client_available(self, client: Any) -> bool:
        try:
            fn = getattr(client, "is_available", None)
            return bool(fn()) if callable(fn) else True
        except Exception:
            return False

    def _eligible_clients(self) -> List[Any]:
        if not self._fallbacks_enabled():
            return self._clients[:1]
        return list(self._clients)

    def _has_available_client_after(self, clients: List[Any], index: int) -> bool:
        for client in clients[index + 1 :]:
            if self._client_available(client):
                return True
        return False

    def is_available(self) -> bool:
        for client in self._eligible_clients():
            if self._client_available(client):
                return True
        return False

    def _should_attempt_fallback(self, primary_text: str, clients: List[Any], index: int) -> bool:
        if not self._fallbacks_enabled():
            return False
        if not self._has_available_client_after(clients, index):
            return False
        return _looks_like_no_answer(primary_text)

    def _should_attempt_fallback_json(self, obj: Any, clients: List[Any], index: int) -> bool:
        if obj is None:
            return self._should_attempt_fallback("", clients, index)

        if isinstance(obj, str):
            return self._should_attempt_fallback(obj, clients, index)

        if isinstance(obj, dict):
            for k in ("answer", "text", "content", "response"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return self._should_attempt_fallback(v, clients, index)
            if any(k in obj for k in ("answer", "confidence", "key_points", "followups")):
                return self._should_attempt_fallback("", clients, index)
            return False

        return False

    def _call_sync(self, client: Any, method_name: str, **kwargs: Any) -> Any:
        fn = getattr(client, method_name)
        try:
            return fn(**kwargs)
        except TypeError:
            kwargs.pop("purpose", None)
            return fn(**kwargs)

    async def _call_async(self, client: Any, method_name: str, **kwargs: Any) -> Any:
        fn = getattr(client, method_name, None)
        if callable(fn):
            try:
                return await fn(**kwargs)
            except TypeError:
                kwargs.pop("purpose", None)
                return await fn(**kwargs)

        sync_name = method_name.removesuffix("_async")
        return await asyncio.to_thread(self._call_sync, client, sync_name, **kwargs)

    def _run_sync(
        self,
        *,
        method_name: str,
        should_continue,
        continue_reason: str,
        **kwargs: Any,
    ) -> Any:
        clients = self._eligible_clients()
        if not clients:
            raise LLMUnavailable("No LLM clients configured")

        last_success: Any = None
        last_error: Optional[BaseException] = None

        for i, client in enumerate(clients):
            name = self._client_name(client)
            if not self._client_available(client):
                log.info("Skipping unavailable LLM backend '%s'.", name)
                continue

            try:
                out = self._call_sync(client, method_name, **kwargs)
            except Exception as e:
                last_error = e
                if self._fallbacks_enabled() and self._has_available_client_after(clients, i):
                    log.warning("LLM backend '%s' failed (%s). Trying next fallback.", name, e)
                    continue
                raise

            last_success = out
            if should_continue(out, clients, i):
                log.info("LLM backend '%s' returned %s; trying next fallback.", name, continue_reason)
                continue
            return out

        if last_success is not None:
            return last_success
        if last_error is not None:
            raise last_error
        raise LLMUnavailable("No available LLM backends in fallback chain")

    async def _run_async(
        self,
        *,
        method_name: str,
        should_continue,
        continue_reason: str,
        **kwargs: Any,
    ) -> Any:
        clients = self._eligible_clients()
        if not clients:
            raise LLMUnavailable("No LLM clients configured")

        last_success: Any = None
        last_error: Optional[BaseException] = None

        for i, client in enumerate(clients):
            name = self._client_name(client)
            if not self._client_available(client):
                log.info("Skipping unavailable LLM backend '%s'.", name)
                continue

            try:
                out = await self._call_async(client, method_name, **kwargs)
            except Exception as e:
                last_error = e
                if self._fallbacks_enabled() and self._has_available_client_after(clients, i):
                    log.warning("LLM backend '%s' failed (%s). Trying next fallback.", name, e)
                    continue
                raise

            last_success = out
            if should_continue(out, clients, i):
                log.info("LLM backend '%s' returned %s; trying next fallback.", name, continue_reason)
                continue
            return out

        if last_success is not None:
            return last_success
        if last_error is not None:
            raise last_error
        raise LLMUnavailable("No available LLM backends in fallback chain")

    def chat_text(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        purpose: Optional[str] = None,
    ) -> str:
        return self._run_sync(
            method_name="chat_text",
            should_continue=self._should_attempt_fallback,
            continue_reason="a low-confidence/no-answer response",
            system=system,
            messages=messages,
            temperature=temperature,
            purpose=purpose,
        )

    def chat_json(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        purpose: Optional[str] = None,
    ) -> Any:
        return self._run_sync(
            method_name="chat_json",
            should_continue=self._should_attempt_fallback_json,
            continue_reason="a low-confidence/no-answer JSON payload",
            system=system,
            messages=messages,
            temperature=temperature,
            purpose=purpose,
        )

    async def chat_text_async(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        timeout_s: Optional[float] = None,
        purpose: Optional[str] = None,
    ) -> str:
        return await self._run_async(
            method_name="chat_text_async",
            should_continue=self._should_attempt_fallback,
            continue_reason="a low-confidence/no-answer response",
            system=system,
            messages=messages,
            temperature=temperature,
            timeout_s=timeout_s,
            purpose=purpose,
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
        return await self._run_async(
            method_name="chat_json_async",
            should_continue=self._should_attempt_fallback_json,
            continue_reason="a low-confidence/no-answer JSON payload",
            system=system,
            messages=messages,
            temperature=temperature,
            timeout_s=timeout_s,
            purpose=purpose,
        )
