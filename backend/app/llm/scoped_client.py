from __future__ import annotations

from typing import Any, Dict, List, Optional


class ScopedLLMClient:
    """A lightweight wrapper that pins an LLM "purpose" for an agent instance.

    This lets us route OpenAI models differently for:
      - Q&A (purpose="qa")
      - Coding pipeline (purpose="code")

    Without changing the orchestrator/agent hierarchy or requiring every agent
    call-site to pass `purpose=` explicitly.
    """

    def __init__(self, base: Any, *, purpose: str) -> None:
        self._base = base
        self._purpose = purpose
        # Preserve a readable backend name for logs.
        self.backend = getattr(base, "backend", "llm")

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
        try:
            return self._base.chat_text(system=system, messages=messages, temperature=temperature, purpose=effective_purpose)
        except TypeError:
            # Backward compat for any custom LLM client that doesn't accept purpose.
            return self._base.chat_text(system=system, messages=messages, temperature=temperature)

    def chat_json(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        purpose: Optional[str] = None,
    ) -> Any:
        effective_purpose = purpose or self._purpose
        try:
            return self._base.chat_json(system=system, messages=messages, temperature=temperature, purpose=effective_purpose)
        except TypeError:
            return self._base.chat_json(system=system, messages=messages, temperature=temperature)

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
        try:
            return await self._base.chat_text_async(
                system=system,
                messages=messages,
                temperature=temperature,
                timeout_s=timeout_s,
                purpose=effective_purpose,
            )
        except TypeError:
            return await self._base.chat_text_async(
                system=system,
                messages=messages,
                temperature=temperature,
                timeout_s=timeout_s,
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
        effective_purpose = purpose or self._purpose
        try:
            return await self._base.chat_json_async(
                system=system,
                messages=messages,
                temperature=temperature,
                timeout_s=timeout_s,
                purpose=effective_purpose,
            )
        except TypeError:
            return await self._base.chat_json_async(
                system=system,
                messages=messages,
                temperature=temperature,
                timeout_s=timeout_s,
            )
