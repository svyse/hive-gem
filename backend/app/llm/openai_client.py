from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.llm.common import LLMUnavailable, safe_json_loads


log = logging.getLogger(__name__)


@dataclass
class OpenAIClient:
    """OpenAI chat-completions client with per-purpose routing and smart cooldowns."""

    api_key: Optional[str] = settings.openai_api_key
    model_default: str = settings.openai_model
    fallback_model_default: Optional[str] = getattr(settings, "openai_fallback_model", None)
    model_qa: Optional[str] = getattr(settings, "openai_model_qa", None)
    model_code: Optional[str] = getattr(settings, "openai_model_code", None)
    fallback_model_qa: Optional[str] = getattr(settings, "openai_fallback_model_qa", None)
    fallback_model_code: Optional[str] = getattr(settings, "openai_fallback_model_code", None)

    _disabled_until: float = field(default=0.0, init=False, repr=False)
    _disabled_reason: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self.backend = "openai"

    def _cooldown_remaining_s(self) -> float:
        return max(0.0, float(self._disabled_until or 0.0) - time.time())

    def _set_cooldown(self, *, seconds: float, reason: str) -> None:
        seconds = max(0.0, float(seconds or 0.0))
        if seconds <= 0.0:
            return
        self._disabled_until = max(self._disabled_until, time.time() + seconds)
        self._disabled_reason = str(reason or "temporary OpenAI backend disable").strip()
        log.warning("OpenAI backend disabled for %.0fs: %s", seconds, self._disabled_reason)

    def _clear_cooldown(self) -> None:
        self._disabled_until = 0.0
        self._disabled_reason = ""

    def _client(self) -> Tuple[str, Any]:
        if self._cooldown_remaining_s() > 0:
            raise LLMUnavailable(
                f"OpenAI backend temporarily disabled for {int(self._cooldown_remaining_s())}s: {self._disabled_reason}"
            )

        if not self.api_key:
            raise LLMUnavailable("OPENAI_API_KEY not set")

        os.environ.setdefault("OPENAI_API_KEY", self.api_key)

        try:
            from openai import OpenAI  # type: ignore

            timeout_s = float(settings.llm_timeout_s or 60)
            try:
                client = OpenAI(api_key=self.api_key, timeout=timeout_s, max_retries=0)
            except TypeError:
                try:
                    client = OpenAI(api_key=self.api_key, timeout=timeout_s)
                except TypeError:
                    client = OpenAI(api_key=self.api_key)
            return "new", client
        except Exception:
            pass

        try:
            import openai  # type: ignore
        except Exception as e:
            raise LLMUnavailable(f"OpenAI SDK unavailable: {e}")

        try:
            openai.api_key = self.api_key
        except Exception:
            pass
        return "legacy", openai

    def _models_for(self, *, purpose: Optional[str] = None) -> Tuple[str, Optional[str]]:
        p = (purpose or "qa").strip().lower()
        if p == "code":
            primary = (self.model_code or "").strip() or (self.model_default or "").strip()
            fallback = (self.fallback_model_code or "").strip() or (self.fallback_model_default or "").strip() or None
            return primary, fallback

        primary = (self.model_qa or "").strip() or (self.model_default or "").strip()
        fallback = (self.fallback_model_qa or "").strip() or (self.fallback_model_default or "").strip() or None
        return primary, fallback

    def _status_code(self, err: BaseException) -> Optional[int]:
        code = getattr(err, "status_code", None)
        if code is None:
            response = getattr(err, "response", None)
            code = getattr(response, "status_code", None)
        try:
            return int(code) if code is not None else None
        except Exception:
            return None

    def _error_payload(self, err: BaseException) -> Any:
        body = getattr(err, "body", None)
        if body is not None:
            return body
        response = getattr(err, "response", None)
        if response is None:
            return None
        try:
            return response.json()
        except Exception:
            pass
        try:
            return response.text
        except Exception:
            return None

    def _error_code(self, err: BaseException) -> Optional[str]:
        code = getattr(err, "code", None)
        if isinstance(code, str) and code.strip():
            return code.strip()

        payload = self._error_payload(err)
        if isinstance(payload, dict):
            top_code = payload.get("code")
            if isinstance(top_code, str) and top_code.strip():
                return top_code.strip()
            err_obj = payload.get("error")
            if isinstance(err_obj, dict):
                nested = err_obj.get("code") or err_obj.get("type")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()

        msg = str(err or "")
        if "insufficient_quota" in msg:
            return "insufficient_quota"
        return None

    def _error_message(self, err: BaseException) -> str:
        payload = self._error_payload(err)
        if isinstance(payload, dict):
            err_obj = payload.get("error")
            if isinstance(err_obj, dict):
                msg = err_obj.get("message") or err_obj.get("type")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
        return str(err or "OpenAI request failed")

    def _is_insufficient_quota(self, err: BaseException) -> bool:
        code = str(self._error_code(err) or "").strip().lower()
        msg = self._error_message(err).lower()
        return code == "insufficient_quota" or "insufficient_quota" in msg or "exceeded your current quota" in msg

    def _should_try_fallback(
        self,
        err: BaseException,
        *,
        primary_model: str,
        fallback_model: Optional[str],
    ) -> bool:
        fb = (fallback_model or "").strip() or None
        if not fb or fb == (primary_model or "").strip():
            return False
        if self._is_insufficient_quota(err):
            return False
        if isinstance(err, (asyncio.TimeoutError, TimeoutError)):
            return True
        status = self._status_code(err)
        if status in {401, 403}:
            return False
        if status is None:
            return True
        return int(status) in {400, 404, 408, 409, 429, 500, 502, 503, 504}

    def _note_failure(self, err: BaseException, *, model: str) -> None:
        status = self._status_code(err)
        message = self._error_message(err)
        if self._is_insufficient_quota(err):
            self._set_cooldown(
                seconds=float(getattr(settings, "openai_insufficient_quota_cooldown_s", 900) or 900),
                reason=f"OpenAI insufficient_quota while calling {model}: {message}",
            )
            return
        if status in {401, 403}:
            self._set_cooldown(
                seconds=300.0,
                reason=f"OpenAI authentication/permission error while calling {model}: {message}",
            )

    def _chat_text_with_model(
        self,
        *,
        model: str,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> str:
        sdk_kind, client = self._client()

        if sdk_kind == "new":
            resp = client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                messages=[{"role": "system", "content": system}, *messages],
                temperature=temperature,
            )
            self._clear_cooldown()
            return (resp.choices[0].message.content or "").strip()

        resp = client.ChatCompletion.create(  # type: ignore[attr-defined]
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=temperature,
            request_timeout=float(settings.llm_timeout_s or 60),
        )
        self._clear_cooldown()
        try:
            return str(resp["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return str(resp)

    def is_available(self) -> bool:
        if self._cooldown_remaining_s() > 0:
            return False
        try:
            self._client()
            return True
        except Exception:
            return False

    def chat_text(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        purpose: Optional[str] = None,
    ) -> str:
        primary, fallback = self._models_for(purpose=purpose)
        try:
            return self._chat_text_with_model(
                model=primary,
                system=system,
                messages=messages,
                temperature=temperature,
            )
        except Exception as e:
            self._note_failure(e, model=primary)
            if self._should_try_fallback(e, primary_model=primary, fallback_model=fallback):
                fb = str(fallback)
                log.warning("OpenAI primary model '%s' failed; retrying with fallback '%s': %s", primary, fb, e)
                try:
                    return self._chat_text_with_model(
                        model=fb,
                        system=system,
                        messages=messages,
                        temperature=temperature,
                    )
                except Exception as e2:
                    self._note_failure(e2, model=fb)
                    raise
            raise

    def chat_json(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        purpose: Optional[str] = None,
    ) -> Any:
        primary, fallback = self._models_for(purpose=purpose)
        try:
            text = self._chat_text_with_model(
                model=primary,
                system=system,
                messages=messages,
                temperature=temperature,
            )
            return safe_json_loads(text)
        except Exception as e:
            self._note_failure(e, model=primary)
            if self._should_try_fallback(e, primary_model=primary, fallback_model=fallback):
                fb = str(fallback)
                log.warning("OpenAI primary JSON failed; retrying with fallback '%s': %s", fb, e)
                try:
                    text2 = self._chat_text_with_model(
                        model=fb,
                        system=system,
                        messages=messages,
                        temperature=temperature,
                    )
                    return safe_json_loads(text2)
                except Exception as e2:
                    self._note_failure(e2, model=fb)
                    raise
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
        t = float(timeout_s or settings.llm_timeout_s or 60)
        hard = max(t + 5.0, 15.0)
        primary, fallback = self._models_for(purpose=purpose)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._chat_text_with_model,
                    model=primary,
                    system=system,
                    messages=messages,
                    temperature=temperature,
                ),
                timeout=hard,
            )
        except Exception as e:
            self._note_failure(e, model=primary)
            if self._should_try_fallback(e, primary_model=primary, fallback_model=fallback):
                fb = str(fallback)
                log.warning(
                    "OpenAI async primary model '%s' failed; retrying with fallback '%s': %s",
                    primary,
                    fb,
                    e,
                )
                try:
                    return await asyncio.wait_for(
                        asyncio.to_thread(
                            self._chat_text_with_model,
                            model=fb,
                            system=system,
                            messages=messages,
                            temperature=temperature,
                        ),
                        timeout=hard,
                    )
                except Exception as e2:
                    self._note_failure(e2, model=fb)
                    raise
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
        t = float(timeout_s or settings.llm_timeout_s or 60)
        hard = max(t + 5.0, 15.0)
        primary, fallback = self._models_for(purpose=purpose)
        try:
            txt = await asyncio.wait_for(
                asyncio.to_thread(
                    self._chat_text_with_model,
                    model=primary,
                    system=system,
                    messages=messages,
                    temperature=temperature,
                ),
                timeout=hard,
            )
            return safe_json_loads(txt)
        except Exception as e:
            self._note_failure(e, model=primary)
            if self._should_try_fallback(e, primary_model=primary, fallback_model=fallback):
                fb = str(fallback)
                log.warning("OpenAI async primary JSON failed; retrying with fallback '%s': %s", fb, e)
                try:
                    txt2 = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._chat_text_with_model,
                            model=fb,
                            system=system,
                            messages=messages,
                            temperature=temperature,
                        ),
                        timeout=hard,
                    )
                    return safe_json_loads(txt2)
                except Exception as e2:
                    self._note_failure(e2, model=fb)
                    raise
            raise
