from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.core.config import settings
from app.llm.common import LLMUnavailable, safe_json_loads


log = logging.getLogger(__name__)


@dataclass
class GeminiClient:
    """Gemini REST client.

    Uses the official Gemini generateContent REST API so the project does not
    need an additional SDK dependency just to support provider fallback.
    """

    api_key: Optional[str] = settings.gemini_api_key
    model_default: str = settings.gemini_model
    fallback_model_default: Optional[str] = getattr(settings, "gemini_fallback_model", None)
    model_qa: Optional[str] = getattr(settings, "gemini_model_qa", None)
    model_code: Optional[str] = getattr(settings, "gemini_model_code", None)
    fallback_model_qa: Optional[str] = getattr(settings, "gemini_fallback_model_qa", None)
    fallback_model_code: Optional[str] = getattr(settings, "gemini_fallback_model_code", None)
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"

    _disabled_until: float = field(default=0.0, init=False, repr=False)
    _disabled_reason: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self.backend = "gemini"

    def _cooldown_remaining_s(self) -> float:
        return max(0.0, float(self._disabled_until or 0.0) - time.time())

    def _set_cooldown(self, *, seconds: float, reason: str) -> None:
        seconds = max(0.0, float(seconds or 0.0))
        if seconds <= 0.0:
            return
        current_remaining = self._cooldown_remaining_s()
        if current_remaining >= seconds and str(reason or "") == str(self._disabled_reason or ""):
            return
        self._disabled_until = time.time() + seconds
        self._disabled_reason = str(reason or "temporary Gemini backend disable").strip()
        log.warning("Gemini backend disabled for %.0fs: %s", seconds, self._disabled_reason)

    def _clear_cooldown(self) -> None:
        self._disabled_until = 0.0
        self._disabled_reason = ""

    def _models_for(self, *, purpose: Optional[str] = None) -> Tuple[str, Optional[str]]:
        p = (purpose or "qa").strip().lower()
        if p == "code":
            primary = (self.model_code or "").strip() or (self.model_default or "").strip()
            fallback = (self.fallback_model_code or "").strip() or (self.fallback_model_default or "").strip() or None
            return primary, fallback

        primary = (self.model_qa or "").strip() or (self.model_default or "").strip()
        fallback = (self.fallback_model_qa or "").strip() or (self.fallback_model_default or "").strip() or None
        return primary, fallback

    def is_available(self) -> bool:
        if self._cooldown_remaining_s() > 0:
            return False
        return bool((self.api_key or "").strip())

    def _status_code(self, err: BaseException) -> Optional[int]:
        response = getattr(err, "response", None)
        code = getattr(response, "status_code", None)
        try:
            return int(code) if code is not None else None
        except Exception:
            return None

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

        if isinstance(err, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
            return True

        status = self._status_code(err)
        if status in {401, 403}:
            return False
        if status is None:
            return True
        return int(status) in {400, 404, 408, 409, 429, 500, 502, 503, 504}

    def _build_payload(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> Dict[str, Any]:
        contents: List[Dict[str, Any]] = []
        for msg in messages or []:
            role = str(msg.get("role") or "user").strip().lower()
            content = str(msg.get("content") or "")
            mapped_role = "model" if role == "assistant" else "user"
            contents.append({"role": mapped_role, "parts": [{"text": content}]})

        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": float(temperature)},
        }
        if str(system or "").strip():
            payload["system_instruction"] = {"parts": [{"text": str(system)}]}
        return payload

    def _extract_text(self, data: Dict[str, Any]) -> str:
        texts: List[str] = []
        for cand in data.get("candidates") or []:
            content = cand.get("content") or {}
            for part in content.get("parts") or []:
                txt = part.get("text")
                if isinstance(txt, str) and txt.strip():
                    texts.append(txt)
        out = "\n".join(t.strip() for t in texts if t and t.strip()).strip()
        if out:
            return out
        raise ValueError(f"Gemini response did not contain text: {data}")

    def _truncate_for_log(self, text: Any, *, limit: int = 1200) -> str:
        raw = str(text or "").strip()
        if len(raw) <= limit:
            return raw
        return raw[: limit - 3] + "..."

    def _should_retry_status(self, status: int) -> bool:
        return int(status) in {429, 500, 502, 503, 504}

    def _request_json_with_model(
        self,
        *,
        model: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._cooldown_remaining_s() > 0:
            raise LLMUnavailable(
                f"Gemini backend temporarily disabled for {int(self._cooldown_remaining_s())}s: {self._disabled_reason}"
            )
        if not self.api_key:
            raise LLMUnavailable("GEMINI_API_KEY / GOOGLE_API_KEY not set")

        url = f"{self.api_base}/models/{model}:generateContent"
        timeout_s = float(settings.llm_timeout_s or 60)
        retries = max(0, int(getattr(settings, "llm_backend_retry_attempts", 2) or 0))
        backoff = max(0.1, float(getattr(settings, "llm_backend_retry_backoff_s", 1.0) or 1.0))

        for attempt in range(retries + 1):
            try:
                with httpx.Client(timeout=timeout_s) as client:
                    resp = client.post(
                        url,
                        headers={
                            "x-goog-api-key": self.api_key,
                            "content-type": "application/json",
                        },
                        json=payload,
                    )
            except httpx.TimeoutException as e:
                if attempt < retries:
                    delay = backoff * (2**attempt)
                    log.warning(
                        "Gemini request timed out (model=%s, attempt=%s/%s). Retrying in %.1fs.",
                        model,
                        attempt + 1,
                        retries + 1,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                self._set_cooldown(
                    seconds=float(getattr(settings, "llm_backend_error_cooldown_s", 300.0) or 300.0),
                    reason=f"Gemini timeout while calling {model}: {e}",
                )
                raise

            if resp.is_success:
                self._clear_cooldown()
                return resp.json()

            status = int(resp.status_code)
            body = self._truncate_for_log(resp.text)
            log.warning(
                "Gemini request failed (model=%s, status=%s, attempt=%s/%s): %s",
                model,
                status,
                attempt + 1,
                retries + 1,
                body,
            )

            if status in {401, 403}:
                self._set_cooldown(
                    seconds=float(getattr(settings, "llm_backend_error_cooldown_s", 300.0) or 300.0),
                    reason=f"Gemini authentication/permission error while calling {model}: {body}",
                )
            elif attempt >= retries and self._should_retry_status(status):
                self._set_cooldown(
                    seconds=float(getattr(settings, "llm_backend_error_cooldown_s", 300.0) or 300.0),
                    reason=f"Gemini HTTP {status} while calling {model}: {body}",
                )

            if attempt < retries and self._should_retry_status(status):
                delay = backoff * (2**attempt)
                log.info(
                    "Retrying Gemini request (model=%s, status=%s) in %.1fs.",
                    model,
                    status,
                    delay,
                )
                time.sleep(delay)
                continue

            resp.raise_for_status()

        raise LLMUnavailable(f"Gemini request failed for model '{model}' with no response")

    def _chat_text_with_model(
        self,
        *,
        model: str,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> str:
        payload = self._build_payload(system=system, messages=messages, temperature=temperature)
        data = self._request_json_with_model(model=model, payload=payload)
        return self._extract_text(data)

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
            if self._should_try_fallback(e, primary_model=primary, fallback_model=fallback):
                fb = str(fallback)
                log.warning("Gemini primary model '%s' failed; retrying with fallback '%s': %s", primary, fb, e)
                return self._chat_text_with_model(
                    model=fb,
                    system=system,
                    messages=messages,
                    temperature=temperature,
                )
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
            if self._should_try_fallback(e, primary_model=primary, fallback_model=fallback):
                fb = str(fallback)
                log.warning("Gemini primary JSON failed; retrying with fallback '%s': %s", fb, e)
                text2 = self._chat_text_with_model(
                    model=fb,
                    system=system,
                    messages=messages,
                    temperature=temperature,
                )
                return safe_json_loads(text2)
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
            if self._should_try_fallback(e, primary_model=primary, fallback_model=fallback):
                fb = str(fallback)
                log.warning(
                    "Gemini async primary model '%s' failed; retrying with fallback '%s': %s",
                    primary,
                    fb,
                    e,
                )
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
            if self._should_try_fallback(e, primary_model=primary, fallback_model=fallback):
                fb = str(fallback)
                log.warning("Gemini async primary JSON failed; retrying with fallback '%s': %s", fb, e)
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
            raise
