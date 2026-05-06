from __future__ import annotations

import json
import re
from typing import Any, Optional

from app.core.config import settings


class LLMUnavailable(RuntimeError):
    """Raised when the configured LLM backend cannot be used."""


class ProviderHTTPError(RuntimeError):
    """Normalized HTTP/provider error with optional structured metadata."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        response_body: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.response_body = response_body
        self.provider = provider


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)


def safe_json_loads(text: str) -> Any:
    """Best-effort JSON extraction.

    Some models will wrap JSON in surrounding text. We try to:
    - parse raw
    - parse fenced blocks
    - extract the first {...} JSON object
    """

    if text is None:
        raise ValueError("text is None")

    t = str(text).strip()
    if not t:
        raise ValueError("empty JSON")

    # Common case: plain JSON
    try:
        return json.loads(t)
    except Exception:
        pass

    # Strip markdown fences
    t2 = t
    if "```" in t2:
        t2 = t2.replace("```json", "```").replace("```JSON", "```")
        parts = [p.strip() for p in t2.split("```") if p.strip()]
        # try each fenced chunk
        for p in parts:
            try:
                return json.loads(p)
            except Exception:
                continue

    # Extract first JSON object-like block
    m = _JSON_BLOCK_RE.search(t)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # As a last resort, try to find the first '[' ... ']' array
    if "[" in t and "]" in t:
        start = t.find("[")
        end = t.rfind("]")
        if 0 <= start < end:
            candidate = t[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

    raise ValueError("Could not parse JSON from model output")


def get_status_code(err: BaseException) -> Optional[int]:
    status = getattr(err, "status_code", None)
    if status is None:
        response = getattr(err, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except Exception:
        return None


def _response_obj(err: BaseException) -> Any:
    return getattr(err, "response", None)


def _truncate_text(text: str, limit: Optional[int] = None) -> str:
    max_len = int(limit or getattr(settings, "llm_provider_error_body_chars", None) or getattr(settings, "llm_error_body_log_chars", 1200) or 1200)
    t = str(text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 16] + " ...<truncated>"


def extract_error_code(err: BaseException) -> Optional[str]:
    code = getattr(err, "error_code", None) or getattr(err, "code", None)
    if code:
        return str(code)

    response = _response_obj(err)
    try:
        if response is not None:
            data = response.json()
            if isinstance(data, dict):
                err_obj = data.get("error") if isinstance(data.get("error"), dict) else data
                for key in ("code", "status", "type"):
                    value = err_obj.get(key) if isinstance(err_obj, dict) else None
                    if value:
                        return str(value)
    except Exception:
        pass
    return None


def extract_error_body(err: BaseException, *, limit: Optional[int] = None) -> Optional[str]:
    response = _response_obj(err)
    if response is not None:
        try:
            data = response.json()
            if data is not None:
                return _truncate_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), limit)
        except Exception:
            pass
        try:
            text = response.text
            if text:
                return _truncate_text(text, limit)
        except Exception:
            pass

    text = str(err or "").strip()
    return _truncate_text(text, limit) if text else None


def format_provider_error(provider: str, err: BaseException, *, limit: Optional[int] = None) -> str:
    status = get_status_code(err)
    code = extract_error_code(err)
    body = extract_error_body(err, limit=limit)

    parts = [f"{provider} request failed"]
    if status is not None:
        parts.append(f"status={status}")
    if code:
        parts.append(f"code={code}")
    if body:
        parts.append(f"body={body}")
    return "; ".join(parts)


def as_provider_error(provider: str, err: BaseException, *, limit: Optional[int] = None) -> ProviderHTTPError:
    return ProviderHTTPError(
        format_provider_error(provider, err, limit=limit),
        status_code=get_status_code(err),
        error_code=extract_error_code(err),
        response_body=extract_error_body(err, limit=limit),
        provider=provider,
    )


def is_insufficient_quota(err: BaseException) -> bool:
    code = (extract_error_code(err) or "").strip().lower()
    if code == "insufficient_quota":
        return True

    body = (extract_error_body(err, limit=2000) or "").strip().lower()
    phrases = [
        "insufficient_quota",
        "you exceeded your current quota",
        "run out of credits",
        "maximum monthly spend",
    ]
    return any(p in body for p in phrases)
