from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.$/@:#-]+|[^\w\s]", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(str(text or "")) if m.group(0).strip()]


def _token_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _TOKEN_RE.finditer(str(text or "")) if m.group(0).strip()]


def _tail_ngram_repeat_count(tokens: List[str], n: int) -> int:
    if n <= 0 or len(tokens) < n * 2:
        return 1
    pattern = tokens[-n:]
    count = 1
    idx = len(tokens) - (2 * n)
    while idx >= 0 and tokens[idx : idx + n] == pattern:
        count += 1
        idx -= n
    return count


def _has_repeated_line_loop(text: str) -> bool:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if len(lines) < 6:
        return False

    run = 1
    for i in range(1, len(lines)):
        if lines[i] == lines[i - 1] and len(lines[i]) >= 2:
            run += 1
            if run >= 4:
                return True
        else:
            run = 1

    counts = Counter(lines)
    most = counts.most_common(1)[0][1] if counts else 0
    return most >= 5 and most / max(1, len(lines)) >= 0.55


def _has_repeated_token_loop(tokens: List[str]) -> bool:
    if len(tokens) < 16:
        return False

    # Long run of the exact same token, e.g. "the the the..." or "} } }...".
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run >= 10:
                return True
        else:
            run = 1

    # Low-token-variety collapse.
    counts = Counter(tokens)
    top = counts.most_common(1)[0][1] if counts else 0
    if top >= 12 and top / max(1, len(tokens)) >= 0.55:
        return True

    recent = tokens[-48:]
    if len(recent) >= 24:
        r_counts = Counter(recent)
        r_top = r_counts.most_common(1)[0][1] if r_counts else 0
        if r_top >= 16 and r_top / max(1, len(recent)) >= 0.60:
            return True
        if len(set(recent)) <= 4 and len(recent) >= 32:
            return True

    # Repeated phrase tail, e.g. "return JSON only return JSON only...".
    for n in range(1, min(12, len(tokens) // 3) + 1):
        repeats = _tail_ngram_repeat_count(tokens, n)
        if n == 1 and repeats >= 10:
            return True
        if 2 <= n <= 4 and repeats >= 6:
            return True
        if n >= 5 and repeats >= 4:
            return True

    # Repeated phrase anywhere in the output.
    for n in range(2, 9):
        if len(tokens) < n * 5:
            continue
        run = 1
        prev = tokens[0:n]
        for i in range(n, len(tokens) - n + 1, n):
            cur = tokens[i : i + n]
            if cur == prev:
                run += 1
                if run >= 5:
                    return True
            else:
                run = 1
                prev = cur

    return False


def is_repetitive_text(text: str, *, min_chars: int = 24) -> bool:
    """Return True when text looks like a local-model repetition collapse.

    This intentionally catches character loops, word/token loops, repeated JSON
    braces, repeated lines, and short phrase cycles. It is conservative for
    normal code because valid code usually has far more token variety than a
    model-collapse loop.
    """
    s = str(text or "").strip()
    if len(s) < min_chars:
        return False

    compact = "".join(ch for ch in s if not ch.isspace())
    if len(compact) >= min_chars:
        # Punctuation/symbol collapse: !!!!!, }}}}}, =====, etc.
        if re.search(r"([^\w\s])\1{15,}", compact):
            return True

        counts = Counter(compact)
        most_common = counts.most_common(2)
        top = most_common[0][1] if most_common else 0
        top2 = sum(v for _, v in most_common)
        if top / max(1, len(compact)) >= 0.72:
            return True
        if len(set(compact)) <= 3 and top2 / max(1, len(compact)) >= 0.85:
            return True

    if _has_repeated_line_loop(s):
        return True

    toks = _tokens(s)
    if _has_repeated_token_loop(toks):
        return True

    return False


def truncate_repetitive_tail(text: str) -> str:
    """Trim a repeated token/phrase tail while preserving a useful prefix.

    If no obvious repeated tail exists, the original text is returned unchanged.
    """
    s = str(text or "")
    spans = _token_spans(s)
    toks = [t for t, _, _ in spans]
    if len(toks) < 12:
        return s

    best_start_idx = None
    best_repeats = 0
    for n in range(1, min(16, len(toks) // 3) + 1):
        repeats = _tail_ngram_repeat_count(toks, n)
        threshold = 8 if n == 1 else 4
        if repeats >= threshold and repeats > best_repeats:
            best_repeats = repeats
            best_start_idx = len(toks) - (n * repeats)

    if best_start_idx is None or best_start_idx <= 0:
        return s

    cut_at = spans[best_start_idx][1]
    trimmed = s[:cut_at].rstrip()
    return trimmed if trimmed else s


def sanitize_operation_content(content: str, *, max_chars: int | None = None) -> str:
    """Normalize generated file content and fail if it is a repetition loop."""
    text = str(content or "")
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
    text = truncate_repetitive_tail(text)
    if is_repetitive_text(text):
        raise ValueError("generated file content appears to be a repeated-token loop")
    return text


def sanitize_operations(operations: Iterable[object]) -> tuple[list[dict], int]:
    """Return operations with repeated-token write_file contents removed.

    The code pipeline consumes LLM-generated operations. This guard prevents a
    collapsed local model from writing repeated-token garbage into the user's
    project or storing it as a successful training example.
    """
    clean: list[dict] = []
    dropped = 0
    for op in operations or []:
        if not isinstance(op, dict):
            dropped += 1
            continue
        item = dict(op)
        if item.get("op") == "write_file":
            content = item.get("content")
            if not isinstance(content, str):
                dropped += 1
                continue
            try:
                item["content"] = sanitize_operation_content(content)
            except ValueError:
                dropped += 1
                continue
        clean.append(item)
    return clean, dropped
