from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


_ORDINAL_RE = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)"
_POINT_MARKER_RE = re.compile(rf"\b{_ORDINAL_RE}\s*(?:point|bullet)\b", re.IGNORECASE)
_POINT_NUMBER_RE = re.compile(
    r"\bpoint\s*(?:number\s*)?(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
    re.IGNORECASE,
)

_FOLLOWING_POINTS_RE = re.compile(r"\b(?:the\s+)?following\s+(?:bullet\s+)?points\b", re.IGNORECASE)
_END_LIST_RE = re.compile(r"\b(?:end|stop)\s+(?:list|points|bullets|options|choices)\b", re.IGNORECASE)

_OPTION_TRIGGER_RE = re.compile(r"\b(?:the\s+)?(?:following\s+)?(?:answer\s+)?(?:choices?|options?)\b", re.IGNORECASE)
_OPTION_MARKER_RE = re.compile(
    r"\b(?:option|choice)\s*(?:number\s*)?([a-j]|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
    re.IGNORECASE,
)
_OPTION_WORD_TO_LABEL = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

_PUNCT_MAP = [
    (re.compile(r"\bcomma\b", re.IGNORECASE), ","),
    (re.compile(r"\b(full\s+stop|period)\b", re.IGNORECASE), "."),
    (re.compile(r"\bquestion\s+mark\b", re.IGNORECASE), "?"),
    (re.compile(r"\bexclamation\s+mark\b", re.IGNORECASE), "!"),
    (re.compile(r"\bnew\s+line\b", re.IGNORECASE), "\n"),
    (re.compile(r"\bunderscore\b", re.IGNORECASE), "_"),
    (re.compile(r"\b(dash|hyphen)\b", re.IGNORECASE), "-"),
    (re.compile(r"\bat\s+sign\b", re.IGNORECASE), "@"),
]


def _clean_list_fragment(text: str) -> str:
    return (text or "").strip(" \n\t-:")


def _option_label(raw: str) -> str:
    token = str(raw or "").strip()
    low = token.lower()
    if low in _OPTION_WORD_TO_LABEL:
        return _OPTION_WORD_TO_LABEL[low]
    if len(token) == 1 and token.isalpha():
        return token.upper()
    if low.isdigit():
        return str(int(low))
    return token.upper() or "?"


def split_intro_and_options(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    t = (text or "").strip()
    if not t:
        return "", []

    def repl(m: re.Match[str]) -> str:
        return f"\n<<<OPT:{_option_label(m.group(1))}>>>\n"

    t2 = _OPTION_MARKER_RE.sub(repl, t)
    if "<<<OPT:" not in t2:
        return t, []

    intro_part, *rest = t2.split("<<<OPT:")
    intro = _clean_list_fragment(intro_part)
    options: List[Tuple[str, str]] = []
    for part in rest:
        try:
            label, body = part.split(">>>", 1)
        except ValueError:
            continue
        body = _clean_list_fragment(body)
        if body:
            options.append((label.strip(), body))
    return intro, options


def _cleanup_artifacts(text: str) -> str:
    text = re.sub(r"([.!?])\s*\d+\b", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _merge_spelled_letters(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        letters = re.findall(r"[A-Za-z]", m.group(0))
        return "".join(letters)

    return re.sub(r"(?:\b[A-Za-z]\b(?:\s+|$)){3,}", repl, text)


def normalize_transcript(text: str, *, mode: str = "qa") -> str:
    t = (text or "").strip()
    if not t:
        return ""
    for rx, rep in _PUNCT_MAP:
        t = rx.sub(rep, t)
    t = re.sub(r"\b(dot)\s+(com|net|org|io|ai)\b", r".\2", t, flags=re.IGNORECASE)
    t = _merge_spelled_letters(t)
    t = _cleanup_artifacts(t)
    return t


def apply_replacements(text: str, replacements: List[Dict[str, str]]) -> str:
    if not text or not replacements:
        return text
    out = text
    reps = sorted(replacements, key=lambda r: len(r.get("src", "")), reverse=True)
    for r in reps[:200]:
        src = (r.get("src") or "").strip()
        dst = (r.get("dst") or "").strip()
        if not src or not dst:
            continue
        if re.fullmatch(r"[A-Za-z0-9_]+", src):
            out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
        else:
            out = re.sub(re.escape(src), dst, out, flags=re.IGNORECASE)
    return out


def split_intro_and_points(text: str) -> Tuple[str, List[str]]:
    t = (text or "").strip()
    if not t:
        return "", []
    sep = "\n<<<PT>>>\n"
    t2 = _POINT_MARKER_RE.sub(sep, t)
    t2 = _POINT_NUMBER_RE.sub(sep, t2)
    if "<<<PT>>>" not in t2:
        return t, []
    parts = [_clean_list_fragment(p) for p in t2.split("<<<PT>>>")]
    parts = [p for p in parts if p]
    if not parts:
        return "", []
    return parts[0], parts[1:]


def format_transcript(
    text: str,
    *,
    mode: str = "qa",
    replacements: List[Dict[str, str]] | None = None,
    list_mode_active: bool = False,
) -> Tuple[str, Dict[str, object]]:
    """Return (formatted_text_to_append, meta).

    Conservative behavior:
    - bullets only if explicit point markers exist or list_mode_active is true
    - options become a clean list when explicit option/choice markers exist
    """

    raw = normalize_transcript(text, mode=mode)
    if replacements:
        raw = apply_replacements(raw, replacements)

    meta: Dict[str, object] = {
        "triggered_list_mode": False,
        "triggered_option_mode": False,
        "used_bullets": False,
        "used_options": False,
    }

    if not raw:
        return "", meta

    if _END_LIST_RE.search(raw):
        raw2 = _END_LIST_RE.sub("", raw).strip()
        return raw2, {**meta, "exit_list_mode": True}

    triggered_points = False
    if _FOLLOWING_POINTS_RE.search(raw):
        triggered_points = True
        raw = _FOLLOWING_POINTS_RE.sub("", raw).strip(" ,:-")
        meta["triggered_list_mode"] = True

    if _OPTION_TRIGGER_RE.search(raw):
        raw = _OPTION_TRIGGER_RE.sub("", raw).strip(" ,:-")
        meta["triggered_option_mode"] = True

    intro_opt, options = split_intro_and_options(raw)
    if options:
        lines: List[str] = []
        if intro_opt:
            lines.append(intro_opt)
        for label, body in options:
            lines.append(f"- {label}. {body}")
        meta["used_options"] = True
        return "\n".join(lines).strip(), meta

    intro, points = split_intro_and_points(raw)
    if points:
        lines: List[str] = []
        if intro:
            lines.append(intro)
        for p in points:
            if p:
                lines.append(f"- {p}")
        meta["used_bullets"] = True
        return "\n".join(lines).strip(), meta

    if triggered_points or list_mode_active:
        word_count = len(re.findall(r"\w+", raw))
        if word_count > 18 and not triggered_points:
            return raw, {**meta, "auto_exit_list_mode": True}
        meta["used_bullets"] = True
        return f"- {raw}", meta

    return raw, meta


def extract_replacements(raw_text: str, final_text: str, *, max_phrase_words: int = 3) -> List[Tuple[str, str]]:
    """Extract conservative replacement pairs from raw -> final text."""
    raw = normalize_transcript(raw_text or "")
    fin = normalize_transcript(final_text or "")
    if not raw or not fin or raw == fin:
        return []

    a = raw.split()
    b = fin.split()
    sm = SequenceMatcher(a=a, b=b)

    out: List[Tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        old = a[i1:i2]
        new = b[j1:j2]
        if not old or not new:
            continue
        if len(old) > max_phrase_words or len(new) > max_phrase_words:
            continue
        src = " ".join(old).strip()
        dst = " ".join(new).strip()
        if len(src) < 3 or len(dst) < 3:
            continue
        if src.lower() == dst.lower():
            continue
        out.append((src, dst))
    return out
