from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.core.config import settings

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")

_DEFAULT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".md",
    ".txt",
    ".html",
    ".css",
    ".scss",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".ps1",
    ".java",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".cs",
    ".sql",
}

_DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".workspace",
    ".workspaces",
    ".agentic_hive_studio",
    ".memory",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".next",
    "coverage",
}


def _tokens(text: Any) -> List[str]:
    seen = set()
    out: List[str] = []
    for m in _TOKEN_RE.finditer(str(text or "").lower()):
        tok = m.group(0)
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _score_text(query_tokens: Sequence[str], text: str) -> float:
    if not query_tokens or not text:
        return 0.0
    low = text.lower()
    score = 0.0
    for tok in query_tokens:
        c = low.count(tok)
        if c:
            score += 1.0 + min(4, c) * 0.35
    # Reward phrase overlap slightly.
    phrase = " ".join(query_tokens[:5])
    if len(phrase) > 8 and phrase in low:
        score += 2.0
    return score


def _truncate(text: Any, max_chars: int) -> str:
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 18)] + "\n...<truncated>..."


def _memory_content(row: Dict[str, Any]) -> str:
    parts = [str(row.get("content") or "")]
    tags = row.get("tags")
    if isinstance(tags, list):
        parts.append(" ".join(str(x) for x in tags))
    elif tags:
        parts.append(str(tags))
    if row.get("agent_type"):
        parts.append(str(row.get("agent_type")))
    return "\n".join(parts)


def _rank_memories(*, query: str, memory_store: Any) -> List[Dict[str, Any]]:
    q_tokens = _tokens(query)
    limit = int(getattr(settings, "rag_memory_scan_limit", 350) or 350)
    desired = int(getattr(settings, "rag_memory_limit", 8) or 8)
    rows: List[Dict[str, Any]] = []

    try:
        rows.extend(memory_store.recent_all(limit=limit, scopes=["hive", "type", "agent"]) or [])
    except Exception:
        rows = []

    # Existing SQL search is exact-LIKE based. It is still useful for phrase hits.
    try:
        for r in memory_store.search_all(query=query, limit=desired * 2, scopes=["hive", "type", "agent"]) or []:
            rows.append(r)
    except Exception:
        pass

    scored: List[Tuple[float, Dict[str, Any]]] = []
    seen_ids = set()
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        text = _memory_content(row)
        score = _score_text(q_tokens, text)
        if score <= 0:
            continue
        # Tiny recency boost because recent conversational context is often relevant.
        score += max(0.0, 0.4 - (idx / max(1, limit)) * 0.4)
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for score, row in scored[:desired]:
        out.append(
            {
                "id": row.get("id"),
                "scope": row.get("scope"),
                "agent_type": row.get("agent_type"),
                "created_at": row.get("created_at"),
                "tags": row.get("tags") or [],
                "score": round(float(score), 3),
                "snippet": _truncate(row.get("content") or "", 900),
            }
        )
    return out


def _iter_project_files(project_root: Path) -> Iterable[Path]:
    skip_dirs = set(_DEFAULT_SKIP_DIRS)
    try:
        skip_dirs.update(str(x).strip() for x in getattr(settings, "rag_skip_dirs", []) or [] if str(x).strip())
    except Exception:
        pass
    max_bytes = int(getattr(settings, "rag_max_file_bytes", 200000) or 200000)

    try:
        root = Path(project_root).resolve()
    except Exception:
        return []

    if not root.exists() or not root.is_dir():
        return []

    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            entries = sorted(cur.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception:
            continue
        for entry in entries:
            name = entry.name
            if entry.is_dir():
                if name in skip_dirs or name.startswith(".") and name not in {".github"}:
                    continue
                stack.append(entry)
                continue
            if name.startswith(".") and entry.suffix.lower() not in {".env"}:
                continue
            if entry.suffix.lower() not in _DEFAULT_EXTENSIONS and name not in {"Dockerfile", "Makefile", "requirements.txt", "package.json"}:
                continue
            try:
                if entry.stat().st_size > max_bytes:
                    continue
            except Exception:
                continue
            yield entry


def _best_file_snippet(text: str, q_tokens: Sequence[str], *, max_chars: int = 1400) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if not lines:
        return _truncate(text, max_chars)

    best_idx = 0
    best_score = -1.0
    for i, line in enumerate(lines):
        sc = _score_text(q_tokens, line)
        if sc > best_score:
            best_score = sc
            best_idx = i

    start = max(0, best_idx - 8)
    end = min(len(lines), best_idx + 14)
    numbered = [f"{i + 1}: {lines[i]}" for i in range(start, end)]
    return _truncate("\n".join(numbered), max_chars)


def _rank_project_files(*, query: str, project_root: Optional[Path]) -> List[Dict[str, Any]]:
    if not project_root:
        return []
    q_tokens = _tokens(query)
    desired = int(getattr(settings, "rag_project_file_limit", 10) or 10)
    scored: List[Tuple[float, Path, str]] = []

    for path in _iter_project_files(Path(project_root)):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = str(path.resolve().relative_to(Path(project_root).resolve()))
        path_score = _score_text(q_tokens, rel.replace("/", " ").replace("_", " ")) * 1.5
        body_score = _score_text(q_tokens, text)
        score = path_score + body_score
        if score <= 0:
            continue
        scored.append((score, path, text))

    scored.sort(key=lambda item: item[0], reverse=True)
    out: List[Dict[str, Any]] = []
    root = Path(project_root).resolve()
    for score, path, text in scored[:desired]:
        try:
            rel = str(path.resolve().relative_to(root))
        except Exception:
            rel = str(path)
        out.append(
            {
                "path": rel,
                "score": round(float(score), 3),
                "snippet": _best_file_snippet(text, q_tokens),
            }
        )
    return out


def _assemble_context_text(memory_matches: List[Dict[str, Any]], file_matches: List[Dict[str, Any]], max_chars: int) -> str:
    parts: List[str] = []
    if file_matches:
        parts.append("Relevant project files:")
        for item in file_matches:
            parts.append(f"\n[project:{item.get('path')}] score={item.get('score')}\n{item.get('snippet') or ''}")
    if memory_matches:
        parts.append("\nRelevant hive memories / prior backend outputs:")
        for item in memory_matches:
            parts.append(
                f"\n[memory:{item.get('scope')}#{item.get('id')}] score={item.get('score')} created_at={item.get('created_at')}\n{item.get('snippet') or ''}"
            )
    return _truncate("\n".join(parts).strip(), max_chars)


def build_rag_context(
    *,
    query: str,
    memory_store: Any,
    project_root: Optional[Path] = None,
    mode: str = "qa",
) -> Dict[str, Any]:
    """Build a compact retrieval-augmented context without external services.

    This deliberately avoids embedding-model dependencies so it works immediately
    on Windows and CPU-only setups. It ranks recent memories, hosted-backend
    outputs, upload chunks, and current project files by token overlap.
    """

    if not bool(getattr(settings, "rag_enabled", True)):
        return {"enabled": False, "reason": "disabled"}

    max_chars = int(getattr(settings, "rag_max_chars", 9000) or 9000)
    memory_matches = _rank_memories(query=query, memory_store=memory_store)
    project_file_matches = _rank_project_files(query=query, project_root=project_root)
    context_text = _assemble_context_text(memory_matches, project_file_matches, max_chars=max_chars)
    return {
        "enabled": True,
        "mode": mode,
        "query": query,
        "project_root": str(project_root) if project_root else None,
        "memory_matches": memory_matches,
        "project_file_matches": project_file_matches,
        "context_text": context_text,
        "stats": {
            "memory_matches": len(memory_matches),
            "project_file_matches": len(project_file_matches),
            "chars": len(context_text),
        },
    }
