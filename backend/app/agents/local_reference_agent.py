from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.agents.base import BaseAgent
from app.core.config import settings


def _is_probably_text(data: bytes) -> bool:
    # Heuristic: reject if it contains NUL bytes.
    return b"\x00" not in data


def _read_text_best_effort(path: Path, *, max_bytes: int = 512_000) -> str:
    data = path.read_bytes()[:max_bytes]
    if not _is_probably_text(data):
        return ""
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode(errors="ignore")


def _extract_keywords(query: str, *, max_keywords: int = 12) -> List[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "these",
        "those",
        "into",
        "able",
        "want",
        "feature",
        "agent",
        "agents",
        "orchestrator",
        "memory",
        "hive",
        "project",
        "code",
        "file",
        "files",
    }
    uniq: List[str] = []
    for w in words:
        if w in stop:
            continue
        if w not in uniq:
            uniq.append(w)
        if len(uniq) >= max_keywords:
            break
    return uniq


def _iter_files(root: Path, *, exts: Optional[List[str]], exclude_dir_names: List[str], max_files: int) -> Iterable[Path]:
    count = 0
    for p in root.rglob("*"):
        if count >= max_files:
            break
        if p.is_dir():
            continue
        if any(part in exclude_dir_names for part in p.parts):
            continue
        if exts is not None and p.suffix.lower() not in {e.lower() for e in exts}:
            continue
        # skip huge files
        try:
            if p.stat().st_size > 1_000_000:
                continue
        except Exception:
            continue
        count += 1
        yield p


def _score_text(text_lower: str, keywords: List[str]) -> int:
    score = 0
    for k in keywords:
        score += text_lower.count(k)
    return score


def _make_snippet(text: str, keyword: str, *, window: int = 200) -> str:
    idx = text.lower().find(keyword.lower())
    if idx < 0:
        return text[: window * 2]
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    return text[start:end].strip()


@dataclass
class ReferenceMatch:
    root: str
    path: str
    score: int
    snippet: str


class LocalReferenceAgent(BaseAgent):
    agent_type = "local_reference"

    async def find_references(
        self,
        *,
        query: str,
        project_root: Optional[Path] = None,
        reference_roots: Optional[List[Path]] = None,
        exts: Optional[List[str]] = None,
        max_files_per_root: int = 1500,
        max_matches: int = 20,
    ) -> Dict[str, Any]:
        """Search locally for code/docs patterns in *other* projects.

        This is meant for "reference" only: the agent returns snippets and paths.
        """
        self.set_state("searching")
        self.log("searching local reference projects")

        # Like other resource agents, local-reference is best-effort.
        # Failures should never crash the overall Q&A turn.
        try:
            return await self._find_impl(
                query=query,
                project_root=project_root,
                reference_roots=reference_roots,
                exts=exts,
                max_files_per_root=max_files_per_root,
                max_matches=max_matches,
            )
        finally:
            try:
                self.set_state("idle")
            except Exception:
                pass

    async def _find_impl(
        self,
        *,
        query: str,
        project_root: Optional[Path],
        reference_roots: Optional[List[Path]],
        exts: Optional[List[str]],
        max_files_per_root: int,
        max_matches: int,
    ) -> Dict[str, Any]:

        # Local scanning can be slow on large projects. Run the CPU / disk-bound
        # scan in a thread so we don't block the FastAPI event loop (which can
        # otherwise make the frontend look like it's "stuck" / "unable to fetch").

        roots = list(reference_roots or settings.reference_project_roots or [])
        # Optionally include current project root too (useful when you don't set REFERENCE_PROJECT_ROOTS)
        if project_root is not None and project_root not in roots:
            roots = [project_root, *roots]

        exts = exts or [".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"]
        keywords = _extract_keywords(query)
        exclude = settings.reference_project_exclude_dirs

        def _scan_sync() -> Dict[str, Any]:
            matches: List[ReferenceMatch] = []

            for root in roots:
                try:
                    rr = root.expanduser().resolve()
                except Exception:
                    continue
                if not rr.exists() or not rr.is_dir():
                    continue

                for fp in _iter_files(rr, exts=exts, exclude_dir_names=exclude, max_files=max_files_per_root):
                    try:
                        text = _read_text_best_effort(fp)
                    except Exception:
                        continue
                    if not text:
                        continue
                    tl = text.lower()

                    # quick filter
                    if keywords and not any(k in tl for k in keywords):
                        continue

                    score = _score_text(tl, keywords) if keywords else 1
                    if score <= 0:
                        continue

                    # snippet: pick first keyword present
                    snip_kw = next((k for k in keywords if k in tl), keywords[0] if keywords else "")
                    snippet = _make_snippet(text, snip_kw, window=220)
                    try:
                        rel = str(fp.relative_to(rr))
                    except Exception:
                        rel = str(fp)

                    matches.append(ReferenceMatch(root=str(rr), path=rel, score=score, snippet=snippet))

            matches.sort(key=lambda m: m.score, reverse=True)
            top = matches[:max_matches]

            return {
                "query": query,
                "keywords": keywords,
                "roots": [str(r) for r in roots],
                "matches": [m.__dict__ for m in top],
                "notes": "Set REFERENCE_PROJECT_ROOTS in backend/.env to add more local projects.",
            }

        out = await asyncio.to_thread(_scan_sync)

        self.latest_result = out
        self.remember(json.dumps(out), tags=["local_reference"], success=True)
        try:
            self.add_type_memory(json.dumps(out), tags=["local_reference"], success=True)
        except Exception as e:
            self.ctx.run_logger(f"local_reference:{self.agent_id} | type-memory write failed: {e}")

        # Compact to hive memory
        top_matches = (out.get("matches") or [])[:5]
        compact = {
            "query": query,
            "top_matches": [
                {"root": m.get("root"), "path": m.get("path"), "score": m.get("score")}
                for m in top_matches
                if isinstance(m, dict)
            ],
        }
        try:
            self.add_hive_memory(json.dumps(compact), tags=["local_reference"], success=True)
        except Exception as e:
            self.ctx.run_logger(f"local_reference:{self.agent_id} | hive-memory write failed: {e}")
        return out
