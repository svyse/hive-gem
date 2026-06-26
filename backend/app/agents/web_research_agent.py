from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.core.config import settings
from app.utils.web_utils import ddg_search, fetch_url, WebFetchError


WEB_SUMMARY_SYSTEM = """You are a web research assistant agent.

You will be given:
- a user query
- a set of fetched documents (URL + extracted text)
Your job:
- extract the most relevant facts and actionable details
- include sources by URL (do not invent sources)
- be concise and structured

Return a JSON object with:
{
  "key_points": ["..."],
  "data": [{"label": "...", "value": "...", "source_url": "..."}],
  "open_questions": ["..."]
}
"""


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "\n...<truncated>...\n"


def _active_llm_backend(ctx: Any) -> str:
    llm = getattr(ctx, "llm", None)
    return str(getattr(llm, "backend", getattr(settings, "llm_backend", "local")) or "local").lower()


def _extractive_summary(query: str, results: List[Dict[str, Any]], docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a deterministic summary without calling the LLM.

    Local code mode uses this by default so web research cannot re-enter the
    fragile local strict-JSON/repeated-token path. The code engine still gets
    useful titles, snippets, URLs, and small fetched excerpts.
    """
    key_points: List[str] = []
    data: List[Dict[str, str]] = []

    for result in results[:5]:
        title = str(result.get("title") or "").strip()
        snippet = str(result.get("snippet") or "").strip()
        url = str(result.get("url") or "").strip()
        if title or snippet:
            point = (title + (f": {snippet}" if snippet else "")).strip()
            key_points.append(point[:500])
        if url:
            data.append({"label": title[:120] or "source", "value": snippet[:300], "source_url": url})

    for doc in docs[:3]:
        url = str(doc.get("url") or "").strip()
        text = str(doc.get("text") or "").strip()
        if not text:
            continue
        first_lines = [line.strip() for line in text.splitlines() if line.strip()]
        excerpt = " ".join(first_lines[:4])[:600]
        if excerpt:
            key_points.append(excerpt)
            data.append({"label": "excerpt", "value": excerpt[:450], "source_url": url})

    return {
        "key_points": key_points[:8],
        "data": data[:8],
        "open_questions": [] if key_points or data else [f"No useful web text was retrieved for: {query}"],
        "summary_mode": "extractive_no_llm",
    }


class WebResearchAgent(BaseAgent):
    agent_type = "web_research"

    async def research(
        self,
        *,
        query: str,
        max_results: int | None = None,
        fetch_top_n: int | None = None,
        max_chars_per_doc: int = 8000,
    ) -> Dict[str, Any]:
        """Search the web and fetch the top documents."""
        self.set_state("researching")
        self.log(f"web research: {query}")

        # Web research is *best effort*. Any failure (network, HTML parsing,
        # sqlite locks, etc.) should never crash the Q&A request.
        #
        # We keep the agent resilient by:
        # - catching exceptions from external calls
        # - wrapping memory writes so DB issues don't bubble up
        # - always restoring state to idle
        try:
            return await self._research_impl(
                query=query,
                max_results=max_results,
                fetch_top_n=fetch_top_n,
                max_chars_per_doc=max_chars_per_doc,
            )
        finally:
            # Ensure state is consistent even on exceptions.
            try:
                self.set_state("idle")
            except Exception:
                pass

    async def _research_impl(
        self,
        *,
        query: str,
        max_results: int | None,
        fetch_top_n: int | None,
        max_chars_per_doc: int,
    ) -> Dict[str, Any]:

        max_results = int(max_results or settings.web_search_max_results)
        fetch_top_n = int(fetch_top_n or settings.web_fetch_top_n)

        try:
            results = await ddg_search(query, max_results=max_results)
        except Exception as e:
            # Record the failure as memory for later debugging.
            out = {"query": query, "error": str(e), "results": [], "documents": [], "summary": None}
            self.latest_result = out
            self.remember(json.dumps(out), tags=["web_research", "error"], success=False)
            try:
                self.add_type_memory(json.dumps(out), tags=["web_research", "error"], success=False)
            except Exception as me:
                self.ctx.run_logger(f"web_research:{self.agent_id} | type-memory write failed: {me}")
            return out

        docs: List[Dict[str, Any]] = []

        # Fetch in parallel (bounded)
        sem = asyncio.Semaphore(4)

        async def _fetch_one(url: str) -> Optional[Dict[str, Any]]:
            async with sem:
                try:
                    r = await fetch_url(url)
                    r["text"] = _truncate(r.get("text", ""), max_chars_per_doc)
                    return r
                except Exception as e:
                    return {"url": url, "error": str(e), "text": ""}

        tasks = []
        for r in results[:fetch_top_n]:
            u = r.get("url")
            if u:
                tasks.append(asyncio.create_task(_fetch_one(u)))

        if tasks:
            fetched = await asyncio.gather(*tasks)
            docs = [d for d in fetched if d]

        summary: Any = None
        active_backend = _active_llm_backend(self.ctx)
        summarize_with_llm = active_backend != "local" or bool(getattr(settings, "local_code_web_research_llm_summary_enabled", False))

        if summarize_with_llm and self.llm_available() and docs:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "query": query,
                                "documents": [{"url": d.get("url"), "text": d.get("text", "")[:8000]} for d in docs],
                            },
                            indent=2,
                        ),
                    }
                ]
                # Hosted/cloud backends can summarize research as JSON. Local
                # code mode avoids this by default to prevent repeated-token JSON
                # loops from contaminating coding context.
                summary = await self.ctx.llm.chat_json_async(system=WEB_SUMMARY_SYSTEM, messages=messages, temperature=0.2)
            except Exception as e:
                summary = {"error": f"LLM summarization failed: {e}", **_extractive_summary(query, results, docs)}
        else:
            summary = _extractive_summary(query, results, docs)

        out = {
            "query": query,
            "results": results,
            "documents": docs,
            "summary": summary,
        }

        self.latest_result = out
        self.remember(json.dumps(out), tags=["web_research"], success=True)

        # Store to type memory (shared among web agents)
        try:
            self.add_type_memory(json.dumps(out), tags=["web_research"], success=True)
        except Exception as me:
            # Don't fail the whole request if the DB is briefly locked.
            self.ctx.run_logger(f"web_research:{self.agent_id} | type-memory write failed: {me}")

        # Store a compact version to hive memory so other agent types can learn from it.
        compact = {
            "query": query,
            "top_sources": [r.get("url") for r in results[: min(3, len(results))]],
            "key_points": (summary or {}).get("key_points") if isinstance(summary, dict) else None,
        }
        try:
            self.add_hive_memory(json.dumps(compact), tags=["web_research"], success=True)
        except Exception as me:
            self.ctx.run_logger(f"web_research:{self.agent_id} | hive-memory write failed: {me}")
        return out
