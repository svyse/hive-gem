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
        if self.llm_available() and docs:
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
                # Run LLM calls in a thread to avoid blocking the FastAPI event loop.
                summary = await self.ctx.llm.chat_json_async(system=WEB_SUMMARY_SYSTEM, messages=messages, temperature=0.2)
            except Exception as e:
                summary = {"error": f"LLM summarization failed: {e}"}
        else:
            # fallback: simple excerpt
            summary = {
                "key_points": [],
                "data": [],
                "open_questions": ["LLM unavailable or no documents fetched; summary is empty."],
            }

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
