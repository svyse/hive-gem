from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.llm.factory import active_backend


def _safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + " ...<truncated>"


def _compact_jsonish(value: Any, *, max_depth: int = 3, max_items: int = 4, max_str: int = 350) -> Any:
    if max_depth <= 0:
        return _truncate_text(value, max_str)
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                out["_truncated"] = f"+{len(value) - max_items} more keys"
                break
            out[str(k)] = _compact_jsonish(v, max_depth=max_depth - 1, max_items=max_items, max_str=max_str)
        return out
    if isinstance(value, list):
        out = [_compact_jsonish(v, max_depth=max_depth - 1, max_items=max_items, max_str=max_str) for v in value[:max_items]]
        if len(value) > max_items:
            out.append(f"... +{len(value) - max_items} more items")
        return out
    if isinstance(value, tuple):
        return _compact_jsonish(list(value), max_depth=max_depth, max_items=max_items, max_str=max_str)
    if isinstance(value, str):
        return _truncate_text(value, max_str)
    return value


LOCAL_QA_TEXT_TEMPLATE = """You are a {domain} interactive Q&A agent inside a multi-agent hive.

Return plain text in the following exact structure:
ANSWER:
<short answer paragraph or two>

KEY POINTS:
- point one
- point two

FOLLOWUPS:
- optional follow-up question
- optional follow-up question

CONFIDENCE: low|medium|high

Do not use markdown code fences. Keep the headings exactly as written above.
For simple code-generation questions, put only the code in ANSWER, leave KEY POINTS and FOLLOWUPS empty, and do not add explanations.
"""


_HEADING_RE = re.compile(
    r"^\s*(answer|key\s*points?|key\s*point|follow\s*ups?|followups?|confidence|confidenlty)\s*(?::|-)?\s*(.*)$",
    re.IGNORECASE,
)


def _heading_kind(line: str) -> tuple[Optional[str], str]:
    m = _HEADING_RE.match(line or "")
    if not m:
        return None, ""
    label = re.sub(r"[^a-z]", "", m.group(1).lower())
    tail = (m.group(2) or "").strip()
    if label.startswith("answer"):
        return "answer", tail
    if label.startswith("keypoint"):
        return "key_points", tail
    if label.startswith("followup"):
        return "followups", tail
    if label.startswith("confidence") or label.startswith("confidenlty"):
        return "confidence", tail
    return None, ""


def _strip_list_marker(text: str) -> str:
    return re.sub(r"^(?:[-*+]\s*|\d+[.)]\s*)", "", str(text or "").strip()).strip()


def _looks_like_simple_code_question(question: str | None) -> bool:
    q = re.sub(r"\s+", " ", str(question or "").strip().lower())
    if not q:
        return False
    has_code_intent = any(term in q for term in ("code", "script", "program", "make", "write", "create", "generate"))
    has_language = any(term in q for term in ("python", ".py", "javascript", "js", "typescript", "java", "c++", "c#", "go ", "rust"))
    has_hello = any(term in q for term in ("hello world", "hello, world", "hello-world", "helloworld"))
    return bool(has_code_intent and (has_language or has_hello))


def _unwrap_single_code_fence(text: str) -> str:
    s = str(text or "").strip()
    m = re.fullmatch(r"```[a-zA-Z0-9_+.-]*\s*\n?([\s\S]*?)\n?```", s)
    if m:
        return m.group(1).strip()
    return s


def _looks_like_code_answer(answer: str) -> bool:
    s = str(answer or "").strip()
    if not s:
        return False
    code_markers = ("print(", "def ", "class ", "import ", "from ", "if __name__", "console.log", "function ")
    return any(marker in s for marker in code_markers)


def _parse_structured_text_answer(text: str, *, question: str | None = None) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {"answer": "", "key_points": [], "sources_used": [], "followups": [], "confidence": "low"}

    sections = {"answer": [], "key_points": [], "followups": []}
    confidence = "medium"
    current = "answer"
    saw_heading = False

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        heading, tail = _heading_kind(stripped)
        if heading:
            saw_heading = True
            if heading == "confidence":
                val = tail.lower()
                for candidate in ("low", "medium", "high"):
                    if candidate in val.split() or val == candidate:
                        confidence = candidate
                        break
                continue
            current = heading
            if tail:
                if current in {"key_points", "followups"}:
                    cleaned = _strip_list_marker(tail)
                    if cleaned:
                        sections[current].append(cleaned)
                else:
                    sections["answer"].append(tail)
            continue

        if current in {"key_points", "followups"}:
            cleaned = _strip_list_marker(stripped)
            if cleaned:
                sections[current].append(cleaned)
        else:
            sections["answer"].append(stripped)

    answer = "\n".join(sections["answer"]).strip() if saw_heading else raw
    answer = answer or raw

    if _looks_like_simple_code_question(question):
        answer = _unwrap_single_code_fence(answer)
        if _looks_like_code_answer(answer):
            sections["key_points"] = []
            sections["followups"] = []

    return {
        "answer": answer,
        "key_points": sections["key_points"][:8],
        "sources_used": [],
        "followups": sections["followups"][:6],
        "confidence": confidence,
    }

QA_SYSTEM_TEMPLATE = """You are a {domain} interactive Q&A agent inside a multi-agent hive.

Goals:
- Answer the user's question accurately and clearly.
- Use provided context (memories, local references, web research) when relevant.
- When the question is ambiguous, ask 1-3 brief clarifying questions in 'followups'.

Constraints:
- Do NOT fabricate citations or URLs.
- If you use any web research items, include their URLs in 'sources_used'.
- If the topic is medical, legal, or financial, include a short safety disclaimer.
- Return VALID JSON only (no markdown fences).

Output schema:
{{
  "answer": "...",
  "key_points": ["..."],
  "sources_used": ["https://..."],
  "followups": ["..."],
  "confidence": "low|medium|high"
}}
"""


class DomainQAAAgent(BaseAgent):
    """Base class for domain Q&A agents.

    Each subclass should set a unique `agent_type` and a `domain_name`.
    `agent_type` becomes the agent's type-memory category, enabling separate
    memories per agent type.
    """

    agent_type = "qa_base"
    domain_name: str = "General"
    safety_disclaimer: str = ""

    async def answer(
        self,
        *,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        max_peer_consults: int = 0,
    ) -> Dict[str, Any]:
        self.set_state("answering")
        self.log(f"answering ({self.domain_name})")

        try:
            return await self._answer_impl(
                question=question,
                context=context,
                max_peer_consults=max_peer_consults,
            )
        except Exception as e:
            out = {
                "answer": f"Internal error in {self.domain_name} agent: {e}",
                "key_points": [],
                "sources_used": [],
                "followups": [],
                "confidence": "low",
            }
            self.latest_result = out
            try:
                self.remember(json.dumps({"q": question, "a": out}), tags=["qa", "error"], success=False)
            except Exception:
                pass
            try:
                self.add_type_memory(json.dumps({"q": question, "a": out}), tags=["qa", "error"], success=False)
            except Exception:
                pass
            return out
        finally:
            try:
                self.set_state("idle")
            except Exception:
                pass

    async def _answer_impl(
        self,
        *,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        max_peer_consults: int = 0,
    ) -> Dict[str, Any]:
        backend = str(active_backend() or getattr(self.ctx.llm, "backend", "") or "").strip().lower()
        is_local = backend == "local"

        mem_bundle = self.get_memory_bundle(
            question,
            include_type=True,
            include_hive=True,
            include_agent=False,
            limit_per_scope=2 if is_local else 6,
        )
        memory_context = self.format_memory_bundle(mem_bundle)
        if is_local:
            memory_context = _truncate_text(memory_context, 500)

        peer_notes: List[Dict[str, Any]] = []
        if max_peer_consults > 0 and self.ctx.registry is not None:
            try:
                peer_notes = await self._consult_peers(question=question, max_consults=max_peer_consults)
            except Exception:
                peer_notes = []

        if not self.llm_available():
            out = {
                "answer": "LLM unavailable. I cannot produce a high-quality domain answer.",
                "key_points": [],
                "sources_used": [],
                "followups": ["Check the configured LLM backend and retry."],
                "confidence": "low",
            }
            self.latest_result = out
            self.remember(json.dumps({"q": question, "a": out}), tags=["qa", self.domain_name.lower()], success=None)
            try:
                self.add_type_memory(json.dumps({"q": question, "a": out}), tags=["qa"], success=None)
            except Exception as me:
                self.ctx.run_logger(f"{self.agent_type}:{self.agent_id} | type-memory write failed: {me}")
            return out

        system = QA_SYSTEM_TEMPLATE.format(domain=self.domain_name)
        payload_context = context or {}
        payload_peers = peer_notes
        if is_local:
            payload_context = _compact_jsonish(payload_context, max_depth=3, max_items=4, max_str=300)
            payload_peers = _compact_jsonish(peer_notes, max_depth=2, max_items=3, max_str=240)

        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "context": payload_context,
                        "memory_context": memory_context,
                        "peer_notes": payload_peers,
                        "domain_safety_disclaimer": self.safety_disclaimer,
                    },
                    indent=2,
                ),
            }
        ]

        if is_local:
            text_out = await self.ctx.llm.chat_text_async(
                system=LOCAL_QA_TEXT_TEMPLATE.format(domain=self.domain_name),
                messages=messages,
                temperature=0.0,
                purpose="qa",
            )
            out = _parse_structured_text_answer(text_out, question=question)
        else:
            out = await self.ctx.llm.chat_json_async(system=system, messages=messages, temperature=0.2)
            if not isinstance(out, dict):
                raise ValueError("Q&A agent output must be JSON object")

        out.setdefault("answer", "")
        out.setdefault("key_points", [])
        out.setdefault("sources_used", [])
        out.setdefault("followups", [])
        out.setdefault("confidence", "medium")

        sources: List[str] = []
        for u in _safe_list(out.get("sources_used"))[:25]:
            s = str(u).strip()
            if s and s not in sources:
                sources.append(s)
        out["sources_used"] = sources

        self.latest_result = out

        payload = {
            "domain": self.domain_name,
            "question": question,
            "answer": out.get("answer"),
            "key_points": out.get("key_points"),
            "sources_used": out.get("sources_used"),
            "confidence": out.get("confidence"),
        }
        self.remember(json.dumps(payload), tags=["qa", self.domain_name.lower()], success=True)
        try:
            self.add_type_memory(json.dumps(payload), tags=["qa"], success=True)
        except Exception as me:
            self.ctx.run_logger(f"{self.agent_type}:{self.agent_id} | type-memory write failed: {me}")
        try:
            self.add_hive_memory(json.dumps(payload), tags=["qa", self.domain_name.lower()], success=True)
        except Exception as me:
            self.ctx.run_logger(f"{self.agent_type}:{self.agent_id} | hive-memory write failed: {me}")

        return out

    async def _consult_peers(self, *, question: str, max_consults: int) -> List[Dict[str, Any]]:
        if self.ctx.registry is None:
            return []

        peer_types = ["web_research"][:max_consults]

        notes: List[Dict[str, Any]] = []
        spawned = []
        try:
            for t in peer_types:
                try:
                    a = await self.ctx.registry.spawn(t)
                    spawned.append(a)
                except Exception:
                    continue

            tasks = []
            for a in spawned:
                if hasattr(a, "research"):
                    tasks.append(asyncio.create_task(a.research(query=question)))

            if tasks:
                results = await asyncio.gather(*tasks)
                for r in results:
                    if isinstance(r, dict):
                        notes.append(
                            {
                                "peer": "web_research",
                                "top_sources": [x.get("url") for x in (r.get("results") or [])[:3] if isinstance(x, dict)],
                                "summary": r.get("summary"),
                            }
                        )
        finally:
            for a in spawned:
                try:
                    await self.ctx.registry.terminate(a.agent_id)
                except Exception:
                    pass

        return notes
