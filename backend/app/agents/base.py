from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from app.api.schemas import AgentStatus
from app.llm.common import LLMUnavailable
from app.llm.factory import get_llm_client
from app.memory.store import MemoryStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentContext:
    run_id: str
    bus: Any  # MessageBus
    memory_store: MemoryStore
    run_logger: Callable[[str], None]
    status_reporter: Callable[[AgentStatus], None]
    registry: Any = None  # AgentRegistry
    # Shared, process-wide LLM client (OpenAI or local HF), selected by settings.
    llm: Any = field(default_factory=get_llm_client)


class BaseAgent:
    agent_type: str = "base"

    def __init__(self, *, agent_id: str, ctx: AgentContext) -> None:
        self.agent_id = agent_id
        self.ctx = ctx
        self.state: str = "idle"
        self._session_memories: List[Dict[str, Any]] = []
        self.latest_result: Optional[Dict[str, Any]] = None

        self.set_state("idle")

    # ---------- status & logs ----------

    def set_state(self, state: str) -> None:
        self.state = state
        self.ctx.status_reporter(
            AgentStatus(agent_id=self.agent_id, agent_type=self.agent_type, state=state, last_update=_now_iso())
        )

    def log(self, message: str) -> None:
        self.ctx.run_logger(f"{self.agent_type}:{self.agent_id} | {message}")

    # ---------- memory ----------

    def remember(self, content: str, *, tags: Optional[List[str]] = None, success: Optional[bool] = None) -> None:
        self._session_memories.append(
            {
                "scope": "agent",
                "agent_type": self.agent_type,
                "agent_id": self.agent_id,
                "content": content,
                "tags": tags or [],
                "success": success,
                "created_at": _now_iso(),
            }
        )

    def add_type_memory(self, content: str, *, tags: Optional[List[str]] = None, success: Optional[bool] = None) -> int:
        return self.ctx.memory_store.add(
            scope="type",
            agent_type=self.agent_type,
            agent_id=None,
            content=content,
            tags=tags or [],
            success=success,
            created_at=_now_iso(),
        )

    def add_hive_memory(self, content: str, *, tags: Optional[List[str]] = None, success: Optional[bool] = None) -> int:
        return self.ctx.memory_store.add(
            scope="hive",
            agent_type=None,
            agent_id=None,
            content=content,
            tags=tags or [],
            success=success,
            created_at=_now_iso(),
        )

    def search_type_memory(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        return self.ctx.memory_store.search(scope="type", agent_type=self.agent_type, query=query, limit=limit)

    def search_hive_memory(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        return self.ctx.memory_store.search(scope="hive", query=query, limit=limit)


    def search_agent_memory(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        # "agent" scope is per spawned agent_id (useful for debugging / long-lived agents).
        return self.ctx.memory_store.search(
            scope="agent",
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            query=query,
            limit=limit,
        )

    def get_memory_bundle(
        self,
        query: str,
        *,
        include_agent: bool = False,
        include_type: bool = True,
        include_hive: bool = True,
        limit_per_scope: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch relevant memories for a query.

        "Learning" here means: the agent can retrieve and reuse patterns from:
        - its *type* memory (shared across agent instances of same type),
        - the shared *hive* memory,
        - (optionally) its own *agent* memory.
        """
        bundle: Dict[str, List[Dict[str, Any]]] = {}

        if include_agent:
            bundle["agent"] = self.search_agent_memory(query, limit=limit_per_scope)

        if include_type:
            bundle["type"] = self.search_type_memory(query, limit=limit_per_scope)

        if include_hive:
            bundle["hive"] = self.search_hive_memory(query, limit=limit_per_scope)

        return bundle

    def format_memory_bundle(self, bundle: Dict[str, List[Dict[str, Any]]]) -> str:
        if not bundle:
            return "(no memories)"
        parts: List[str] = []
        for scope in ["agent", "type", "hive"]:
            entries = bundle.get(scope) or []
            if not entries:
                continue
            parts.append(f"## {scope} memory ({len(entries)})")
            for e in entries:
                tags = e.get("tags") or []
                succ = e.get("success")
                parts.append(f"- created_at={e.get('created_at')} tags={tags} success={succ}")
                parts.append(str(e.get("content", ""))[:4000])
        return "\n".join(parts) if parts else "(no relevant memories)"

    async def flush_session_memory(self) -> None:
        # Best-effort: failures in the memory DB should not crash the agent
        # shutdown path (which can surface as 500s / hung UI).
        for m in self._session_memories:
            try:
                self.ctx.memory_store.add(
                    scope=m["scope"],
                    agent_type=m.get("agent_type"),
                    agent_id=m.get("agent_id"),
                    content=m["content"],
                    tags=m.get("tags") or [],
                    success=m.get("success"),
                    created_at=m["created_at"],
                )
            except Exception as e:
                try:
                    self.ctx.run_logger(f"{self.agent_type}:{self.agent_id} | session-memory flush failed: {e}")
                except Exception:
                    pass
        self._session_memories = []

    # ---------- comms ----------

    async def send(self, topic: str, payload: Any) -> None:
        await self.ctx.bus.publish(topic, payload)

    async def broadcast(self, payload: Any) -> None:
        await self.ctx.bus.broadcast(payload)

    # ---------- LLM convenience ----------

    def llm_available(self) -> bool:
        llm = getattr(self.ctx, "llm", None)
        if llm is None:
            return False

        # Preferred: unified interface.
        try:
            fn = getattr(llm, "is_available", None)
            if callable(fn):
                return bool(fn())
        except Exception:
            return False

        # Back-compat: OpenAIClient had a private _client() method.
        try:
            _ = llm._client()  # type: ignore[attr-defined]
            return True
        except Exception:
            return False

    # ---------- lifecycle ----------

    async def terminate(self) -> None:
        self.set_state("terminating")
        await self.flush_session_memory()
        self.set_state("terminated")
