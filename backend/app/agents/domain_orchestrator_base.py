from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.agents.base import BaseAgent
from app.core.config import settings
from app.llm.factory import active_backend
from app.utils.sequence_learning import (
    QueryTraceRecorder,
    extract_sequence_traces_from_memories,
    merge_preferred_order,
    rank_similar_traces,
    recommend_most_common_sequence,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uniq(seq: List[str]) -> List[str]:
    out: List[str] = []
    for x in seq:
        x = (x or "").strip()
        if not x:
            continue
        if x not in out:
            out.append(x)
    return out


def _wants_workspace_modules(question: str) -> bool:
    q = (question or "").lower()
    # "module" is the explicit keyword, but we also include common tool names.
    keywords = [
        "module",
        "modules",
        "workspace",
        "calculator",
        "script_runner",
        "snake",
        "tool",
        "utility",
        "run module",
        "create module",
    ]
    return any(k in q for k in keywords)


def _slim_messages(messages: List[Dict[str, Any]], *, max_messages: int, max_chars: int) -> List[Dict[str, Any]]:
    """Reduce message payload size before handing to the LLM.

    We keep only safe fields and truncate content.
    """

    def _trunc(s: Any) -> str:
        t = str(s or "")
        if len(t) <= max_chars:
            return t
        return t[:max_chars] + "\n...<truncated>..."

    slim: List[Dict[str, Any]] = []
    for m in (messages or [])[-max_messages:]:
        slim.append(
            {
                "id": m.get("id"),
                "role": m.get("role"),
                "orchestrator": m.get("orchestrator"),
                "input_mode": m.get("input_mode"),
                "created_at": m.get("created_at"),
                "content": _trunc(m.get("content")),
            }
        )
    return slim


class DomainOrchestratorBase(BaseAgent):
    """Common utilities for domain orchestrators.

    Domain orchestrators are *orchestrators* focused on Q&A/analysis tasks.

    Core features:
    - Context gathering (all memories + optional local references + optional web research)
    - Persistent multi-turn conversation threads (conversation_id)
    - Spawning specialist Q&A agents under the orchestrator
    - Optional cross-orchestrator consultation
    - Optional strategist agent usage
    - Logging + writing back to type and hive memories

    New in this version:
    - Per-query *sequence traces* that record the ordered sequence of:
        * memory accesses
        * agent spawns/calls/terminations
      These traces are persisted to orchestrator type memory + hive memory.

    - *Sequence learning*: for a new query, the orchestrator looks up similar
      past traces and derives a recommended agent-call order. This recommendation
      is used to reorder agent selection.
    """

    agent_type = "domain_orchestrator"
    domain_label: str = "Domain"

    # ---------------------------------------------------------------------
    # Trace helpers
    # ---------------------------------------------------------------------

    def _start_trace(self, *, question: str, conversation_id: Optional[str]) -> None:
        try:
            self._trace_recorder = QueryTraceRecorder(
                orchestrator=self.agent_type,
                run_id=str(getattr(self.ctx, "run_id", "")),
                query=question,
                domain=self.domain_label,
                conversation_id=conversation_id,
            )
            self._trace_event("query_start", question=question, conversation_id=conversation_id)
        except Exception:
            self._trace_recorder = None

    def _trace_event(self, kind: str, **data: Any) -> None:
        rec = getattr(self, "_trace_recorder", None)
        if rec is None:
            return
        try:
            rec.event(kind, **(data or {}))
        except Exception:
            return

    async def _commit_trace(self, *, success: Optional[bool], extra: Optional[Dict[str, Any]] = None) -> None:
        rec = getattr(self, "_trace_recorder", None)
        if rec is None:
            return

        try:
            payload = rec.to_dict(finished_at=_now_iso(), success=success, extra=extra)
        except Exception:
            return

        # Persist to orchestrator type memory and hive memory for learning.
        type_mem_id: Optional[int] = None
        hive_mem_id: Optional[int] = None
        try:
            tags = [
                "sequence_trace",
                self.agent_type,
                (self.domain_label or "domain").lower().replace(" ", "_"),
            ]
            type_mem_id = self.add_type_memory(json.dumps(payload, ensure_ascii=False), tags=tags, success=success)
            hive_mem_id = self.add_hive_memory(json.dumps(payload, ensure_ascii=False), tags=tags, success=success)
        except Exception:
            pass

        # Link trace to conversation turn for the trace viewer UI.
        # We treat ctx.run_id as the Q&A turn_id in interactive mode.
        try:
            conv_id = payload.get("conversation_id")
            turn_id = payload.get("run_id")
            if conv_id and turn_id:
                self.ctx.memory_store.add_conversation_trace(
                    conversation_id=str(conv_id),
                    turn_id=str(turn_id),
                    orchestrator=self.agent_type,
                    hive_memory_id=hive_mem_id,
                    type_memory_id=type_mem_id,
                    created_at=_now_iso(),
                )
        except Exception:
            pass

        # Publish to the bus so LoggingAgent can capture it.
        try:
            await self.ctx.bus.publish("trace", payload)
        except Exception:
            pass

    def _sequence_learning_context(self, *, question: str) -> Dict[str, Any]:
        """Derive recommendations from similar past traces.

        Implementation notes:
        - We scan recent memories (type + hive) and parse `sequence_trace` payloads.
        - We rank traces by a lightweight Jaccard similarity over tokens.
        - We recommend the most common agent sequence among the top similar traces.

        This is intentionally simple and dependency-free.
        """

        if not getattr(settings, "sequence_learning_enabled", True):
            return {"enabled": False}

        # Fetch recent entries (type+hive). This avoids relying on SQL LIKE matching.
        scan_limit = int(getattr(settings, "sequence_learning_scan_limit", 250) or 250)
        max_traces = int(getattr(settings, "sequence_learning_max_traces", 8) or 8)
        min_sim = float(getattr(settings, "sequence_learning_min_similarity", 0.08) or 0.08)

        try:
            self._trace_event(
                "memory_access",
                op="recent_all",
                scope="type,hive",
                limit=scan_limit,
                purpose="sequence_learning_scan",
            )
            recent = self.ctx.memory_store.recent_all(limit=scan_limit, scopes=["type", "hive"])
            #parsed = extract_sequence_traces_from_memories(recent)
            parsed = []
        except Exception as e:
            return {"enabled": False, "error": str(e)}

        try:
            similar = rank_similar_traces(
                query=question,
                traces=parsed,
                max_traces=max_traces,
                min_similarity=min_sim,
                orchestrator=self.agent_type,
            )

            # Recommend a stable ordering of agent types.
            sequences = []
            for t in similar:
                seq = t.get("agent_sequence") or []
                # Filter down to answer-capable domain agents only.
                filtered = [
                    str(x)
                    for x in seq
                    if isinstance(x, str)
                    and x
                    and (not x.endswith("_orchestrator"))
                    and (x != self.agent_type)
                    and (str(x).startswith("qa_") or str(x) == "workspace_module")
                ]
                sequences.append(filtered)

            recommended = recommend_most_common_sequence(sequences)

            # Keep the top-N trace summaries only (avoid huge context)
            summaries: List[Dict[str, Any]] = []
            for t in similar:
                summaries.append(
                    {
                        "query": t.get("query"),
                        "signature": t.get("signature"),
                        "success": t.get("success"),
                        "finished_at": t.get("finished_at") or t.get("_memory_created_at"),
                    }
                )

            self._trace_event(
                "sequence_learning",
                enabled=True,
                similar_traces=len(similar),
                recommended_agents=recommended,
            )

            return {
                "enabled": True,
                "similar_traces": summaries,
                "recommended_agents": recommended,
            }

        except Exception as e:
            return {"enabled": False, "error": str(e)}

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    async def answer(
        self,
        *,
        question: str,
        project_root: Optional[Path] = None,
        use_web: bool = True,
        use_local_refs: bool = True,
        max_web_queries: int = 2,
        # Multi-turn Q&A
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        conversation_max_messages: int = 12,
    ) -> Dict[str, Any]:
        """Default answer implementation.

        This default does:
        - gather context
        - select agents (with optional sequence-learning reorder)
        - collect their answers
        - combine into one final response
        - (optionally) append to a persistent conversation thread
        - persist orchestration memories (type + hive)
        - persist a sequence trace (type + hive)
        """

        self.set_state("orchestrating")
        self.log(f"answering ({self.domain_label})")

        self._start_trace(question=question, conversation_id=conversation_id)
        self._trace_event(
            "query_config",
            use_web=bool(use_web),
            use_local_refs=bool(use_local_refs),
            max_web_queries=int(max_web_queries or 0),
            conversation_id=conversation_id,
        )

        success: Optional[bool] = True
        error: Optional[str] = None
        out: Dict[str, Any] = {}

        try:
            ctx = await self._build_context(
                question=question,
                project_root=project_root,
                use_web=use_web,
                use_local_refs=use_local_refs,
                max_web_queries=max_web_queries,
                conversation_id=conversation_id,
                conversation_history=conversation_history,
                conversation_max_messages=conversation_max_messages,
            )

            seq_ctx = self._sequence_learning_context(question=question)
            try:
                ctx["sequence_learning"] = seq_ctx
            except Exception:
                pass

            base_agent_types = _uniq(self.select_agents(question=question, context=ctx))

            # If the user is asking about workspace tools/modules, add the
            # WorkspaceModuleAgent so the hive can actually create/list/run them.
            if _wants_workspace_modules(question):
                if "workspace_module" not in base_agent_types:
                    base_agent_types.append("workspace_module")

            is_local_backend = str(active_backend() or "").strip().lower() == "local"

            recommended = []
            if (not is_local_backend) and isinstance(seq_ctx, dict):
                recommended = [
                    str(x)
                    for x in (seq_ctx.get("recommended_agents") or [])
                    if str(x).strip() and (str(x).startswith("qa_") or str(x) == "workspace_module")
                ]
                recommended = [x for x in _uniq(recommended) if x in base_agent_types or x == "workspace_module"]

            agent_types = list(base_agent_types)
            if recommended:
                agent_types = merge_preferred_order(recommended, agent_types)

            # Keep the answer coherent: limit spawned Q&A agents.
            max_agents = 1 if is_local_backend else max(1, min(4, len(agent_types)))
            agent_types = agent_types[:max_agents]

            self._trace_event(
                "agent_selection",
                selected=agent_types,
                base_selection=base_agent_types,
                recommended=recommended,
            )

            answers, agents_used, agent_failures = await self._collect_answers(
                question=question,
                agent_types=agent_types,
                context=ctx,
            )
            self._trace_event(
                "agents_completed",
                agents_used=agents_used,
                answers_count=len(answers),
                failure_count=len(agent_failures),
            )

            final = self._combine_answers(
                question=question,
                context=ctx,
                answers=answers,
                agents_used=agents_used,
                failures=agent_failures,
            )

            out = {
                "domain": self.domain_label,
                "question": question,
                "answer": final.get("answer", ""),
                "key_points": final.get("key_points", []),
                "sources_used": _uniq([str(u) for u in (final.get("sources_used") or []) if str(u).strip()]),
                "followups": final.get("followups", []),
                "confidence": final.get("confidence", "medium"),
                "agents_used": agents_used,
                "agent_failures": final.get("agent_failures") or agent_failures,
                "conversation_id": conversation_id,
                "context": {
                    "web_sources": (ctx.get("web_research") or {}).get("sources", []) if isinstance(ctx.get("web_research"), dict) else [],
                    "local_reference_matches": (ctx.get("local_references") or {}).get("matches", []) if isinstance(ctx.get("local_references"), dict) else [],
                    "conversation": ctx.get("conversation"),
                    "sequence_learning": seq_ctx,
                    "agent_failures": final.get("agent_failures") or agent_failures,
                },
            }

            # Orchestrator-level learning (high-level result summary)
            mem_payload = {
                "domain": self.domain_label,
                "question": question,
                "answer": out.get("answer"),
                "key_points": out.get("key_points"),
                "sources_used": out.get("sources_used"),
                "agents_used": agents_used,
                "agent_failures": out.get("agent_failures") or agent_failures,
                "conversation_id": conversation_id,
            }

            try:
                self.add_type_memory(
                    json.dumps(mem_payload, ensure_ascii=False),
                    tags=["qa_orchestration", self.domain_label.lower().replace(" ", "_")],
                    success=True,
                )
                self.add_hive_memory(
                    json.dumps(mem_payload, ensure_ascii=False),
                    tags=["qa_orchestration", self.domain_label.lower().replace(" ", "_")],
                    success=True,
                )
            except Exception:
                pass

            # Append domain answer to conversation thread
            if conversation_id:
                try:
                    self.ctx.memory_store.add_conversation_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=str(out.get("answer", "")),
                        orchestrator=self.agent_type,
                        input_mode=None,
                        meta={
                            # Used by the trace viewer UI to associate this message with
                            # the stored per-turn sequence trace(s).
                            "turn_id": str(getattr(self.ctx, "run_id", "")),
                            "kind": "domain_answer",
                            "domain": self.domain_label,
                            "result": {
                                "key_points": out.get("key_points"),
                                "sources_used": out.get("sources_used"),
                                "followups": out.get("followups"),
                                "confidence": out.get("confidence"),
                                "agents_used": out.get("agents_used"),
                                "agent_failures": out.get("agent_failures") or [],
                            },
                        },
                        created_at=_now_iso(),
                    )

                    self.ctx.memory_store.update_conversation(
                        conversation_id=conversation_id,
                        updated_at=_now_iso(),
                        metadata_patch={
                            "last_domain": self.domain_label,
                            "last_orchestrator": self.agent_type,
                        },
                    )
                except Exception:
                    pass

            # Bus event (picked up by LoggingAgent)
            try:
                await self.ctx.bus.publish(
                    "qa",
                    {
                        "domain": self.domain_label,
                        "question": question,
                        "answer": out.get("answer"),
                        "agents_used": agents_used,
                        "agent_failures": out.get("agent_failures") or agent_failures,
                        "sources_used": out.get("sources_used"),
                        "conversation_id": conversation_id,
                    },
                )
            except Exception:
                pass

            # Best-effort: mark success False when answer is empty, an internal-error string leaked through,
            # or no agent produced a usable payload.
            answer_text = str(out.get("answer") or "").strip()
            answer_low = answer_text.lower()
            if (not answer_text) or (not answers) or answer_low.startswith("internal error in") or answer_low.startswith("an internal error occurred"):
                success = False

            self._trace_event(
                "query_end",
                success=success,
                confidence=out.get("confidence"),
                agents_used=agents_used,
                failure_count=len(out.get("agent_failures") or []),
                sources_used_count=len(out.get("sources_used") or []),
            )

        except Exception as e:
            success = False
            error = str(e)
            self.log(f"error while answering ({self.domain_label}): {e}")
            self._trace_event("error", error=error)
            out = {
                "domain": self.domain_label,
                "question": question,
                "answer": f"An internal error occurred in {self.domain_label} orchestrator: {error}",
                "key_points": [],
                "sources_used": [],
                "followups": [],
                "confidence": "low",
                "agents_used": [],
                "agent_failures": [],
                "conversation_id": conversation_id,
                "context": {},
            }

        # Persist trace regardless of success
        try:
            await self._commit_trace(
                success=success,
                extra={
                    "error": error,
                    "agents_used": out.get("agents_used"),
                    "agent_failures": out.get("agent_failures") or [],
                    "confidence": out.get("confidence"),
                    "sources_used": out.get("sources_used"),
                    "conversation_id": conversation_id,
                },
            )
        except Exception:
            pass

        self.latest_result = out
        self.set_state("idle")
        return out

    def select_agents(self, *, question: str, context: Dict[str, Any]) -> List[str]:
        """Return a list of agent_type strings to use for this question."""
        return []

    async def _build_context(
        self,
        *,
        question: str,
        project_root: Optional[Path],
        use_web: bool,
        use_local_refs: bool,
        max_web_queries: int,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        conversation_max_messages: int = 12,
    ) -> Dict[str, Any]:
        self.set_state("gathering_context")

        # All orchestrators should have access to all memory scopes.
        try:
            memory_relevant = self.ctx.memory_store.search_all(query=question, limit=30, scopes=None)
            self._trace_event(
                "memory_access",
                op="search_all",
                scope="hive,type,agent",
                query=question,
                limit=30,
                result_count=len(memory_relevant),
            )
        except Exception as e:
            memory_relevant = []
            self._trace_event("memory_access", op="search_all", scope="hive,type,agent", error=str(e))

        try:
            memory_recent = self.ctx.memory_store.recent_all(limit=15, scopes=None)
            self._trace_event(
                "memory_access",
                op="recent_all",
                scope="hive,type,agent",
                limit=15,
                result_count=len(memory_recent),
            )
        except Exception as e:
            memory_recent = []
            self._trace_event("memory_access", op="recent_all", scope="hive,type,agent", error=str(e))

        # Conversation context (multi-turn threads)
        conversation_ctx: Optional[Dict[str, Any]] = None
        if conversation_id:
            try:
                if conversation_history is None:
                    all_msgs = self.ctx.memory_store.get_conversation_messages(conversation_id=conversation_id, limit=80)
                    self._trace_event(
                        "memory_access",
                        op="get_conversation_messages",
                        scope="conversation_messages",
                        conversation_id=conversation_id,
                        limit=80,
                        result_count=len(all_msgs),
                    )
                else:
                    all_msgs = conversation_history

                if conversation_history is None:
                    thread_msgs = self.ctx.memory_store.get_conversation_thread(
                        conversation_id=conversation_id,
                        orchestrator=self.agent_type,
                        limit=80,
                        include_master=True,
                    )
                    self._trace_event(
                        "memory_access",
                        op="get_conversation_thread",
                        scope="conversation_messages",
                        conversation_id=conversation_id,
                        limit=80,
                        result_count=len(thread_msgs),
                    )
                else:
                    # Best-effort filtering when caller provides history
                    thread_msgs = []
                    for m in all_msgs:
                        if m.get("role") == "user":
                            thread_msgs.append(m)
                        elif m.get("role") == "assistant":
                            orch = (m.get("orchestrator") or "").strip()
                            if orch in (self.agent_type, "hive_master_orchestrator"):
                                thread_msgs.append(m)

                conversation_ctx = {
                    "conversation_id": conversation_id,
                    "recent_messages": _slim_messages(all_msgs, max_messages=conversation_max_messages, max_chars=1400),
                    "thread_messages": _slim_messages(thread_msgs, max_messages=conversation_max_messages, max_chars=1400),
                    "total_messages": len(all_msgs),
                }
            except Exception as e:
                conversation_ctx = {
                    "conversation_id": conversation_id,
                    "error": str(e),
                    "recent_messages": [],
                    "thread_messages": [],
                    "total_messages": 0,
                }

        local_refs: Any = None
        if use_local_refs and self.ctx.registry is not None:
            try:
                ref_agent = await self.ctx.registry.spawn("local_reference")
                self._trace_event("agent_spawn", agent_type="local_reference")
                try:
                    self._trace_event("agent_call", agent_type="local_reference", method="find_references", query=question)
                    local_refs = await asyncio.wait_for(
                        ref_agent.find_references(query=question, project_root=project_root),
                        timeout=float(settings.resource_call_timeout_s or 25),
                    )
                finally:
                    await self.ctx.registry.terminate(ref_agent.agent_id)
                    self._trace_event("agent_terminate", agent_type="local_reference")
            except Exception as e:
                local_refs = {"error": str(e)}

        web_research: Any = None
        effective_backend = str(active_backend() or getattr(settings, "llm_backend", "local") or "local").strip().lower()
        local_web_allowed = bool(getattr(settings, "local_qa_web_research_enabled", False))
        if use_web and settings.web_research_enabled and self.ctx.registry is not None and (effective_backend != "local" or local_web_allowed):
            # Web research is best-effort: failures (timeouts, DB locks, etc.)
            # must not crash the orchestrator turn.
            queries = [question]
            queries = queries[: max(1, int(max_web_queries or 1))]

            pool: List[Any] = []
            items: List[Dict[str, Any]] = []
            err: Optional[str] = None
            try:
                for _ in range(min(len(queries), int(settings.web_research_pool_size))):
                    try:
                        a = await self.ctx.registry.spawn("web_research")
                        pool.append(a)
                        self._trace_event("agent_spawn", agent_type="web_research")
                    except Exception as e:
                        err = f"spawn failed: {e}"
                        break

                if pool:
                    tasks = []
                    for i, q in enumerate(queries):
                        agent = pool[i % len(pool)]
                        self._trace_event("agent_call", agent_type="web_research", method="research", query=q)
                        tasks.append(
                            asyncio.create_task(
                                asyncio.wait_for(
                                    agent.research(query=q),
                                    timeout=float(settings.resource_call_timeout_s or 25),
                                )
                            )
                        )

                    if tasks:
                        parts = await asyncio.gather(*tasks, return_exceptions=True)
                        for p in parts:
                            if isinstance(p, dict):
                                items.append(p)
                            elif isinstance(p, BaseException) and err is None:
                                err = str(p)

                sources: List[str] = []
                for it in items:
                    for r in (it.get("results") or []):
                        if isinstance(r, dict):
                            u = r.get("url")
                            if u and u not in sources:
                                sources.append(u)

                web_research = {
                    "enabled": err is None,
                    "error": err,
                    "queries": queries,
                    "items": items,
                    "sources": sources[:25],
                }
            except Exception as e:
                web_research = {"enabled": False, "error": str(e), "queries": queries, "items": [], "sources": []}
            finally:
                for a in pool:
                    try:
                        await self.ctx.registry.terminate(a.agent_id)
                        self._trace_event("agent_terminate", agent_type="web_research")
                    except Exception:
                        pass
        elif use_web and settings.web_research_enabled and effective_backend == "local" and not local_web_allowed:
            web_research = {
                "enabled": False,
                "skipped": "disabled_for_local_backend",
                "queries": [],
                "items": [],
                "sources": [],
            }

        rag_context: Any = None
        try:
            if bool(getattr(settings, "rag_enabled", True)):
                from app.rag import build_rag_context

                rag_context = build_rag_context(
                    query=question,
                    memory_store=self.ctx.memory_store,
                    project_root=project_root,
                    mode="qa",
                )
                stats = (rag_context or {}).get("stats") if isinstance(rag_context, dict) else {}
                self._trace_event(
                    "memory_access",
                    op="rag",
                    scope="hive,type,agent,project",
                    query=question,
                    result_count=int((stats or {}).get("memory_matches", 0) or 0) + int((stats or {}).get("project_file_matches", 0) or 0),
                )
        except Exception as e:
            rag_context = {"enabled": False, "error": str(e)}

        ctx = {
            "memory_relevant": memory_relevant,
            "memory_recent": memory_recent,
            "local_references": local_refs,
            "web_research": web_research,
            "conversation": conversation_ctx,
            "rag": rag_context,
        }

        # Expose workspace modules to downstream agents so they can propose
        # using them (or ask the WorkspaceModuleAgent to run them).
        try:
            from app.workspace_modules.manager import WorkspaceModuleManager

            mgr = WorkspaceModuleManager()
            ctx["workspace_modules"] = [m.to_dict() for m in mgr.list_modules()][:50]
        except Exception:
            ctx["workspace_modules"] = []

        self.set_state("idle")
        return ctx

    def _iter_llm_clients_for_timeouts(self) -> List[Any]:
        llm = getattr(self.ctx, "llm", None)
        if llm is None:
            return []

        raw_clients: List[Any] = [llm]
        try:
            raw_clients.extend(list(getattr(llm, "_clients", []) or []))
        except Exception:
            pass

        seen: set[int] = set()
        out: List[Any] = []
        for client in raw_clients:
            if client is None:
                continue
            ident = id(client)
            if ident in seen:
                continue
            seen.add(ident)
            out.append(client)
        return out

    def _agent_answer_timeout_s(self) -> float:
        base = float(settings.qa_call_timeout_s or 120.0)
        best = base

        for client in self._iter_llm_clients_for_timeouts():
            if str(getattr(client, "backend", "")).strip().lower() != "local":
                continue

            device = str(getattr(client, "_device", "")).strip().lower()
            regular = float(
                getattr(settings, "local_llm_timeout_s", None)
                or getattr(settings, "llm_timeout_s", 60.0)
                or 60.0
            )
            initial = float(
                getattr(settings, "local_llm_initial_timeout_s", None)
                or max(regular, 180.0)
            )
            loaded = getattr(client, "_model", None) is not None and getattr(client, "_tokenizer", None) is not None
            buffer_s = 60.0 if device.startswith("cpu") else 15.0
            best = max(best, (regular if loaded else initial) + buffer_s)

        return best

    def _orchestrator_answer_timeout_s(self) -> float:
        base = float(settings.orchestrator_call_timeout_s or 180.0)
        return max(base, self._agent_answer_timeout_s() + 60.0)

    async def _collect_answers(
        self,
        *,
        question: str,
        agent_types: List[str],
        context: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        if self.ctx.registry is None:
            return [], [], []

        spawned = []
        failures: List[Dict[str, Any]] = []
        try:
            for t in agent_types:
                try:
                    self._trace_event("agent_spawn", agent_type=t)
                    a = await self.ctx.registry.spawn(t)
                    spawned.append(a)
                except Exception as e:
                    msg = str(e)
                    self.log(f"failed to spawn agent type={t}: {msg}")
                    self._trace_event("agent_spawn_failed", agent_type=t, error=msg)
                    failures.append({"agent_type": t, "reason": "spawn_failed", "message": msg})

            tasks = []
            task_meta: List[Dict[str, Any]] = []
            agents_used: List[str] = []
            for a in spawned:
                if hasattr(a, "answer"):
                    agents_used.append(a.agent_type)
                    timeout_s = float(self._agent_answer_timeout_s())
                    self._trace_event("agent_call", agent_type=a.agent_type, method="answer", timeout_s=timeout_s)
                    tasks.append(
                        asyncio.create_task(
                            asyncio.wait_for(
                                a.answer(question=question, context=context),
                                timeout=timeout_s,
                            )
                        )
                    )
                    task_meta.append({"agent_type": a.agent_type, "timeout_s": timeout_s})
                else:
                    failures.append(
                        {
                            "agent_type": getattr(a, "agent_type", "unknown"),
                            "reason": "missing_answer_method",
                            "message": "Spawned agent does not implement answer().",
                        }
                    )

            answers: List[Dict[str, Any]] = []
            if tasks:
                res = await asyncio.gather(*tasks, return_exceptions=True)
                for meta, r in zip(task_meta, res):
                    agent_name = str(meta.get("agent_type") or "unknown")
                    timeout_s = float(meta.get("timeout_s") or settings.qa_call_timeout_s or 120.0)
                    if isinstance(r, dict):
                        answer_text = str(r.get("answer") or "").strip()
                        low = answer_text.lower()
                        if (not answer_text) or low.startswith("internal error in") or low.startswith("an internal error occurred"):
                            msg = answer_text or "Agent returned an empty answer."
                            failures.append({"agent_type": agent_name, "reason": "agent_error", "message": msg})
                            self._trace_event("agent_failure", agent_type=agent_name, error=msg)
                        else:
                            answers.append(r)
                    elif isinstance(r, asyncio.TimeoutError):
                        msg = f"Timed out after {timeout_s:.1f}s"
                        self.log(f"agent answer timed out ({agent_name}) after {timeout_s:.1f}s")
                        self._trace_event("agent_timeout", agent_type=agent_name, timeout_s=timeout_s)
                        failures.append({"agent_type": agent_name, "reason": "timeout", "message": msg})
                    elif isinstance(r, BaseException):
                        msg = str(r)
                        self.log(f"agent answer failed ({agent_name}): {msg}")
                        self._trace_event("agent_failure", agent_type=agent_name, error=msg)
                        failures.append({"agent_type": agent_name, "reason": "error", "message": msg})
                    else:
                        msg = "Agent returned no result."
                        self.log(f"agent answer returned no result ({agent_name})")
                        self._trace_event("agent_empty_result", agent_type=agent_name)
                        failures.append({"agent_type": agent_name, "reason": "empty", "message": msg})

            return answers, agents_used, failures
        finally:
            for a in spawned:
                try:
                    await self.ctx.registry.terminate(a.agent_id)
                    self._trace_event("agent_terminate", agent_type=a.agent_type)
                except Exception:
                    pass

    def _combine_answers(
        self,
        *,
        question: str,
        context: Dict[str, Any],
        answers: List[Dict[str, Any]],
        agents_used: List[str],
        failures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Combine multiple agent answers into one.

        Default policy:
        - Start from the first answer
        - Merge sources/key_points/followups across answers
        """

        web_sources: List[str] = []
        if isinstance(context.get("web_research"), dict):
            ws = (context.get("web_research") or {}).get("sources") or []
            if isinstance(ws, list):
                web_sources = [str(u).strip() for u in ws if str(u).strip()]

        if not answers:
            selected_agents = _uniq([str(a) for a in (agents_used or []) if str(a).strip()])
            timeout_failures = [f for f in (failures or []) if str(f.get("reason") or "") == "timeout"]
            if selected_agents:
                key_points = [f"Selected agents: {', '.join(selected_agents)}"]
                followups: List[str] = []
                if timeout_failures:
                    answer = "A domain agent was selected, but it timed out before it could return an answer."
                    key_points.append("The configured agent timeout was reached before a response was produced.")
                    followups = [
                        "Retry now that the local model may be warm.",
                        "Increase QA_CALL_TIMEOUT_S or LOCAL_LLM_INITIAL_TIMEOUT_S if local CPU inference is expected.",
                    ]
                else:
                    answer = "A domain agent was selected, but it failed before it could return an answer."
                    if failures:
                        first = failures[0]
                        detail = str(first.get("message") or first.get("reason") or "unknown failure")
                        key_points.append(f"First failure: {detail}")
                    followups = ["Check the provider/backend logs for the selected agent and retry."]

                return {
                    "answer": answer,
                    "key_points": key_points,
                    "sources_used": web_sources[:25],
                    "followups": followups,
                    "confidence": "low",
                    "agent_failures": failures[:8],
                }

            return {
                "answer": "No domain agents matched the question or could be started for it.",
                "key_points": [],
                "sources_used": web_sources[:25],
                "followups": [],
                "confidence": "low",
                "agent_failures": failures[:8],
            }

        base = dict(answers[0])

        sources = _uniq(
            [
                *[u for a in answers for u in (a.get("sources_used") or []) if isinstance(u, str)],
                *web_sources,
            ]
        )
        base["sources_used"] = sources[:25]
        base["agent_failures"] = failures[:8]

        # Merge key points
        kps: List[str] = []
        for a in answers:
            for kp in (a.get("key_points") or []):
                if isinstance(kp, str) and kp not in kps:
                    kps.append(kp)
        base["key_points"] = kps[:12]

        # Merge followups
        fups: List[str] = []
        for a in answers:
            for fu in (a.get("followups") or []):
                if isinstance(fu, str) and fu not in fups:
                    fups.append(fu)
        base["followups"] = fups[:8]

        # If multiple answers exist, indicate synthesis.
        if len(answers) > 1:
            base["answer"] = f"(Synthesized from {len(answers)} agents)\n\n" + str(base.get("answer", ""))

        return base

    async def consult_orchestrator(
        self,
        *,
        orchestrator_type: str,
        question: str,
        project_root: Optional[Path] = None,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        use_web: bool = True,
        use_local_refs: bool = True,
        max_web_queries: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """Spawn another orchestrator and ask it a question."""
        if self.ctx.registry is None:
            return None

        orch = None
        try:
            self._trace_event("orchestrator_consult", orchestrator_type=orchestrator_type, question=question)
            orch = await self.ctx.registry.spawn(orchestrator_type)
            if hasattr(orch, "answer"):
                self._trace_event("agent_call", agent_type=orchestrator_type, method="answer")
                # New: thread through capability flags so domain orchestrators
                # behave consistently with the master settings.
                try:
                    return await asyncio.wait_for(
                        orch.answer(
                        question=question,
                        project_root=project_root,
                        conversation_id=conversation_id,
                        conversation_history=conversation_history,
                        use_web=use_web,
                        use_local_refs=use_local_refs,
                        max_web_queries=max_web_queries,
                        ),
                        timeout=self._orchestrator_answer_timeout_s(),
                    )
                except TypeError:
                    # Backwards-compat for orchestrators that don't accept the
                    # extended signature.
                    return await asyncio.wait_for(
                        orch.answer(
                        question=question,
                        project_root=project_root,
                        conversation_id=conversation_id,
                        conversation_history=conversation_history,
                        ),
                        timeout=self._orchestrator_answer_timeout_s(),
                    )
            return None
        finally:
            if orch is not None:
                try:
                    await self.ctx.registry.terminate(orch.agent_id)
                except Exception:
                    pass

    async def get_strategy(self, *, goal: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Spawn a strategist agent and get a plan."""
        if self.ctx.registry is None:
            return None

        strat = None
        try:
            self._trace_event("agent_spawn", agent_type="strategist")
            strat = await self.ctx.registry.spawn("strategist")
            if hasattr(strat, "strategize"):
                self._trace_event("agent_call", agent_type="strategist", method="strategize", goal=goal)
                return await asyncio.wait_for(
                    strat.strategize(goal=goal, context=context),
                    timeout=float(settings.qa_call_timeout_s or 120),
                )
            return None
        finally:
            if strat is not None:
                try:
                    await self.ctx.registry.terminate(strat.agent_id)
                    self._trace_event("agent_terminate", agent_type="strategist")
                except Exception:
                    pass
