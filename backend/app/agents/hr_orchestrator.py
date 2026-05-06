from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.domain_orchestrator_base import DomainOrchestratorBase, _uniq, _now_iso


def _split_hr_tasks(question: str, *, max_tasks: int = 5) -> List[str]:
    """Very small heuristic splitter for multi-part HR requests."""
    if not question:
        return []

    lines = [l.strip() for l in question.splitlines() if l.strip()]
    bullets = [l.lstrip("-*\t ") for l in lines if l.startswith("-") or l.startswith("*")]
    if len(bullets) >= 2:
        return bullets[:max_tasks]

    # Fallback: split on ';'
    parts = [p.strip() for p in question.split(";") if p.strip()]
    if len(parts) >= 2:
        return parts[:max_tasks]

    return [question]


class HROrchestrator(DomainOrchestratorBase):
    """Specialized HR orchestrator.

    Capabilities:
    - Can spawn multiple HR agents for multi-part HR requests
    - Writes HR Q&A into HR type-memory and hive memory
    - Records per-query sequence traces + sequence learning
    """

    agent_type = "hr_orchestrator"
    domain_label = "Human Resources"

    def select_agents(self, *, question: str, context: Dict[str, Any]) -> List[str]:
        # Default single HR agent.
        return ["qa_hr"]

    async def answer(
        self,
        *,
        question: str,
        project_root: Optional[Path] = None,
        use_web: bool = True,
        use_local_refs: bool = True,
        max_web_queries: int = 2,
        # Multi-turn
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        conversation_max_messages: int = 12,
    ) -> Dict[str, Any]:
        tasks = _split_hr_tasks(question)
        if len(tasks) <= 1 or self.ctx.registry is None:
            return await super().answer(
                question=question,
                project_root=project_root,
                use_web=use_web,
                use_local_refs=use_local_refs,
                max_web_queries=max_web_queries,
                conversation_id=conversation_id,
                conversation_history=conversation_history,
                conversation_max_messages=conversation_max_messages,
            )

        # Multi-task path
        self.set_state("orchestrating")
        self.log(f"multi-task HR request: {len(tasks)} tasks")

        self._start_trace(question=question, conversation_id=conversation_id)
        self._trace_event(
            "query_config",
            multi_task=True,
            sub_tasks=len(tasks),
            use_web=bool(use_web),
            use_local_refs=bool(use_local_refs),
            max_web_queries=int(max_web_queries or 0),
            conversation_id=conversation_id,
        )

        success: Optional[bool] = True
        error: Optional[str] = None
        out: Dict[str, Any] = {}

        spawned = []
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

            # Add sequence-learning context so the HR agents can see it if useful.
            seq_ctx = self._sequence_learning_context(question=question)
            try:
                ctx["sequence_learning"] = seq_ctx
            except Exception:
                pass

            # Spawn up to 4 HR agents
            for _ in range(min(len(tasks), 4)):
                try:
                    self._trace_event("agent_spawn", agent_type="qa_hr")
                    a = await self.ctx.registry.spawn("qa_hr")
                    spawned.append(a)
                except Exception:
                    break

            jobs = []
            agents_used: List[str] = []
            for i, subq in enumerate(tasks[: len(spawned)]):
                a = spawned[i]
                agents_used.append(a.agent_type)
                self._trace_event("agent_call", agent_type=a.agent_type, method="answer", sub_question=subq)
                jobs.append(asyncio.create_task(a.answer(question=subq, context=ctx)))

            parts = await asyncio.gather(*jobs) if jobs else []

            # Combine
            answer_blocks: List[str] = []
            sources: List[str] = []
            key_points: List[str] = []
            followups: List[str] = []
            confidences: List[str] = []

            for i, r in enumerate(parts):
                if not isinstance(r, dict):
                    continue
                subq = tasks[i] if i < len(tasks) else ""
                answer_blocks.append(f"### Task {i+1}\n{subq}\n\n{r.get('answer','')}")
                for u in (r.get("sources_used") or []):
                    if isinstance(u, str) and u not in sources:
                        sources.append(u)
                for kp in (r.get("key_points") or []):
                    if isinstance(kp, str) and kp not in key_points:
                        key_points.append(kp)
                for fu in (r.get("followups") or []):
                    if isinstance(fu, str) and fu not in followups:
                        followups.append(fu)
                confidences.append(str(r.get("confidence", "")))

            final_answer = "\n\n".join(answer_blocks)

            out = {
                "domain": self.domain_label,
                "question": question,
                "answer": final_answer,
                "key_points": key_points[:12],
                "sources_used": _uniq(sources)[:25],
                "followups": followups[:8],
                "confidence": "high" if "high" in confidences else ("medium" if "medium" in confidences else "low"),
                "agents_used": agents_used,
                "conversation_id": conversation_id,
                "context": {
                    "web_sources": (ctx.get("web_research") or {}).get("sources", []) if isinstance(ctx.get("web_research"), dict) else [],
                    "local_reference_matches": (ctx.get("local_references") or {}).get("matches", []) if isinstance(ctx.get("local_references"), dict) else [],
                    "conversation": ctx.get("conversation"),
                    "sequence_learning": seq_ctx,
                },
            }

            # Publish for logging
            try:
                await self.ctx.bus.publish(
                    "qa",
                    {
                        "domain": self.domain_label,
                        "question": question,
                        "answer": out["answer"],
                        "agents_used": agents_used,
                        "sources_used": out.get("sources_used"),
                        "conversation_id": conversation_id,
                    },
                )
            except Exception:
                pass

            # Persist orchestration memory (type + hive)
            try:
                mem_payload = {
                    "domain": self.domain_label,
                    "question": question,
                    "answer": out.get("answer"),
                    "key_points": out.get("key_points"),
                    "sources_used": out.get("sources_used"),
                    "agents_used": agents_used,
                    "conversation_id": conversation_id,
                }
                self.add_type_memory(json.dumps(mem_payload, ensure_ascii=False), tags=["qa_orchestration", "hr"], success=True)
                self.add_hive_memory(json.dumps(mem_payload, ensure_ascii=False), tags=["qa_orchestration", "hr"], success=True)
            except Exception:
                pass

            # Append to conversation thread
            if conversation_id:
                try:
                    self.ctx.memory_store.add_conversation_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=str(out.get("answer", "")),
                        orchestrator=self.agent_type,
                        input_mode=None,
                        meta={"kind": "domain_answer", "domain": self.domain_label, "result": out},
                        created_at=_now_iso(),
                    )
                except Exception:
                    pass

            if not str(out.get("answer") or "").strip():
                success = False

            self._trace_event(
                "query_end",
                success=success,
                confidence=out.get("confidence"),
                agents_used=agents_used,
                sources_used_count=len(out.get("sources_used") or []),
            )

        except Exception as e:
            success = False
            error = str(e)
            self.log(f"error while answering (HR): {e}")
            self._trace_event("error", error=error)
            out = {
                "domain": self.domain_label,
                "question": question,
                "answer": f"An internal error occurred in HR orchestrator: {error}",
                "key_points": [],
                "sources_used": [],
                "followups": [],
                "confidence": "low",
                "agents_used": [],
                "conversation_id": conversation_id,
                "context": {},
            }

        finally:
            for a in spawned:
                try:
                    await self.ctx.registry.terminate(a.agent_id)
                    self._trace_event("agent_terminate", agent_type=a.agent_type)
                except Exception:
                    pass

            try:
                await self._commit_trace(
                    success=success,
                    extra={
                        "error": error,
                        "conversation_id": conversation_id,
                        "agents_used": out.get("agents_used") or [],
                        "confidence": out.get("confidence"),
                    },
                )
            except Exception:
                pass

            self.latest_result = out
            self.set_state("idle")

        return out
