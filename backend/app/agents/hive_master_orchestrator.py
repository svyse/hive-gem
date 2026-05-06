from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.domain_orchestrator_base import DomainOrchestratorBase, _uniq
from app.core.config import settings
from app.utils.sequence_learning import (
    extract_sequence_traces_from_memories,
    merge_preferred_order,
    rank_similar_traces,
    recommend_most_common_sequence,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = _normalize(text)
    for raw_kw in keywords:
        kw = _normalize(raw_kw)
        if not kw:
            continue
        if " " in kw or "-" in kw or "_" in kw:
            if kw in t:
                return True
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(kw)}(?![a-z0-9_])", t):
            return True
    return False


ORCH_ALIAS_TO_TYPE = {
    "interactive": "interactive_orchestrator",
    "stem": "interactive_orchestrator",
    "hr": "hr_orchestrator",
    "law": "law_orchestrator",
    "law_india": "law_india_orchestrator",
    "law_international": "law_international_orchestrator",
    "finance": "finance_orchestrator",
    "economics": "economics_orchestrator",
    "social": "social_orchestrator",
    "medicine": "medicine_orchestrator",
    "cyber": "cyber_orchestrator",
    "web_design": "web_design_orchestrator",
    "web": "web_design_orchestrator",
    "computer_vision": "computer_vision_orchestrator",
    "cv": "computer_vision_orchestrator",
}


def _looks_like_no_answer_payload(payload: Dict[str, Any]) -> bool:
    answer = str((payload or {}).get("answer") or "").strip().lower()
    if not answer:
        return True

    phrases = [
        "no domain agents were available to answer",
        "no domain agents matched the question or could be started",
        "selected domain agents were called, but none returned a usable answer",
        "a domain agent was selected, but it timed out before it could return an answer",
        "a domain agent was selected, but it failed before it could return an answer",
        "no orchestrators were available to answer",
    ]
    return any(p in answer for p in phrases)


class HiveMasterOrchestrator(DomainOrchestratorBase):
    """Top-level orchestrator for interactive Q&A.

    Responsibilities:
    - Route a question to one or more specialist orchestrators
    - Allow orchestrators to collaborate (multi-domain questions)
    - Ensure the *main orchestrator* has visibility into all results and memories
    - Support persistent multi-turn conversations via conversation_id

    Additional capabilities:
    - Sequence trace logging (agent calls + memory access order)
    - Sequence learning: prefer orchestrator routes that worked for similar past queries
    """

    agent_type = "hive_master_orchestrator"
    domain_label = "Hive Master"

    def _orchestrator_learning_context(self, *, question: str) -> Dict[str, Any]:
        """Recommend orchestrator routing based on similar past master traces."""

        if not getattr(settings, "sequence_learning_enabled", True):
            return {"enabled": False}

        scan_limit = int(getattr(settings, "sequence_learning_scan_limit", 250) or 250)
        max_traces = int(getattr(settings, "sequence_learning_max_traces", 8) or 8)
        min_sim = float(getattr(settings, "sequence_learning_min_similarity", 0.08) or 0.08)

        try:
            self._trace_event(
                "memory_access",
                op="recent_all",
                scope="type,hive",
                limit=scan_limit,
                purpose="master_orchestrator_learning_scan",
            )
            recent = self.ctx.memory_store.recent_all(limit=scan_limit, scopes=["type", "hive"])
            traces = extract_sequence_traces_from_memories(recent)
        except Exception as e:
            return {"enabled": False, "error": str(e)}

        try:
            similar = rank_similar_traces(
                query=question,
                traces=traces,
                max_traces=max_traces,
                min_similarity=min_sim,
                orchestrator=self.agent_type,
            )

            sequences: List[List[str]] = []
            for t in similar:
                # Prefer explicit extra.orchestrators_used if present.
                orch_seq: List[str] = []
                extra = t.get("extra")
                if isinstance(extra, dict) and isinstance(extra.get("orchestrators_used"), list):
                    orch_seq = [str(x) for x in extra.get("orchestrators_used") if str(x).strip()]

                # Fallback: derive from agent_sequence (which may contain consulted orchestrators).
                if not orch_seq:
                    seq = t.get("agent_sequence") or []
                    orch_seq = [
                        str(x)
                        for x in seq
                        if isinstance(x, str)
                        and x
                        and x.endswith("_orchestrator")
                        and x != self.agent_type
                    ]

                sequences.append(orch_seq)

            recommended = recommend_most_common_sequence(sequences)

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
                recommended_orchestrators=recommended,
            )

            return {
                "enabled": True,
                "similar_traces": summaries,
                "recommended_orchestrators": recommended,
            }

        except Exception as e:
            return {"enabled": False, "error": str(e)}

    async def answer(
        self,
        *,
        question: str,
        project_root: Optional[Path] = None,
        target_orchestrator: Optional[str] = None,
        use_web: bool = True,
        use_local_refs: bool = True,
        max_web_queries: int = 2,
        # Multi-turn
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        conversation_max_messages: int = 12,
    ) -> Dict[str, Any]:
        self.set_state("orchestrating")
        self.log("routing Q&A request")

        self._start_trace(question=question, conversation_id=conversation_id)
        self._trace_event(
            "query_config",
            use_web=bool(use_web),
            use_local_refs=bool(use_local_refs),
            max_web_queries=int(max_web_queries or 0),
            conversation_id=conversation_id,
            target_orchestrator=target_orchestrator,
        )

        success: Optional[bool] = True
        error: Optional[str] = None
        final: Dict[str, Any] = {}

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

            # Heuristic routing
            base_orchestrators = self._pick_orchestrators(question=question, target_orchestrator=target_orchestrator)

            # Sequence-learning routing adjustment (optional)
            learning = self._orchestrator_learning_context(question=question)
            is_local_backend = str(getattr(settings, "llm_backend", "") or "").strip().lower() == "local"
            recommended = []
            if (not is_local_backend) and (not target_orchestrator) and isinstance(learning, dict):
                recommended = [str(x) for x in (learning.get("recommended_orchestrators") or []) if str(x).strip()]

            orchestrators = list(base_orchestrators)
            if recommended:
                orchestrators = merge_preferred_order(recommended, orchestrators)

            # Keep local Q&A narrow and predictable.
            orchestrators = orchestrators[: (1 if is_local_backend else 3)]

            self._trace_event(
                "orchestrator_selection",
                selected=orchestrators,
                base_selection=base_orchestrators,
                recommended=recommended,
            )

            # Scale out: multi-orchestrator for multi-domain queries
            results: List[Dict[str, Any]] = []
            if self.ctx.registry is not None:
                tasks = []
                for ot in orchestrators:
                    tasks.append(
                        asyncio.create_task(
                            self.consult_orchestrator(
                                orchestrator_type=ot,
                                question=question,
                                project_root=project_root,
                                conversation_id=conversation_id,
                                conversation_history=conversation_history,
                                use_web=use_web,
                                use_local_refs=use_local_refs,
                                max_web_queries=max_web_queries,
                            )
                        )
                    )

                parts = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
                for p in parts:
                    if isinstance(p, dict):
                        results.append(p)
                    elif isinstance(p, BaseException):
                        self.log(f"consult orchestrator failed: {p}")

            # Combine
            if not results:
                final = {
                    "domain": self.domain_label,
                    "question": question,
                    "answer": "No orchestrators were available to answer.",
                    "key_points": [],
                    "sources_used": [],
                    "followups": [],
                    "confidence": "low",
                    "orchestrators_used": orchestrators,
                    "agents_used": [],
                    "conversation_id": conversation_id,
                    "context": {
                        "web_sources": (ctx.get("web_research") or {}).get("sources", []) if isinstance(ctx.get("web_research"), dict) else [],
                        "local_reference_matches": (ctx.get("local_references") or {}).get("matches", []) if isinstance(ctx.get("local_references"), dict) else [],
                        "conversation": ctx.get("conversation"),
                        "sequence_learning": learning,
                    },
                }
                success = False

            elif len(results) == 1:
                one = dict(results[0])
                one["master_orchestrator"] = self.agent_type
                one["orchestrators_used"] = orchestrators
                one["conversation_id"] = conversation_id
                # Carry learning context for transparency/debugging
                try:
                    if isinstance(one.get("context"), dict):
                        one["context"]["sequence_learning_master"] = learning
                except Exception:
                    pass
                final = one

            else:
                sources: List[str] = []
                key_points: List[str] = []
                followups: List[str] = []
                agents_used: List[str] = []

                blocks: List[str] = []
                for r in results:
                    blocks.append(f"## {r.get('domain','Domain')}\n\n{r.get('answer','')}")
                    for u in (r.get("sources_used") or []):
                        if isinstance(u, str) and u not in sources:
                            sources.append(u)
                    for kp in (r.get("key_points") or []):
                        if isinstance(kp, str) and kp not in key_points:
                            key_points.append(kp)
                    for fu in (r.get("followups") or []):
                        if isinstance(fu, str) and fu not in followups:
                            followups.append(fu)
                    for a in (r.get("agents_used") or []):
                        if isinstance(a, str) and a not in agents_used:
                            agents_used.append(a)

                final_answer = "\n\n".join(blocks)

                final = {
                    "domain": self.domain_label,
                    "question": question,
                    "answer": final_answer,
                    "key_points": key_points[:15],
                    "sources_used": _uniq(sources)[:25],
                    "followups": followups[:10],
                    "confidence": "medium",
                    "orchestrators_used": orchestrators,
                    "agents_used": agents_used,
                    "conversation_id": conversation_id,
                    "context": {
                        "web_sources": (ctx.get("web_research") or {}).get("sources", []) if isinstance(ctx.get("web_research"), dict) else [],
                        "local_reference_matches": (ctx.get("local_references") or {}).get("matches", []) if isinstance(ctx.get("local_references"), dict) else [],
                        "conversation": ctx.get("conversation"),
                        "sequence_learning": learning,
                    },
                }

            # Persist master memory
            try:
                mem_payload = {
                    "domain": self.domain_label,
                    "question": question,
                    "answer": final.get("answer"),
                    "key_points": final.get("key_points"),
                    "sources_used": final.get("sources_used"),
                    "orchestrators_used": orchestrators,
                    "agents_used": final.get("agents_used"),
                    "conversation_id": conversation_id,
                }
                self.add_type_memory(json.dumps(mem_payload, ensure_ascii=False), tags=["qa_orchestration", "hive_master"], success=True)
                self.add_hive_memory(json.dumps(mem_payload, ensure_ascii=False), tags=["qa_orchestration", "hive_master"], success=True)
            except Exception:
                pass

            try:
                await self.ctx.bus.publish(
                    "qa",
                    {
                        "domain": self.domain_label,
                        "question": question,
                        "answer": final.get("answer"),
                        "orchestrators_used": orchestrators,
                        "conversation_id": conversation_id,
                    },
                )
            except Exception:
                pass

            # Best-effort: mark failure if the downstream orchestrator produced no usable answer.
            if _looks_like_no_answer_payload(final):
                success = False

            self._trace_event(
                "query_end",
                success=success,
                orchestrators_used=orchestrators,
                agents_used=final.get("agents_used") or [],
                sources_used_count=len(final.get("sources_used") or []),
            )

        except Exception as e:
            success = False
            error = str(e)
            self.log(f"error while answering (master): {e}")
            self._trace_event("error", error=error)
            final = {
                "domain": self.domain_label,
                "question": question,
                "answer": f"An internal error occurred in HiveMasterOrchestrator: {error}",
                "key_points": [],
                "sources_used": [],
                "followups": [],
                "confidence": "low",
                "orchestrators_used": [],
                "agents_used": [],
                "conversation_id": conversation_id,
                "context": {},
            }

        # Persist trace regardless of success
        try:
            await self._commit_trace(
                success=success,
                extra={
                    "error": error,
                    "orchestrators_used": final.get("orchestrators_used") or [],
                    "agents_used": final.get("agents_used") or [],
                    "confidence": final.get("confidence"),
                    "conversation_id": conversation_id,
                },
            )
        except Exception:
            pass

        self.latest_result = final
        self.set_state("idle")
        return final

    def _pick_orchestrators(self, *, question: str, target_orchestrator: Optional[str]) -> List[str]:
        if target_orchestrator:
            t = ORCH_ALIAS_TO_TYPE.get(target_orchestrator, target_orchestrator)
            return [t]

        q = question or ""

        picks: List[str] = []

        # Medicine
        if _contains_any(q, ["symptom", "diagnosis", "medicine", "treatment", "dose", "dosage", "side effect", "fever", "pain", "blood", "infection"]):
            picks.append("medicine_orchestrator")

        # Law
        if _contains_any(q, ["law", "contract", "section", "act", "court", "illegal", "ipc", "crpc", "constitution", "gdpr"]):
            picks.append("law_orchestrator")

        # Finance/Economics
        if _contains_any(q, ["stock", "invest", "investment", "portfolio", "mutual fund", "bond", "interest rate", "loan", "tax", "credit"]):
            picks.append("finance_orchestrator")
        if _contains_any(q, ["gdp", "inflation", "unemployment", "monetary", "fiscal", "macro", "microeconomics", "economics"]):
            picks.append("economics_orchestrator")

        # Social
        if _contains_any(q, ["negotiation", "persuasion", "speech", "communication", "conflict", "relationship", "team dynamics", "social"]):
            picks.append("social_orchestrator")

        # HR
        if _contains_any(q, ["hiring", "interview", "performance", "hr", "payroll", "policy", "termination", "recruitment"]):
            picks.append("hr_orchestrator")

        # Web design
        if _contains_any(
            q,
            [
                "web design",
                "ui",
                "ux",
                "a11y",
                "accessibility",
                "responsive",
                "css",
                "html",
                "tailwind",
                "bootstrap",
                "figma",
                "landing page",
                "design system",
                "typography",
                "color palette",
                "wireframe",
            ],
        ):
            picks.append("web_design_orchestrator")

        # Computer vision
        if _contains_any(
            q,
            [
                "computer vision",
                "opencv",
                "image",
                "video",
                "object detection",
                "segmentation",
                "classification",
                "yolo",
                "cnn",
                "vision transformer",
                "vit",
                "pose",
                "tracking",
            ],
        ):
            picks.append("computer_vision_orchestrator")

        # Cyber
        if _contains_any(q, ["cyber", "phishing", "malware", "ransomware", "vulnerability", "cve", "encryption", "firewall", "incident", "penetration", "pentest", "nmap"]):
            picks.append("cyber_orchestrator")

        # Default
        if not picks:
            picks = ["interactive_orchestrator"]

        uniq: List[str] = []
        for p in picks:
            if p not in uniq:
                uniq.append(p)
        return uniq[:3]
