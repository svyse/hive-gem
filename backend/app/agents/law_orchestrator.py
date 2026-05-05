from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.domain_orchestrator_base import DomainOrchestratorBase


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


class LawIndiaOrchestrator(DomainOrchestratorBase):
    agent_type = "law_india_orchestrator"
    domain_label = "Law (India)"

    def select_agents(self, *, question: str, context: Dict[str, Any]) -> List[str]:
        return ["qa_law_india"]


class LawInternationalOrchestrator(DomainOrchestratorBase):
    agent_type = "law_international_orchestrator"
    domain_label = "Law (International)"

    def select_agents(self, *, question: str, context: Dict[str, Any]) -> List[str]:
        return ["qa_law_international"]


class LawOrchestrator(DomainOrchestratorBase):
    """Router orchestrator that delegates to Indian or International law orchestrators."""

    agent_type = "law_orchestrator"
    domain_label = "Law"

    async def answer(
        self,
        *,
        question: str,
        project_root: Optional[Path] = None,
        use_web: bool = True,
        use_local_refs: bool = True,
        max_web_queries: int = 2,
    ) -> Dict[str, Any]:
        # Heuristic routing
        q = question or ""
        indiaish = _contains_any(q, [
            "india",
            "indian",
            "ipc",
            "crpc",
            "companies act",
            "gst",
            "itr",
            "supreme court",
            "high court",
            "constitution of india",
        ])
        internationalish = _contains_any(q, ["international", "treaty", "eu", "uk", "usa", "un", "gdpr", "icc", "wto"])

        if indiaish and not internationalish:
            res = await self.consult_orchestrator(orchestrator_type="law_india_orchestrator", question=question, project_root=project_root)
            if res:
                return res
        if internationalish and not indiaish:
            res = await self.consult_orchestrator(orchestrator_type="law_international_orchestrator", question=question, project_root=project_root)
            if res:
                return res

        # Ambiguous: answer with both agents directly (keeps it simple)
        return await super().answer(
            question=question,
            project_root=project_root,
            use_web=use_web,
            use_local_refs=use_local_refs,
            max_web_queries=max_web_queries,
        )

    def select_agents(self, *, question: str, context: Dict[str, Any]) -> List[str]:
        # If the router couldn't disambiguate, consult both.
        return ["qa_law_india", "qa_law_international"]
