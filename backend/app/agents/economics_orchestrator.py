from __future__ import annotations

from typing import Any, Dict, List

from app.agents.domain_orchestrator_base import DomainOrchestratorBase


class EconomicsOrchestrator(DomainOrchestratorBase):
    agent_type = "economics_orchestrator"
    domain_label = "Economics"

    def select_agents(self, *, question: str, context: Dict[str, Any]) -> List[str]:
        return ["qa_economics"]
