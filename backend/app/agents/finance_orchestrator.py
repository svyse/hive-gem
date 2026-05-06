from __future__ import annotations

from typing import Any, Dict, List

from app.agents.domain_orchestrator_base import DomainOrchestratorBase


class FinanceOrchestrator(DomainOrchestratorBase):
    agent_type = "finance_orchestrator"
    domain_label = "Finance"

    def select_agents(self, *, question: str, context: Dict[str, Any]) -> List[str]:
        return ["qa_finance"]
