from __future__ import annotations

from typing import Any, Dict, List

from app.agents.domain_orchestrator_base import DomainOrchestratorBase


class MedicineOrchestrator(DomainOrchestratorBase):
    agent_type = "medicine_orchestrator"
    domain_label = "Medicine"

    def select_agents(self, *, question: str, context: Dict[str, Any]) -> List[str]:
        return ["qa_medicine"]
