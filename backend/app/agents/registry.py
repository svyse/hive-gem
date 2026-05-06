from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from app.agents.base import AgentContext, BaseAgent
from app.agents.command_agent import CommandAgent
from app.agents.compiling_agent import CompilingAgent
from app.agents.complex_compiling_agent import ComplexCompilingAgent
from app.agents.docker_agent import DockerAgent
from app.agents.logic_agent import LogicAgent
from app.agents.module_agent import ModuleAgent
from app.agents.orchestrator import OrchestratorAgent
from app.agents.testing_agent import TestingAgent
from app.agents.local_learning_agent import LocalLearningAgent

from app.agents.local_reference_agent import LocalReferenceAgent
from app.agents.web_research_agent import WebResearchAgent
from app.agents.web_design_agent import WebDesignAgent
from app.agents.web_scraping_agent import WebScrapingAgent

from app.agents.logging_agent import LoggingAgent
from app.agents.strategist_agent import StrategistAgent
from app.agents.workspace_module_agent import WorkspaceModuleAgent

from app.agents.qa_base import DomainQAAAgent
from app.agents.qa_domain_agents import (
    BiologyQAAgent,
    ComputerVisionAgent,
    CyberNetworkingAgent,
    CyberSecurityAgent,
    EconomicsAgent,
    EngineeringQAAgent,
    FinanceAgent,
    HRAgent,
    LawIndiaAgent,
    LawInternationalAgent,
    MathQAAgent,
    MedicineAgent,
    PentestingAgent,
    ScienceQAAgent,
    SocialDynamicsAgent,
    ComputingQAAgent,
    WebDesignAgent as QAWebDesignAgent,
)

from app.agents.hive_master_orchestrator import HiveMasterOrchestrator
from app.agents.interactive_orchestrator import InteractiveOrchestrator
from app.agents.hr_orchestrator import HROrchestrator
from app.agents.law_orchestrator import LawOrchestrator, LawIndiaOrchestrator, LawInternationalOrchestrator
from app.agents.finance_orchestrator import FinanceOrchestrator
from app.agents.economics_orchestrator import EconomicsOrchestrator
from app.agents.social_orchestrator import SocialOrchestrator
from app.agents.medicine_orchestrator import MedicineOrchestrator
from app.agents.cyber_orchestrator import CyberOrchestrator
from app.agents.web_design_orchestrator import WebDesignOrchestrator
from app.agents.computer_vision_orchestrator import ComputerVisionOrchestrator

from app.api.schemas import AgentStatus
from app.core.config import settings
from app.llm.factory import get_llm_client
from app.llm.scoped_client import ScopedLLMClient
from app.memory.store import MemoryStore


AGENT_CLASSES: Dict[str, Type[BaseAgent]] = {
    # Core
    "base": BaseAgent,
    "orchestrator": OrchestratorAgent,

    # Code pipeline workers
    "module": ModuleAgent,
    "logic": LogicAgent,
    "command": CommandAgent,
    "testing": TestingAgent,
    "compiling": CompilingAgent,
    "complex_compiling": ComplexCompilingAgent,
    "docker": DockerAgent,

    # Resources
    "local_reference": LocalReferenceAgent,
    "web_research": WebResearchAgent,
    "web_scraper": WebScrapingAgent,
    "web_design": WebDesignAgent,

    # Meta
    "logging": LoggingAgent,
    "strategist": StrategistAgent,
    "local_learning": LocalLearningAgent,

    # Workspace tools
    "workspace_module": WorkspaceModuleAgent,

    # Q&A
    "qa_base": DomainQAAAgent,
    "qa_science": ScienceQAAgent,
    "qa_computing": ComputingQAAgent,
    "qa_engineering": EngineeringQAAgent,
    "qa_math": MathQAAgent,
    "qa_biology": BiologyQAAgent,
    "qa_hr": HRAgent,
    "qa_law_india": LawIndiaAgent,
    "qa_law_international": LawInternationalAgent,
    "qa_finance": FinanceAgent,
    "qa_economics": EconomicsAgent,
    "qa_social": SocialDynamicsAgent,
    "qa_medicine": MedicineAgent,
    "qa_cyber": CyberSecurityAgent,
    "qa_cyber_networking": CyberNetworkingAgent,
    "qa_pentesting": PentestingAgent,
    "qa_web_design": QAWebDesignAgent,
    "qa_computer_vision": ComputerVisionAgent,

    # Orchestrators (Q&A)
    "hive_master_orchestrator": HiveMasterOrchestrator,
    "interactive_orchestrator": InteractiveOrchestrator,
    "hr_orchestrator": HROrchestrator,
    "law_orchestrator": LawOrchestrator,
    "law_india_orchestrator": LawIndiaOrchestrator,
    "law_international_orchestrator": LawInternationalOrchestrator,
    "finance_orchestrator": FinanceOrchestrator,
    "economics_orchestrator": EconomicsOrchestrator,
    "social_orchestrator": SocialOrchestrator,
    "medicine_orchestrator": MedicineOrchestrator,
    "cyber_orchestrator": CyberOrchestrator,
    "web_design_orchestrator": WebDesignOrchestrator,
    "computer_vision_orchestrator": ComputerVisionOrchestrator,
}


def _is_orchestrator_type(agent_type: str) -> bool:
    t = (agent_type or "").lower()
    return t == "orchestrator" or t.endswith("_orchestrator")


# Agents that primarily perform software-factory / coding work.
# We keep the existing orchestrator+agent hierarchy intact; this is only used
# to provide an LLM routing hint (purpose="code") so OpenAI can optionally use
# a different model for code runs.
_CODE_AGENT_TYPES = {
    "orchestrator",
    "module",
    "logic",
    "command",
    "testing",
    "compiling",
    "complex_compiling",
    "docker",
    "workspace_module",
}


@dataclass
class AgentRegistry:
    """Spawn and track agents with per-type concurrency limits."""

    bus: Any
    memory_store: MemoryStore
    run_id: Optional[str]
    run_logger: Callable[[str], None]
    status_reporter: Callable[[AgentStatus], None]
    agent_max_per_type: int = int(getattr(settings, "agent_max_per_type", 6) or 6)
    max_orchestrators: int = int(getattr(settings, "max_orchestrators", 2) or 2)
    agent_type_limits: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.agent_type_limits = dict(getattr(settings, "agent_type_limits", {}) or {}) if self.agent_type_limits is None else dict(self.agent_type_limits)
        self._lock = asyncio.Lock()
        self._agents: Dict[str, BaseAgent] = {}
        self._agents_by_type: Dict[str, List[str]] = {}
        self.llm = get_llm_client()

    def _limit_for(self, agent_type: str) -> int:
        if agent_type in (self.agent_type_limits or {}):
            return int(self.agent_type_limits[agent_type])
        if _is_orchestrator_type(agent_type):
            return int(self.max_orchestrators)
        return int(self.agent_max_per_type)

    def _count_for(self, agent_type: str) -> int:
        return len(self._agents_by_type.get(agent_type, []))

    async def spawn(self, agent_type: str) -> BaseAgent:
        agent_type = (agent_type or "").strip()
        if not agent_type:
            raise ValueError("agent_type must not be empty")

        cls = AGENT_CLASSES.get(agent_type)
        if cls is None:
            raise KeyError(f"Unknown agent type: {agent_type}. Known={sorted(AGENT_CLASSES.keys())}")

        async with self._lock:
            limit = self._limit_for(agent_type)
            current = self._count_for(agent_type)
            if current >= limit:
                raise RuntimeError(f"Agent type '{agent_type}' at concurrency limit ({current}/{limit})")

            agent_id = f"{agent_type}-{uuid.uuid4().hex[:8]}"

            # Route OpenAI model selection by "purpose" without changing the
            # agent/orchestrator hierarchy. If OPENAI_MODEL_CODE is not set,
            # code agents will simply use OPENAI_MODEL.
            # Heuristic:
            # - In Q&A turns, run_id is "turn-..." and the OrchestratorAgent is sometimes
            #   spawned as a lightweight helper to shard work (e.g. web research). In that
            #   context we treat it as QA, not code.
            # - In software-factory runs, run_id is a short hex id and "orchestrator" is
            #   the main code pipeline orchestrator.
            rid = str(self.run_id or "")
            is_qa_turn = rid.startswith("turn-")
            if is_qa_turn and agent_type == "orchestrator":
                purpose = "qa"
            else:
                purpose = "code" if agent_type in _CODE_AGENT_TYPES else "qa"

            ctx = AgentContext(
                run_id=str(self.run_id or "qa"),
                bus=self.bus,
                memory_store=self.memory_store,
                run_logger=self.run_logger,
                status_reporter=self.status_reporter,
                registry=self,
                llm=ScopedLLMClient(self.llm, purpose=purpose),
            )

            agent = cls(agent_id=agent_id, ctx=ctx)  # type: ignore[call-arg]
            self._agents[agent_id] = agent
            self._agents_by_type.setdefault(agent_type, []).append(agent_id)

        return agent

    def get(self, agent_id: str) -> Optional[BaseAgent]:
        return self._agents.get(agent_id)

    def list_agents(self) -> List[BaseAgent]:
        return list(self._agents.values())

    async def terminate(self, agent_id: str) -> None:
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        try:
            await agent.terminate()
        finally:
            async with self._lock:
                self._agents.pop(agent_id, None)
                # remove from type list
                ids = self._agents_by_type.get(agent.agent_type, [])
                if agent_id in ids:
                    ids.remove(agent_id)
                if not ids and agent.agent_type in self._agents_by_type:
                    self._agents_by_type.pop(agent.agent_type, None)

    async def shutdown(self) -> None:
        # Terminate all agents (best-effort)
        ids = list(self._agents.keys())
        for aid in ids:
            try:
                await self.terminate(aid)
            except Exception:
                continue

    async def shutdown_all(self) -> None:
        """Backward-compatible alias.

        Some runtime components historically called `shutdown_all()`.
        The registry implementation uses `shutdown()` as the canonical
        method, so we expose this alias to avoid runtime failures.
        """

        await self.shutdown()
