from __future__ import annotations

from typing import Any, Dict, List

from app.agents.domain_orchestrator_base import DomainOrchestratorBase


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


class WebDesignOrchestrator(DomainOrchestratorBase):
    """Specialized orchestrator for web design.

    Focus:
    - UI/UX, accessibility (a11y), responsive layout, design systems
    - HTML/CSS, modern frontend styling patterns

    It can be spawned by the HiveMasterOrchestrator when the user's query
    is clearly web-design oriented, but it can also be consulted by any
    other orchestrator via DomainOrchestratorBase.consult_orchestrator.
    """

    agent_type = "web_design_orchestrator"
    domain_label = "Web design"

    def select_agents(self, *, question: str, context: Dict[str, Any]) -> List[str]:
        q = question or ""
        picks: List[str] = ["qa_web_design"]

        # If the question is implementation-heavy (React/Vue/etc), include a computing agent too.
        if _contains_any(
            q,
            [
                "react",
                "next.js",
                "nextjs",
                "vue",
                "svelte",
                "angular",
                "tailwind",
                "bootstrap",
                "css",
                "html",
                "javascript",
                "typescript",
                "frontend",
                "component",
                "design system",
            ],
        ):
            picks.append("qa_computing")

        return picks[:3]
