from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.agents.base import BaseAgent


STRATEGIST_SYSTEM = """You are a Strategist agent in a multi-orchestrator hive.

You help orchestrators and specialist agents by:
- clarifying goals
- proposing a step-by-step strategy
- identifying risks and dependencies
- recommending which specialist agents/orchestrators should handle which subtask

Return VALID JSON only (no markdown fences):
{
  "summary": "...",
  "steps": ["..."],
  "delegations": [{"to": "agent_or_orchestrator_type", "task": "..."}],
  "risks": ["..."],
  "open_questions": ["..."]
}
"""


class StrategistAgent(BaseAgent):
    """A helper agent that produces strategy and delegation suggestions."""

    agent_type = "strategist"

    async def strategize(self, *, goal: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.set_state("strategizing")
        self.log("building strategy")

        mem_bundle = self.get_memory_bundle(goal, include_type=True, include_hive=True, include_agent=False, limit_per_scope=6)
        memory_context = self.format_memory_bundle(mem_bundle)

        if not self.llm_available():
            out = {
                "summary": "LLM unavailable; returning a minimal strategy skeleton.",
                "steps": ["Set OPENAI_API_KEY to enable strategist reasoning."],
                "delegations": [],
                "risks": ["No LLM configured."],
                "open_questions": [],
            }
            self.latest_result = out
            self.remember(json.dumps(out), tags=["strategy"], success=None)
            self.add_type_memory(json.dumps(out), tags=["strategy"], success=None)
            self.set_state("idle")
            return out

        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "goal": goal,
                        "context": context or {},
                        "memory_context": memory_context,
                        "constraints": [
                            "Be concrete and actionable.",
                            "If delegation helps, suggest specific agent/orchestrator types.",
                            "Do not invent capabilities.",
                            "Keep it concise.",
                        ],
                    },
                    indent=2,
                ),
            }
        ]

        # Run LLM calls in a thread to avoid blocking the FastAPI event loop.
        out = await self.ctx.llm.chat_json_async(system=STRATEGIST_SYSTEM, messages=messages, temperature=0.2)
        if not isinstance(out, dict):
            raise ValueError("StrategistAgent must return a JSON object")

        out.setdefault("summary", "")
        out.setdefault("steps", [])
        out.setdefault("delegations", [])
        out.setdefault("risks", [])
        out.setdefault("open_questions", [])

        self.latest_result = out
        self.remember(json.dumps(out), tags=["strategy"], success=True)
        self.add_type_memory(json.dumps(out), tags=["strategy"], success=True)
        compact = {"goal": goal, "summary": out.get("summary"), "steps": (out.get("steps") or [])[:6]}
        self.add_hive_memory(json.dumps(compact), tags=["strategy"], success=True)

        self.set_state("idle")
        return out
