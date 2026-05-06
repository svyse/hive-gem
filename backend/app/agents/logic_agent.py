from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.agents.base import BaseAgent
from app.llm.prompts import LOGIC_REVIEW_SYSTEM
from app.utils.file_utils import list_files, read_text


def _truncate(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else (s[:max_chars] + "\n...<truncated>...\n")


class LogicAgent(BaseAgent):
    agent_type = "logic"

    async def review_plan(self, *, project_root: Path, user_prompt: str, plan: Dict[str, Any], context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.set_state("reviewing")
        self.log("reviewing plan")

        if not self.llm_available():
            review = {
                "approved": True,
                "issues": ["LLM unavailable; performed no deep logic review."],
                "recommended_plan_patch": {"operations": []},
            }
            self.latest_result = review
            self.remember(json.dumps(review), tags=["logic", "review"], success=None)
            self.set_state("idle")
            return review

        # Provide a small snapshot context of touched files
        touched = []
        for op in plan.get("operations", []):
            p = op.get("path")
            if p and isinstance(p, str):
                touched.append(p)
        touched = list(dict.fromkeys(touched))[:10]

        touched_contents = []
        for p in touched:
            try:
                content = read_text(project_root, p)
                touched_contents.append({"path": p, "content": _truncate(content, 2500)})
            except Exception:
                pass

        mem_bundle = self.get_memory_bundle("review", include_type=True, include_hive=True, include_agent=False, limit_per_scope=6)
        memory_context = self.format_memory_bundle(mem_bundle)

        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_prompt": user_prompt,
                        "plan": plan,
                        "touched_file_snapshot": touched_contents,
                        "memory_context": memory_context,
                        "external_context": context or {},
                        "required_output_schema": {
                            "approved": "boolean",
                            "issues": ["string"],
                            "recommended_plan_patch": {
                                "operations": [
                                    {"op": "mkdir|write_file|delete_file", "path": "...", "content": "..."}
                                ]
                            },
                        },
                        "constraints": [
                            "Output must be valid JSON only.",
                            "If approved=false, provide concrete issues.",
                            "recommended_plan_patch.operations should be minimal edits/patches.",
                        ],
                    },
                    indent=2,
                ),
            }
        ]

        review = await self.ctx.llm.chat_json_async(system=LOGIC_REVIEW_SYSTEM, messages=messages, temperature=0.2)

        if not isinstance(review, dict):
            raise ValueError("LogicAgent review must be a JSON object")

        review.setdefault("approved", True)
        review.setdefault("issues", [])
        review.setdefault("recommended_plan_patch", {"operations": []})

        self.latest_result = review
        self.remember(json.dumps(review), tags=["logic", "review"], success=None)
        self.add_type_memory(json.dumps(review), tags=["logic", "review"], success=None)
        self.set_state("idle")
        return review
