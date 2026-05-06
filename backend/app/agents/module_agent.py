from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.llm.prompts import MODULE_PLAN_SYSTEM
from app.utils.file_utils import list_files, read_text


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...<truncated>...\n"


class ModuleAgent(BaseAgent):
    agent_type = "module"

    async def make_plan(self, *, project_root: Path, user_prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.set_state("planning")
        self.log("building change plan")

        py_files = list_files(project_root, exts=[".py"], max_files=200)
        # Prefer top-level and smaller contexts; sample up to 12 files
        sample_files = py_files[:12]

        snapshot: List[Dict[str, str]] = []
        for fp in sample_files:
            try:
                content = read_text(project_root, fp)
            except Exception:
                continue
            snapshot.append({"path": fp, "content": _truncate(content, 3500)})

        hive_recent = self.ctx.memory_store.recent(scope="hive", limit=5)

        if not self.llm_available():
            plan = {
                "summary": "LLM unavailable (set OPENAI_API_KEY). Created a placeholder plan file.",
                "operations": [
                    {
                        "op": "write_file",
                        "path": "AGENTIC_PLAN.md",
                        "content": (
                            "# Agentic Plan Placeholder\n\n"
                            "The ModuleAgent could not call the LLM because OPENAI_API_KEY is not set.\n\n"
                            f"## Prompt\n{user_prompt}\n"
                        ),
                    }
                ],
                "test_commands": [],
                "notes": "Set OPENAI_API_KEY in backend/.env to enable LLM-backed planning.",
            }
            self.latest_result = plan
            self.remember(json.dumps(plan), tags=["plan", "module"], success=None)
            self.set_state("idle")
            return plan

        # Use type/hive memory to prime successful patterns
        mem_bundle = self.get_memory_bundle("plan", include_type=True, include_hive=True, include_agent=False, limit_per_scope=6)
        memory_context = self.format_memory_bundle(mem_bundle)

        # Ask LLM for a structured plan
        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_prompt": user_prompt,
                        "project_root": str(project_root),
                        "python_files": py_files,
                        "snapshot": snapshot,
                        "hive_recent": hive_recent,
                        "memory_context": memory_context,
                        "external_context": context or {},
                        "required_output_schema": {
                            "summary": "string",
                            "operations": [
                                {
                                    "op": "mkdir|write_file|delete_file",
                                    "path": "relative/path",
                                    "content": "string (required for write_file)",
                                }
                            ],
                            "test_commands": ["string"],
                            "notes": "string",
                        },
                        "constraints": [
                            "Paths must be relative to project_root.",
                            "Prefer minimal changes and small, testable modules.",
                            "Do not include markdown code fences in JSON.",
                        ],
                    },
                    indent=2,
                ),
            }
        ]

        plan = await self.ctx.llm.chat_json_async(system=MODULE_PLAN_SYSTEM, messages=messages, temperature=0.2)

        # Minimal validation/sanitization
        if not isinstance(plan, dict):
            raise ValueError("ModuleAgent plan must be a JSON object")

        plan.setdefault("operations", [])
        plan.setdefault("test_commands", [])
        plan.setdefault("summary", "")
        plan.setdefault("notes", "")

        self.latest_result = plan
        self.remember(json.dumps(plan), tags=["plan", "module"], success=None)
        self.add_type_memory(json.dumps(plan), tags=["plan", "module"], success=None)

        self.set_state("idle")
        return plan
