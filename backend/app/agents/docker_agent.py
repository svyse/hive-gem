from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.llm.prompts import DOCKER_PLAN_SYSTEM
from app.utils.file_utils import list_files


def _guess_entry(project_root: Path) -> str:
    for name in ["main.py", "app.py", "__main__.py"]:
        if (project_root / name).exists():
            return name
    # fallback: first top-level py file
    for p in project_root.glob("*.py"):
        return p.name
    return "app.py"


class DockerAgent(BaseAgent):
    agent_type = "docker"

    async def docker_plan(self, *, project_root: Path, user_prompt: str) -> Dict[str, Any]:
        self.set_state("planning")
        self.log("creating docker asset plan")

        py_files = list_files(project_root, exts=[".py"], max_files=200)[:50]
        has_requirements = (project_root / "requirements.txt").exists()
        has_pyproject = (project_root / "pyproject.toml").exists()
        entry = _guess_entry(project_root)

        if not self.llm_available():
            dockerfile = self._default_dockerfile(entry=entry, has_requirements=has_requirements, has_pyproject=has_pyproject)
            plan = {
                "summary": "Generated a generic Dockerfile (.dockerignore included).",

                "operations": [
                    {"op": "write_file", "path": "Dockerfile", "content": dockerfile},
                    {"op": "write_file", "path": ".dockerignore", "content": "__pycache__\n.venv\n.git\n"},
                ],
                "notes": "Adjust CMD/ENTRYPOINT to match your project.",
            }
            self.latest_result = plan
            self.remember(json.dumps(plan), tags=["docker"], success=None)
            self.set_state("idle")
            return plan

        mem_bundle = self.get_memory_bundle("docker", include_type=True, include_hive=True, include_agent=False, limit_per_scope=5)
        memory_context = self.format_memory_bundle(mem_bundle)

        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_prompt": user_prompt,
                        "project_root": str(project_root),
                        "python_files": py_files,
                        "memory_context": memory_context,
                        "hints": {
                            "has_requirements": has_requirements,
                            "has_pyproject": has_pyproject,
                            "entry_guess": entry,
                        },
                        "required_output_schema": {
                            "summary": "string",
                            "operations": [
                                {"op": "write_file", "path": "Dockerfile", "content": "..."},
                                {"op": "write_file", "path": ".dockerignore", "content": "..."},
                            ],
                            "notes": "string",
                        },
                        "constraints": [
                            "Output valid JSON only.",
                            "Prefer python:3.11-slim base.",
                            "If requirements.txt exists, install from it; else include guidance comments.",
                        ],
                    },
                    indent=2,
                ),
            }
        ]

        plan = await self.ctx.llm.chat_json_async(system=DOCKER_PLAN_SYSTEM, messages=messages, temperature=0.2)
        if not isinstance(plan, dict):
            raise ValueError("DockerAgent plan must be JSON object")
        plan.setdefault("operations", [])
        plan.setdefault("notes", "")
        plan.setdefault("summary", "")
        self.latest_result = plan
        self.remember(json.dumps(plan), tags=["docker"], success=None)
        self.add_type_memory(json.dumps(plan), tags=["docker"], success=None)
        self.set_state("idle")
        return plan

    def _default_dockerfile(self, *, entry: str, has_requirements: bool, has_pyproject: bool) -> str:
        lines: List[str] = []
        lines.append("FROM python:3.11-slim")
        lines.append("WORKDIR /app")
        lines.append("\n# System deps (add build-essential if you compile wheels)\nRUN apt-get update && apt-get install -y --no-install-recommends \\")
        lines.append("    ca-certificates \\")
        lines.append(" && rm -rf /var/lib/apt/lists/*")
        lines.append("\nCOPY . /app\n")
        if has_requirements:
            lines.append("RUN pip install --no-cache-dir -r requirements.txt")
        elif has_pyproject:
            lines.append("# This project has pyproject.toml. Consider installing with:\n# RUN pip install --no-cache-dir .")
        else:
            lines.append("# No requirements.txt found. Add dependency installation steps here.")
        lines.append(f"\nCMD [\"python\", \"{entry}\"]")
        return "\n".join(lines) + "\n"
