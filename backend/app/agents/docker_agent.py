from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.llm.prompts import DOCKER_PLAN_SYSTEM
from app.utils.file_utils import list_files
from app.utils.repetition_guard import sanitize_operations
from app.utils.local_code_fallbacks import looks_like_known_python_project_request
from app.utils.nlp_code_planner import is_local_backend


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

    def _default_plan(self, *, entry: str, has_requirements: bool, has_pyproject: bool, reason: str = "") -> Dict[str, Any]:
        dockerfile = self._default_dockerfile(entry=entry, has_requirements=has_requirements, has_pyproject=has_pyproject)
        notes = "Adjust CMD/ENTRYPOINT to match your project."
        if reason:
            notes = f"{notes} {reason}"
        return {
            "summary": "Generated a generic Dockerfile (.dockerignore included).",
            "operations": [
                {"op": "write_file", "path": "Dockerfile", "content": dockerfile},
                {"op": "write_file", "path": ".dockerignore", "content": "__pycache__\n.venv\n.git\nnode_modules\ndist\nbuild\n"},
            ],
            "notes": notes,
            "docker_commands": [],
        }

    def _default_node_plan(self, *, reason: str = "") -> Dict[str, Any]:
        notes = "Generated a generic Node.js Dockerfile."
        if reason:
            notes = f"{notes} {reason}"
        return {
            "summary": "Generated a generic Node.js Dockerfile (.dockerignore included).",
            "operations": [
                {"op": "write_file", "path": "Dockerfile", "content": self._default_node_dockerfile()},
                {"op": "write_file", "path": ".dockerignore", "content": "node_modules\ndist\nbuild\n.git\n.env\n"},
            ],
            "notes": notes,
            "docker_commands": [],
        }

    def _skip_plan(self, *, reason: str = "") -> Dict[str, Any]:
        return {
            "summary": "Docker asset generation skipped.",
            "operations": [],
            "notes": reason or "No stable local Docker template matched this project.",
            "docker_commands": [],
        }

    async def docker_plan(self, *, project_root: Path, user_prompt: str) -> Dict[str, Any]:
        self.set_state("planning")
        self.log("creating docker asset plan")

        py_files = list_files(project_root, exts=[".py"], max_files=200)[:50]
        has_requirements = (project_root / "requirements.txt").exists()
        has_pyproject = (project_root / "pyproject.toml").exists()
        entry = _guess_entry(project_root)

        # Local coding mode should not call another strict JSON planner for Docker.
        # The main ModuleAgent already created/edited project files using text
        # prompts; use stable templates here to avoid code-json-docker loops.
        if is_local_backend(self.ctx.llm):
            if py_files or has_requirements or has_pyproject:
                plan = self._default_plan(
                    entry=entry,
                    has_requirements=has_requirements,
                    has_pyproject=has_pyproject,
                    reason="Local backend detected; skipped code-json-docker LLM call.",
                )
            elif (project_root / "package.json").exists():
                plan = self._default_node_plan(reason="Local backend detected; skipped code-json-docker LLM call.")
            else:
                plan = self._skip_plan(reason="Local backend detected and no Python/Node Docker template matched.")
            plan["metadata"] = {"local_docker_template": True, "skip_local_json_docker": True}
            self.latest_result = plan
            self.remember(json.dumps(plan), tags=["docker", "local_template"], success=None)
            self.add_type_memory(json.dumps(plan), tags=["docker", "local_template"], success=None)
            self.set_state("idle")
            return plan

        if looks_like_known_python_project_request(user_prompt, project_root=project_root):
            plan = self._default_plan(
                entry=entry,
                has_requirements=has_requirements,
                has_pyproject=has_pyproject,
                reason="Simple Python request detected; skipped local code-json-docker LLM call.",
            )
            plan["metadata"] = {"deterministic_fallback": True, "fallback_kind": "simple_python_docker"}
            self.latest_result = plan
            self.remember(json.dumps(plan), tags=["docker", "deterministic_fallback"], success=None)
            self.add_type_memory(json.dumps(plan), tags=["docker", "deterministic_fallback"], success=None)
            self.set_state("idle")
            return plan

        if not self.llm_available():
            plan = self._default_plan(entry=entry, has_requirements=has_requirements, has_pyproject=has_pyproject, reason="LLM unavailable.")
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

        try:
            plan = await self.ctx.llm.chat_json_async(
                system=DOCKER_PLAN_SYSTEM, messages=messages, temperature=0.2, purpose="code-json-docker"
            )
        except Exception as e:
            plan = self._default_plan(
                entry=entry,
                has_requirements=has_requirements,
                has_pyproject=has_pyproject,
                reason=f"Local Docker planner failed ({type(e).__name__}); used stable fallback.",
            )
        if not isinstance(plan, dict):
            plan = self._default_plan(
                entry=entry,
                has_requirements=has_requirements,
                has_pyproject=has_pyproject,
                reason="Docker planner returned a non-object response; used stable fallback.",
            )
        plan.setdefault("operations", [])
        plan.setdefault("notes", "")
        plan.setdefault("summary", "")
        plan["operations"], dropped_ops = sanitize_operations(plan.get("operations") or [])
        if dropped_ops:
            plan["notes"] = (str(plan.get("notes") or "") + f"\nDropped {dropped_ops} unsafe repeated-token Docker operation(s).").strip()

        if not plan.get("operations") and (py_files or looks_like_known_python_project_request(user_prompt, project_root=project_root)):
            plan = self._default_plan(
                entry=entry,
                has_requirements=has_requirements,
                has_pyproject=has_pyproject,
                reason="Docker planner produced no safe operations; used stable fallback.",
            )

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

    def _default_node_dockerfile(self) -> str:
        return (
            "FROM node:20-slim\n"
            "WORKDIR /app\n"
            "COPY package*.json ./\n"
            "RUN npm install --omit=dev || npm install\n"
            "COPY . .\n"
            "CMD [\"npm\", \"start\"]\n"
        )


