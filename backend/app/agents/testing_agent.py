from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.agents.command_agent import CommandAgent
from app.llm.prompts import TEST_PLAN_SYSTEM
from app.utils.file_utils import list_files, read_text


class TestingAgent(BaseAgent):
    agent_type = "testing"

    async def ensure_and_run_tests(
        self,
        *,
        project_root: Path,
        user_prompt: str,
        plan: Dict[str, Any],
        command_agent: CommandAgent,
    ) -> Dict[str, Any]:
        self.set_state("testing")
        self.log("starting test phase")

        test_commands: List[str] = plan.get("test_commands") or []
        results: List[Dict[str, Any]] = []

        # If plan didn't specify tests, attempt a default.
        if not test_commands:
            if (project_root / "tests").exists():
                test_commands = ["pytest -q"]
            else:
                test_commands = []

        # Optionally generate tests if none exist and LLM is available.
        generated_tests: List[Dict[str, str]] = []
        if not test_commands and self.llm_available():
            self.log("no tests specified; attempting to generate basic pytest scaffold")
            py_files = list_files(project_root, exts=[".py"], max_files=50)[:8]
            snapshot = []
            for fp in py_files:
                try:
                    snapshot.append({"path": fp, "content": read_text(project_root, fp)[:2500]})
                except Exception:
                    pass

            mem_bundle = self.get_memory_bundle("testing", include_type=True, include_hive=True, include_agent=False, limit_per_scope=5)
            memory_context = self.format_memory_bundle(mem_bundle)

            messages = [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_prompt": user_prompt,
                            "plan_summary": plan.get("summary", ""),
                            "project_snapshot": snapshot,
                            "memory_context": memory_context,
                            "required_output_schema": {
                                "operations": [
                                    {"op": "mkdir|write_file", "path": "...", "content": "..."}
                                ],
                                "test_commands": ["string"],
                                "notes": "string",
                            },
                            "constraints": [
                                "Create minimal pytest tests to validate the prompt feature.",
                                "Prefer creating tests/test_basic.py.",
                                "Output valid JSON only.",
                            ],
                        },
                        indent=2,
                    ),
                }
            ]
            test_plan = await self.ctx.llm.chat_json_async(system=TEST_PLAN_SYSTEM, messages=messages, temperature=0.2)
            if isinstance(test_plan, dict):
                ops = test_plan.get("operations") or []
                for op in ops:
                    if op.get("op") == "mkdir":
                        command_agent.mkdir(project_root, op["path"])
                    elif op.get("op") == "write_file":
                        command_agent.write_file(project_root, op["path"], op.get("content", ""))
                test_commands = (test_plan.get("test_commands") or []) or ["pytest -q"]
                generated_tests = ops

        for cmd_str in test_commands:
            cmd = shlex.split(cmd_str)
            try:
                res = command_agent.exec(cmd, cwd=project_root, timeout_s=300)
                results.append(
                    {
                        "command": cmd_str,
                        "returncode": res.returncode,
                        "stdout": res.stdout[-8000:],
                        "stderr": res.stderr[-8000:],
                    }
                )
            except Exception as e:
                results.append({"command": cmd_str, "error": str(e)})

        summary = "passed" if results and all(r.get("returncode", 1) == 0 for r in results) else "failed_or_skipped"
        out = {"summary": summary, "results": results, "generated_test_ops": generated_tests}

        self.latest_result = out
        self.remember(json.dumps(out), tags=["testing"], success=(summary == "passed"))
        self.add_type_memory(json.dumps(out), tags=["testing"], success=(summary == "passed"))

        self.set_state("idle")
        return out
