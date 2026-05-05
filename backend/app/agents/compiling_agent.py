from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.agents.base import BaseAgent
from app.agents.command_agent import CommandAgent
from app.utils.file_utils import list_files


class CompilingAgent(BaseAgent):
    agent_type = "compiling"

    async def compile_check(self, *, project_root: Path, command_agent: CommandAgent) -> Dict[str, Any]:
        """For Python, a pragmatic 'compile' step is: run py_compile across files."""
        self.set_state("compiling")
        self.log("running python -m py_compile across project")

        py_files = list_files(project_root, exts=[".py"], max_files=500)
        errors: List[Dict[str, Any]] = []

        for fp in py_files:
            cmd = ["python", "-m", "py_compile", str(project_root / fp)]
            try:
                res = command_agent.exec(cmd, cwd=project_root, timeout_s=120)
                if res.returncode != 0:
                    errors.append({"file": fp, "stderr": res.stderr[-4000:], "stdout": res.stdout[-2000:]})
            except Exception as e:
                errors.append({"file": fp, "error": str(e)})

        ok = len(errors) == 0
        out = {"ok": ok, "checked_files": len(py_files), "errors": errors}

        self.latest_result = out
        self.remember(json.dumps(out), tags=["compiling"], success=ok)
        self.add_type_memory(json.dumps(out), tags=["compiling"], success=ok)

        self.set_state("idle")
        return out
