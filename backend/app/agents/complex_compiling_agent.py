from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.agents.base import BaseAgent
from app.agents.command_agent import CommandAgent
from app.utils.file_utils import list_files


class ComplexCompilingAgent(BaseAgent):
    agent_type = "complex_compiling"

    async def assess_structure(self, *, project_root: Path, command_agent: CommandAgent) -> Dict[str, Any]:
        self.set_state("structuring")
        self.log("assessing project structure")

        files = list_files(project_root, exts=None, max_files=2000)
        has_pyproject = (project_root / "pyproject.toml").exists()
        has_setup = (project_root / "setup.py").exists()
        has_requirements = (project_root / "requirements.txt").exists()
        has_tests = (project_root / "tests").exists()

        recommendations: List[str] = []
        if not has_pyproject and not has_setup:
            recommendations.append("Consider adding pyproject.toml (packaging) if this is a library/app.")
        if not has_requirements and not has_pyproject:
            recommendations.append("Consider adding requirements.txt or pyproject dependencies.")
        if not has_tests:
            recommendations.append("Consider adding a tests/ folder and pytest.")

        out = {
            "files_count": len(files),
            "has_pyproject": has_pyproject,
            "has_setup": has_setup,
            "has_requirements": has_requirements,
            "has_tests": has_tests,
            "recommendations": recommendations,
        }

        self.latest_result = out
        self.remember(json.dumps(out), tags=["structure"], success=None)
        self.add_type_memory(json.dumps(out), tags=["structure"], success=None)

        self.set_state("idle")
        return out
