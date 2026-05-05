from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

from app.agents.base import BaseAgent
from app.utils import file_utils
from app.utils.safe_exec import run_command, ExecResult, CommandNotAllowed


class CommandAgent(BaseAgent):
    agent_type = "command"

    def mkdir(self, project_root: Path, rel_path: str) -> None:
        self.set_state("working")
        self.log(f"mkdir {rel_path}")
        file_utils.mkdir(project_root, rel_path)
        self.remember(f"mkdir: {rel_path}", tags=["fs"])
        self.set_state("idle")

    def write_file(self, project_root: Path, rel_path: str, content: str) -> None:
        self.set_state("working")
        self.log(f"write_file {rel_path} ({len(content)} bytes)")
        file_utils.write_text(project_root, rel_path, content)
        self.remember(f"write_file: {rel_path}", tags=["fs"])
        self.set_state("idle")

    def read_file(self, project_root: Path, rel_path: str) -> str:
        self.set_state("working")
        self.log(f"read_file {rel_path}")
        content = file_utils.read_text(project_root, rel_path)
        self.set_state("idle")
        return content

    def list_py_files(self, project_root: Path, max_files: int = 2000) -> List[str]:
        self.set_state("working")
        files = file_utils.list_files(project_root, exts=[".py"], max_files=max_files)
        self.log(f"found {len(files)} python files")
        self.set_state("idle")
        return files


    def delete_file(self, project_root: Path, rel_path: str) -> None:
        self.set_state("working")
        self.log(f"delete_file {rel_path}")
        p = file_utils.safe_resolve(project_root, rel_path)
        if p.exists():
            p.unlink()
        self.remember(f"delete_file: {rel_path}", tags=["fs"])
        self.set_state("idle")

    def exec(self, cmd: List[str], *, cwd: Path, timeout_s: int = 300) -> ExecResult:
        self.set_state("working")
        self.log(f"exec: {' '.join(cmd)} (cwd={cwd})")
        try:
            res = run_command(cmd, cwd=cwd, timeout_s=timeout_s)
        except CommandNotAllowed as e:
            self.log(f"blocked command: {e}")
            raise
        finally:
            self.set_state("idle")
        self.remember(f"exec: {' '.join(cmd)} rc={res.returncode}", tags=["cmd"], success=(res.returncode == 0))
        return res

    def pip_install(self, packages: List[str], *, cwd: Path, timeout_s: int = 900) -> ExecResult:
        """Install one or more pip packages into the current environment.

        Uses `python -m pip install` so it works reliably inside a venv.
        """
        pkgs = [str(p).strip() for p in (packages or []) if str(p).strip()]
        if not pkgs:
            return ExecResult(returncode=0, stdout="", stderr="")

        cmd = ["python", "-m", "pip", "install", "--no-input", "--disable-pip-version-check", *pkgs]
        return self.exec(cmd, cwd=cwd, timeout_s=timeout_s)

    def pip_install_requirements(self, requirements_file: Path, *, cwd: Path, timeout_s: int = 900) -> ExecResult:
        """Install from a requirements.txt file."""
        rf = Path(requirements_file)
        cmd = ["python", "-m", "pip", "install", "--no-input", "--disable-pip-version-check", "-r", str(rf)]
        return self.exec(cmd, cwd=cwd, timeout_s=timeout_s)

    def append_requirements(self, requirements_file: Path, packages: List[str]) -> List[str]:
        """Append packages to a requirements.txt file (idempotent)."""
        from app.utils.dependency_utils import append_requirements

        return append_requirements(Path(requirements_file), packages)

