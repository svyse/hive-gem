from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.config import settings


class CommandNotAllowed(RuntimeError):
    pass


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str


def run_command(
    cmd: List[str],
    *,
    cwd: Path,
    timeout_s: int = 300,
    env: Optional[Dict[str, str]] = None,
) -> ExecResult:
    if not cmd:
        raise ValueError("cmd must not be empty")

    base = cmd[0]
    if base not in settings.command_allowlist:
        raise CommandNotAllowed(f"Command not allowlisted: {base}. Allowlist={settings.command_allowlist}")

    if base == "docker" and settings.docker_exec_disabled:
        raise CommandNotAllowed("Docker execution disabled by DOCKER_EXEC_DISABLED=true. You can still generate Dockerfiles.")

    cwd = cwd.resolve()
    # Soft safety: ensure cwd exists.
    if not cwd.exists():
        raise FileNotFoundError(f"cwd not found: {cwd}")

    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=proc_env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return ExecResult(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
