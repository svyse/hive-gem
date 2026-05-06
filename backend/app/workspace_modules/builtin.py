from __future__ import annotations

from typing import Dict

from app.workspace_modules.manager import WorkspaceModuleManager


CALCULATOR_RUN_PY = r'''"""Calculator module (safe expression eval).

Usage:
  python run.py --expr "2+2*10"

This intentionally supports only a safe subset of Python expressions:
- numbers (int/float)
- +, -, *, /, //, %, **
- parentheses
"""

import argparse
import ast
import operator as op


_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError(f"Unsupported expression: {type(node).__name__}")


def safe_eval(expr: str):
    tree = ast.parse(expr, mode="eval")
    return _eval(tree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr", "-e", default=None, help="Expression to evaluate")
    args = ap.parse_args()

    expr = args.expr
    if not expr:
        expr = input("expr> ").strip()

    if not expr:
        print("No expression provided")
        return 2

    try:
        res = safe_eval(expr)
    except Exception as e:
        print(f"error: {e}")
        return 1

    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


SCRIPT_RUNNER_RUN_PY = r'''"""Script Runner module.

This module is meant as a simple, reusable utility that can run a python
script located anywhere under the WORKSPACE_ROOT, especially under:
  <WORKSPACE_ROOT>/runs/<run_id>/...

Examples:
  python run.py --script "runs/abc123/project/snake.py"
  python run.py --latest --pattern "snake*.py"

Notes:
- This runner is intentionally minimal. If you need a richer runner (venv,
  args passthrough, etc.) extend it.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def _workspace_root() -> Path:
    # WORKSPACE_ROOT is set by the backend .env; fallback to a local .workspace
    # when running this module directly from a terminal.
    env = os.environ.get("WORKSPACE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / ".workspace").resolve()


def _find_latest_run(root: Path) -> Path | None:
    runs = root / "runs"
    if not runs.exists():
        return None
    dirs = [p for p in runs.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def _find_in_dir(d: Path, pattern: str) -> Path | None:
    try:
        matches = list(d.rglob(pattern))
    except Exception:
        matches = []
    matches = [m for m in matches if m.is_file()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=None, help="Path relative to WORKSPACE_ROOT")
    ap.add_argument("--latest", action="store_true", help="Use latest run under WORKSPACE_ROOT/runs")
    ap.add_argument("--pattern", default="snake*.py", help="Glob pattern used with --latest")
    ap.add_argument("--", dest="passthrough", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    root = _workspace_root()
    script_rel = args.script

    script_path: Path | None = None
    if script_rel:
        script_path = (root / script_rel).resolve()
    elif args.latest:
        latest = _find_latest_run(root)
        if latest is not None:
            script_path = _find_in_dir(latest, args.pattern)

    if script_path is None or not script_path.exists():
        print("Could not find script to run.")
        print("Try: python run.py --script \"runs/<run_id>/project/<your_script>.py\"")
        print("Or:  python run.py --latest --pattern \"snake*.py\"")
        return 2

    cmd = [sys.executable, str(script_path)]
    if args.passthrough:
        cmd.extend(args.passthrough)

    print(f"Running: {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, cwd=str(script_path.parent))
        return int(p.returncode)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
'''


def ensure_builtin_modules(manager: WorkspaceModuleManager | None = None) -> None:
    """Create a small set of built-in workspace modules if missing."""

    mgr = manager or WorkspaceModuleManager()

    existing = {m.name for m in mgr.list_modules()}

    if "calculator" not in existing:
        mgr.create_module(
            name="calculator",
            description="Safe arithmetic calculator (CLI)",
            entrypoint="run.py",
            files={"run.py": CALCULATOR_RUN_PY, "README.md": "# calculator\n\nSafe CLI calculator.\n"},
            overwrite=False,
        )

    if "script_runner" not in existing:
        mgr.create_module(
            name="script_runner",
            description="Run a Python script from WORKSPACE_ROOT (useful for running games/tools in runs)",
            entrypoint="run.py",
            files={"run.py": SCRIPT_RUNNER_RUN_PY, "README.md": "# script_runner\n\nRuns a python script under WORKSPACE_ROOT.\n"},
            overwrite=False,
        )
