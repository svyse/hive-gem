from __future__ import annotations

"""CLI for workspace modules.

Examples:
  # List modules
  python -m app.workspace_modules.cli list

  # Run calculator
  python -m app.workspace_modules.cli run calculator -- --expr "2+2*10"

  # Run a script from the latest run folder
  python -m app.workspace_modules.cli run script_runner -- --latest --pattern "snake*.py"
"""

import argparse
import json

from app.workspace_modules.builtin import ensure_builtin_modules
from app.workspace_modules.manager import WorkspaceModuleManager


def cmd_list(mgr: WorkspaceModuleManager) -> int:
    mods = [m.to_dict() for m in mgr.list_modules()]
    print(json.dumps({"modules": mods}, indent=2, ensure_ascii=False))
    return 0


def cmd_run(mgr: WorkspaceModuleManager, name: str, args: list[str]) -> int:
    res = mgr.run_module(name=name, args=args)
    if res.stdout:
        print(res.stdout, end="" if res.stdout.endswith("\n") else "\n")
    if res.stderr:
        print(res.stderr, end="" if res.stderr.endswith("\n") else "\n")
    return int(res.returncode)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List available modules")

    runp = sub.add_parser("run", help="Run a module")
    runp.add_argument("name")
    runp.add_argument("--", dest="args", nargs=argparse.REMAINDER)

    ns = ap.parse_args(argv)

    mgr = WorkspaceModuleManager()
    ensure_builtin_modules(mgr)

    if ns.cmd == "list":
        return cmd_list(mgr)
    if ns.cmd == "run":
        return cmd_run(mgr, ns.name, list(ns.args or []))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
