from __future__ import annotations

import json
import re
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.core.config import settings
from app.workspace_modules.builtin import ensure_builtin_modules
from app.workspace_modules.manager import WorkspaceModuleManager


_KW_MODULE = ["module", "modules", "workspace tool", "workspace", "tooling", "tool"]


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


def _parse_run_intent(q: str) -> Optional[Dict[str, Any]]:
    """Very small parser for: run module <name> [args...]"""

    # Examples:
    # - "run module calculator --expr 2+2"
    # - "run calculator --expr 2+2"
    s = (q or "").strip()
    m = re.search(r"\b(run|execute)\b\s+(module\s+)?(?P<name>[a-zA-Z0-9_-]+)(?P<rest>.*)$", s, re.I)
    if not m:
        return None
    name = (m.group("name") or "").strip()
    rest = (m.group("rest") or "").strip()
    args = []
    if rest:
        # naive split; users can also use the API/UI for complex quoting
        args = [p for p in rest.split() if p]
    return {"name": name, "args": args}


def _parse_create_intent(q: str) -> Optional[Dict[str, Any]]:
    # Examples:
    # - "create module calculator"
    # - "create a module named my_tool"
    s = (q or "").strip()
    if not re.search(r"\b(create|make|build|scaffold|generate)\b", s, re.I):
        return None
    if not re.search(r"\b(module|tool)\b", s, re.I):
        return None

    m = re.search(r"\bnamed\b\s+(?P<name>[a-zA-Z0-9_-]+)", s, re.I)
    if not m:
        m = re.search(r"\bmodule\b\s+(?P<name>[a-zA-Z0-9_-]+)", s, re.I)

    name = (m.group("name") if m else "").strip() if m else ""
    return {"name": name or None}


MODULE_BUILDER_SYSTEM = """You create small Python utilities as workspace modules.

Return VALID JSON ONLY (no markdown fences):
{
  "name": "snake_case_or_kebab-case",
  "description": "short description",
  "entrypoint": "run.py",
  "code": "<python code for run.py>",
  "usage": "one-line usage string",
  "requirements": ["optional pip requirement spec strings"]
}

Constraints:
- Prefer stdlib only; if you truly need third-party packages, list them in `requirements`.
- Requirements must be simple PyPI package specs (no URLs, no editable installs).
- Provide a CLI using argparse
- Do not access network by default
- Do not delete files
"""


class WorkspaceModuleAgent(BaseAgent):
    """Agent that can create/list/run modules under WORKSPACE_ROOT/modules."""

    agent_type = "workspace_module"

    async def answer(self, *, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.set_state("answering")
        self.log("workspace modules")

        mgr = WorkspaceModuleManager()

        # Ensure built-ins exist so the user has something immediately usable.
        try:
            ensure_builtin_modules(mgr)
        except Exception:
            pass

        q = question or ""

        try:
            # --- list ---
            if re.search(r"\b(list|show)\b", q, re.I) and re.search(r"\bmodules?\b", q, re.I):
                mods = [m.to_dict() for m in mgr.list_modules()]
                lines = [f"- {m['name']}: {m.get('description','')}" for m in mods]
                out = {
                    "answer": "Workspace modules available:\n" + ("\n".join(lines) if lines else "(none)"),
                    "key_points": [
                        "Modules live under WORKSPACE_ROOT/modules",
                        "Run a module via: POST /api/modules/{name}/run or CLI: python -m app.workspace_modules.cli run <name> -- <args>",
                    ],
                    "sources_used": [],
                    "followups": [
                        "Run module calculator --expr 2+2",
                        "Create module named my_tool that does X",
                    ],
                    "confidence": "high",
                }
                self.latest_result = out
                return out

            # --- run ---
            run_intent = _parse_run_intent(q)
            if run_intent and run_intent.get("name"):
                name = str(run_intent["name"])
                args = [str(x) for x in (run_intent.get("args") or [])]
                try:
                    # run_command uses subprocess.run, which is blocking.
                    # Offload to a thread to keep the FastAPI event loop responsive.
                    res = await asyncio.to_thread(mgr.run_module, name=name, args=args)
                    stdout = (res.stdout or "")[-8000:]
                    stderr = (res.stderr or "")[-8000:]
                    out = {
                        "answer": f"Module '{name}' finished with code {res.returncode}.\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}",
                        "key_points": [f"returncode={res.returncode}", "Backend execution uses the CommandAgent allowlist (python)."],
                        "sources_used": [],
                        "followups": [
                            "List modules",
                            f"Run module {name} (with different args)",
                        ],
                        "confidence": "high" if res.returncode == 0 else "medium",
                    }
                    self.latest_result = out
                    return out
                except Exception as e:
                    out = {
                        "answer": f"Failed to run module '{name}': {e}",
                        "key_points": [],
                        "sources_used": [],
                        "followups": ["List modules", "Create module named calculator"],
                        "confidence": "low",
                    }
                    self.latest_result = out
                    return out

            # --- create ---
            create_intent = _parse_create_intent(q)
            if create_intent is not None:
                # Template shortcuts
                if "calculator" in q.lower():
                    ensure_builtin_modules(mgr)
                    out = {
                        "answer": "Created/verified built-in module: calculator.\n\nRun it with: run module calculator --expr 2+2",
                        "key_points": ["Module path: WORKSPACE_ROOT/modules/calculator"],
                        "sources_used": [],
                        "followups": ["Run module calculator --expr (1+2)*3", "List modules"],
                        "confidence": "high",
                    }
                    self.latest_result = out
                    return out

                if "snake" in q.lower():
                    ensure_builtin_modules(mgr)
                    out = {
                        "answer": (
                            "Created/verified built-in module: script_runner.\n\n"
                            "This can run a snake game (or any script) you created under WORKSPACE_ROOT/runs.\n\n"
                            "Example: run module script_runner --script runs/<run_id>/project/snake.py\n"
                            "Or:      run module script_runner --latest --pattern snake*.py"
                        ),
                        "key_points": ["Module path: WORKSPACE_ROOT/modules/script_runner"],
                        "sources_used": [],
                        "followups": ["List modules"],
                        "confidence": "high",
                    }
                    self.latest_result = out
                    return out

                # LLM-backed scaffolding for custom tools
                if not self.llm_available():
                    out = {
                        "answer": (
                            "I can create custom workspace modules, but the LLM backend is unavailable. "
                            "Set LLM_BACKEND=local (or configure OpenAI) and retry, or create via POST /api/modules."
                        ),
                        "key_points": ["You can always create a module via the /api/modules endpoint."],
                        "sources_used": [],
                        "followups": ["Create module named my_tool that prints hello"],
                        "confidence": "low",
                    }
                    self.latest_result = out
                    return out

                # Ask LLM for module scaffold
                payload = {
                    "user_request": q,
                    "existing_modules": [m.to_dict() for m in mgr.list_modules()][:25],
                    "constraints": [
                        "Keep it dependency-free (stdlib)",
                        "Implement argparse CLI",
                        "Write everything into a single run.py",
                    ],
                }
                spec = await self.ctx.llm.chat_json_async(system=MODULE_BUILDER_SYSTEM, messages=[{"role": "user", "content": json.dumps(payload, indent=2)}], temperature=0.2)
                if not isinstance(spec, dict):
                    raise ValueError("module builder returned non-object")

                name = str(spec.get("name") or "").strip() or (create_intent.get("name") or "custom_tool")
                desc = str(spec.get("description") or "")
                code = str(spec.get("code") or "")
                usage = str(spec.get("usage") or "")
                reqs = spec.get("requirements") or []
                if not isinstance(reqs, list):
                    reqs = []
                reqs = [str(x) for x in reqs if str(x).strip()]

                if not code.strip():
                    raise ValueError("module builder returned empty code")

                mgr.create_module(
                    name=name,
                    description=desc,
                    entrypoint="run.py",
                    files={"run.py": code, "README.md": f"# {name}\n\n{desc}\n\nUsage: {usage}\n"},
                    requirements=reqs,
                    overwrite=True,
                )

                # Record a small code-learning example so the local LoRA worker
                # can gradually learn how you like CLI tools structured.
                try:
                    if getattr(settings, "code_training_enabled", True):
                        max_chars = int(getattr(settings, "code_training_max_chars", 12000) or 12000)
                        if len(code) <= max_chars:
                            prompt_txt = (
                                f"Create a CLI workspace module named '{name}'.\n\n"
                                f"User request: {q.strip()}\n\n"
                                "Write the complete contents of run.py. "
                                "Return ONLY the file contents (no markdown, no explanation)."
                            )
                            self.ctx.memory_store.add_training_example(
                                source="workspace_module",
                                prompt=prompt_txt,
                                completion=code,
                                created_at=datetime.now(timezone.utc).isoformat(),
                            )
                except Exception:
                    pass

                out = {
                    "answer": f"Created workspace module '{name}'.\n\nUsage: {usage or f'run module {name} -- --help'}",
                    "key_points": [f"Module stored under WORKSPACE_ROOT/modules/{name}", "To run: run module <name> ..."],
                    "sources_used": [],
                    "followups": [f"Run module {name} -- --help", "List modules"],
                    "confidence": "high",
                }
                self.latest_result = out
                # Store a memory so other agents can discover it.
                try:
                    self.add_hive_memory(
                        json.dumps({"event": "workspace_module_created", "name": name, "description": desc, "usage": usage}, ensure_ascii=False),
                        tags=["workspace_module", name],
                        success=True,
                    )
                except Exception:
                    pass
                return out

            # Default: explain capability
            mods = [m.to_dict() for m in mgr.list_modules()][:15]
            out = {
                "answer": (
                    "I can manage Workspace Modules stored under WORKSPACE_ROOT/modules.\n\n"
                    "Try one of these commands:\n"
                    "- list modules\n"
                    "- create module calculator\n"
                    "- run module calculator --expr 2+2\n"
                    "- create module named my_tool that ..."
                ),
                "key_points": [f"Detected {len(mods)} module(s)"],
                "sources_used": [],
                "followups": ["List modules"],
                "confidence": "medium" if _contains_any(q, _KW_MODULE) else "low",
            }
            self.latest_result = out
            return out

        finally:
            try:
                self.set_state("idle")
            except Exception:
                pass
