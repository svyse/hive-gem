from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.llm.prompts import MODULE_PLAN_SYSTEM
from app.utils.file_utils import list_files, read_text
from app.utils.repetition_guard import sanitize_operations
from app.utils.local_code_fallbacks import (
    build_known_python_project_plan,
    maybe_simple_python_plan_fallback,
    plan_has_meaningful_operations,
    plan_looks_like_repetition_noop,
)
from app.utils.nlp_code_planner import (
    NLP_CODE_PLAN_SYSTEM,
    build_generic_python_scaffold_plan,
    build_plan_from_nlp_output,
    env_nlp_planner_enabled,
    is_local_backend,
)
from app.utils.local_code_engine import build_local_code_engine_plan, env_code_engine_enabled


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...<truncated>...\n"


def _allow_legacy_local_nlp_fallback() -> bool:
    import os

    raw = os.getenv("LOCAL_CODE_ENGINE_ALLOW_LEGACY_NLP_FALLBACK", "0").strip().lower()
    return raw in {"1", "true", "yes", "on", "y"}


class ModuleAgent(BaseAgent):
    agent_type = "module"

    async def make_plan(self, *, project_root: Path, user_prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.set_state("planning")
        self.log("building change plan")

        # The local code engine below is the primary path for all NLP project
        # prompts. Demo-specific deterministic templates are deliberately not
        # used here by default, so requests such as calculator, hello/name, games,
        # APIs, websites, etc. all flow through the same prompt-driven engine.

        py_files = list_files(project_root, exts=[".py"], max_files=200)
        # Prefer top-level and smaller contexts; sample up to 12 files
        sample_files = py_files[:12]

        snapshot: List[Dict[str, str]] = []
        for fp in sample_files:
            try:
                content = read_text(project_root, fp)
            except Exception:
                continue
            snapshot.append({"path": fp, "content": _truncate(content, 3500)})

        hive_recent = self.ctx.memory_store.recent(scope="hive", limit=1)

        if not self.llm_available():
            plan = build_known_python_project_plan(user_prompt, project_root=project_root, reason="LLM unavailable") or {
                "summary": "LLM unavailable. Created a placeholder plan file.",
                "operations": [
                    {
                        "op": "write_file",
                        "path": "AGENTIC_PLAN.md",
                        "content": (
                            "# Agentic Plan Placeholder\n\n"
                            "The ModuleAgent could not call the configured LLM.\n\n"
                            f"## Prompt\n{user_prompt}\n"
                        ),
                    }
                ],
                "test_commands": [],
                "notes": "Configure backend/.env or use a supported local fallback prompt.",
            }
            self.latest_result = plan
            self.remember(json.dumps(plan), tags=["plan", "module"], success=None)
            self.set_state("idle")
            return plan

        # Use type/hive memory to prime successful patterns
        mem_bundle = self.get_memory_bundle("plan", include_type=True, include_hive=True, include_agent=False, limit_per_scope=6)
        memory_context = self.format_memory_bundle(mem_bundle)

        # General local code-engine path.  This is the main local coding mode:
        # ask for a compact project manifest, then ask for each file's complete
        # content using normal text prompts.  It supports new project creation as
        # well as editing/debugging existing local files without requiring JSON.
        if env_code_engine_enabled() and is_local_backend(self.ctx.llm) and hasattr(self.ctx.llm, "chat_text_async"):
            try:
                engine_plan = await build_local_code_engine_plan(
                    llm=self.ctx.llm,
                    project_root=project_root,
                    user_prompt=user_prompt,
                    context=context,
                    log=self.log,
                    phase="plan",
                )
                if engine_plan is not None and plan_has_meaningful_operations(engine_plan):
                    self.latest_result = engine_plan
                    self.remember(json.dumps(engine_plan), tags=["plan", "module", "local_code_engine"], success=None)
                    self.add_type_memory(json.dumps(engine_plan), tags=["plan", "module", "local_code_engine"], success=None)
                    try:
                        lesson = ((engine_plan.get("metadata") or {}).get("quality_lesson") or {})
                        if lesson:
                            self.add_type_memory(
                                json.dumps(lesson, ensure_ascii=False),
                                tags=["code_generation_lesson", "local_code_engine", "quality_gate"],
                                success=None,
                            )
                    except Exception:
                        pass
                    self.set_state("idle")
                    return engine_plan
                self.log("local code engine returned no usable operations after strict validation")
                if not _allow_legacy_local_nlp_fallback():
                    fallback = build_generic_python_scaffold_plan(
                        user_prompt,
                        reason="local code engine returned no safe files; legacy NLP/JSON fallbacks disabled to avoid writing hallucinated project files",
                    )
                    self.latest_result = fallback
                    self.remember(json.dumps(fallback), tags=["plan", "module", "local_code_engine", "safety_scaffold"], success=None)
                    self.add_type_memory(json.dumps(fallback), tags=["plan", "module", "local_code_engine", "safety_scaffold"], success=None)
                    self.set_state("idle")
                    return fallback
                self.log("legacy one-shot local NLP fallback is enabled; falling back to file-block planner")
            except Exception as e:
                self.log(f"local code engine failed: {type(e).__name__}: {e}")
                if not _allow_legacy_local_nlp_fallback():
                    fallback = build_generic_python_scaffold_plan(
                        user_prompt,
                        reason=f"local code engine error: {type(e).__name__}: {e}; legacy NLP/JSON fallbacks disabled to avoid writing hallucinated project files",
                    )
                    self.latest_result = fallback
                    self.remember(json.dumps(fallback), tags=["plan", "module", "local_code_engine", "safety_scaffold"], success=None)
                    self.add_type_memory(json.dumps(fallback), tags=["plan", "module", "local_code_engine", "safety_scaffold"], success=None)
                    self.set_state("idle")
                    return fallback
                self.log("legacy one-shot local NLP fallback is enabled; falling back to file-block planner")

        # Prefer an NLP/file-block planner for the local backend.  This is the
        # important compatibility path: Q&A-capable local models often produce
        # useful code blocks but fail strict JSON.  We parse those code blocks
        # into normal write_file operations before falling back to JSON.
        if (not env_code_engine_enabled() or _allow_legacy_local_nlp_fallback()) and env_nlp_planner_enabled() and is_local_backend(self.ctx.llm) and hasattr(self.ctx.llm, "chat_text_async"):
            nlp_messages = [
                {
                    "role": "user",
                    "content": (
                        "Create the requested project from this prompt.\n\n"
                        f"USER_PROMPT:\n{user_prompt}\n\n"
                        f"PROJECT_ROOT:\n{project_root}\n\n"
                        f"EXISTING_PYTHON_FILES:\n{py_files[:40]}\n\n"
                        "Return FILE blocks only. Do not return JSON. Do not explain. Make the project runnable."
                    ),
                }
            ]
            try:
                self.log("using local NLP file-block planner before strict JSON planner")
                raw_nlp = await self.ctx.llm.chat_text_async(
                    system=NLP_CODE_PLAN_SYSTEM,
                    messages=nlp_messages,
                    temperature=0.25,
                    purpose="code-nlp-plan",
                )
                nlp_plan = build_plan_from_nlp_output(
                    raw_nlp,
                    user_prompt=user_prompt,
                    reason="local backend NLP file-block planner succeeded",
                )
                if nlp_plan is not None and plan_has_meaningful_operations(nlp_plan):
                    self.latest_result = nlp_plan
                    self.remember(json.dumps(nlp_plan), tags=["plan", "module", "local_nlp"], success=None)
                    self.add_type_memory(json.dumps(nlp_plan), tags=["plan", "module", "local_nlp"], success=None)
                    self.set_state("idle")
                    return nlp_plan
                self.log("local NLP planner returned no parseable file blocks; falling back to JSON planner")
            except Exception as e:
                self.log(f"local NLP planner failed; falling back to JSON planner: {type(e).__name__}: {e}")

        # Ask LLM for a structured plan
        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_prompt": user_prompt,
                        "project_root": str(project_root),
                        "python_files": py_files,
                        "snapshot": snapshot,
                        "hive_recent": hive_recent,
                        "memory_context": memory_context,
                        "external_context": context or {},
                        "required_output_schema": {
                            "summary": "string",
                            "operations": [
                                {
                                    "op": "mkdir|write_file|delete_file",
                                    "path": "relative/path",
                                    "content": "string (required for write_file)",
                                }
                            ],
                            "test_commands": ["string"],
                            "notes": "string",
                        },
                        "constraints": [
                            "Paths must be relative to project_root.",
                            "Prefer minimal changes and small, testable modules.",
                            "Do not include markdown code fences in JSON.",
                        ],
                    },
                    indent=2,
                ),
            }
        ]

        try:
            plan = await self.ctx.llm.chat_json_async(
                system=MODULE_PLAN_SYSTEM, messages=messages, temperature=0.2, purpose="code-json-plan"
            )
        except Exception as e:
            fallback = build_known_python_project_plan(user_prompt, project_root=project_root, reason=f"planner error: {type(e).__name__}: {e}")
            if fallback is None:
                fallback = build_generic_python_scaffold_plan(user_prompt, reason=f"planner error: {type(e).__name__}: {e}")
            plan = fallback

        # Minimal validation/sanitization
        if not isinstance(plan, dict):
            fallback = build_known_python_project_plan(user_prompt, project_root=project_root, reason="planner returned a non-object response")
            if fallback is None:
                fallback = build_generic_python_scaffold_plan(user_prompt, reason="planner returned a non-object response")
            plan = fallback

        plan.setdefault("operations", [])
        plan.setdefault("test_commands", [])
        plan.setdefault("summary", "")
        plan.setdefault("notes", "")
        plan["operations"], dropped_ops = sanitize_operations(plan.get("operations") or [])
        if dropped_ops:
            plan["notes"] = (str(plan.get("notes") or "") + f"\nDropped {dropped_ops} unsafe repeated-token operation(s).").strip()

        fallback = maybe_simple_python_plan_fallback(
            user_prompt=user_prompt,
            plan=plan,
            reason="planner returned no safe operations or repeated-token fallback",
            project_root=project_root,
            extra_signal_text=str(plan.get("notes") or ""),
        )
        if fallback is not None:
            if plan_looks_like_repetition_noop(plan):
                self.log("local planner repeated tokens; using deterministic Python fallback")
            elif not plan_has_meaningful_operations(plan):
                self.log("planner produced no safe operations for a known Python request; using deterministic fallback")
            plan = fallback

        if is_local_backend(self.ctx.llm) and not plan_has_meaningful_operations(plan):
            self.log("local planner produced no safe operations; creating generic prompt-based Python scaffold")
            plan = build_generic_python_scaffold_plan(
                user_prompt,
                reason="strict JSON planner produced no safe operations",
            )

        self.latest_result = plan
        self.remember(json.dumps(plan), tags=["plan", "module"], success=None)
        self.add_type_memory(json.dumps(plan), tags=["plan", "module"], success=None)

        self.set_state("idle")
        return plan

    async def make_repair_plan(
        self,
        *,
        project_root: Path,
        user_prompt: str,
        error_text: str,
        phase: str,
        prior_plan: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ask the local code engine to edit files after compile/test failures.

        This keeps debug/fix work in the same code-pipeline mode: the local
        model sees the current project files plus the exact error text and
        returns replacement file contents, not JSON.
        """

        self.set_state("repairing")
        self.log(f"building local repair plan for {phase} errors")

        if not (env_code_engine_enabled() and is_local_backend(self.ctx.llm) and hasattr(self.ctx.llm, "chat_text_async")):
            repair = {
                "summary": "Local code engine repair is unavailable.",
                "operations": [],
                "test_commands": [],
                "notes": "Repair skipped because the active backend is not the local code engine path.",
                "metadata": {"local_code_engine_plan": False, "phase": "repair"},
            }
            self.latest_result = repair
            self.set_state("idle")
            return repair

        try:
            repair = await build_local_code_engine_plan(
                llm=self.ctx.llm,
                project_root=project_root,
                user_prompt=user_prompt,
                context=context,
                log=self.log,
                mode="edit_debug",
                error_text=error_text,
                phase="repair",
                prior_plan=prior_plan,
            )
        except Exception as e:
            repair = None
            self.log(f"local repair planner failed: {type(e).__name__}: {e}")

        if repair is None:
            repair = {
                "summary": "Local code engine could not produce a repair plan.",
                "operations": [],
                "test_commands": [],
                "notes": "Repair skipped because no safe file operations were generated.",
                "metadata": {"local_code_engine_plan": True, "phase": "repair", "empty_repair": True},
            }

        self.latest_result = repair
        self.remember(json.dumps(repair), tags=["plan", "module", "local_code_engine", "repair", phase], success=None)
        self.add_type_memory(json.dumps(repair), tags=["plan", "module", "local_code_engine", "repair", phase], success=None)
        self.set_state("idle")
        return repair
