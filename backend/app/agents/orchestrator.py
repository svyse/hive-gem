from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.agents.base import BaseAgent
from app.core.config import settings
from app.utils.file_utils import safe_resolve
from app.utils.repetition_guard import sanitize_operation_content, sanitize_operations
from app.utils.local_code_fallbacks import (
    maybe_simple_python_plan_fallback,
    plan_has_meaningful_operations,
    plan_looks_like_repetition_noop,
)
from app.utils.nlp_code_planner import build_generic_python_scaffold_plan, is_local_backend
from app.utils.local_code_quality import (
    build_quality_lesson_payload,
    filter_operations_for_quality,
    short_quality_issue_summary,
    validate_generated_file_content,
    validate_generated_path_for_prompt,
)
from app.utils.sequence_learning import (
    QueryTraceRecorder,
    extract_sequence_traces_from_memories,
    rank_similar_traces,
    recommend_most_common_sequence,
)


WEB_QUERY_SYSTEM = """You are an orchestrator assistant that generates web-search queries.

Goal: help the hive collect external information (docs, APIs, examples) relevant to implementing
the user's request in a codebase.

Return JSON only:
{
  "queries": ["..."],
  "notes": "..."
}

Constraints:
- 1 to 8 queries
- Avoid duplicates
- Prefer vendor / official docs queries when possible
- No markdown code fences
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunk(items: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _uniq(seq: List[str]) -> List[str]:
    out: List[str] = []
    for x in seq:
        x = (x or "").strip()
        if not x:
            continue
        if x not in out:
            out.append(x)
    return out


class OrchestratorAgent(BaseAgent):
    """Code pipeline orchestrator.

    This orchestrator is used in "code" mode (software-factory style runs).

    New in this version:
    - Records per-run *sequence traces* of agent calls + memory accesses.
    - Performs lightweight sequence-learning: looks up similar past traces and
      exposes them in the context passed to downstream agents.
    """

    agent_type = "orchestrator"

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------

    def _start_trace(self, *, user_prompt: str) -> None:
        try:
            self._trace_recorder = QueryTraceRecorder(
                orchestrator=self.agent_type,
                run_id=str(getattr(self.ctx, "run_id", "")),
                query=user_prompt,
                domain="Code pipeline",
                conversation_id=None,
            )
            self._trace_event("query_start", prompt=user_prompt)
        except Exception:
            self._trace_recorder = None

    def _trace_event(self, kind: str, **data: Any) -> None:
        rec = getattr(self, "_trace_recorder", None)
        if rec is None:
            return
        try:
            rec.event(kind, **(data or {}))
        except Exception:
            return

    async def _commit_trace(self, *, success: Optional[bool], extra: Optional[Dict[str, Any]] = None) -> None:
        rec = getattr(self, "_trace_recorder", None)
        if rec is None:
            return

        try:
            payload = rec.to_dict(finished_at=_now_iso(), success=success, extra=extra)
        except Exception:
            return

        tags = ["sequence_trace", self.agent_type, "code_pipeline"]

        type_memory_id: Optional[int] = None
        hive_memory_id: Optional[int] = None

        # Persist the trace into both the orchestrator-type memory and hive memory.
        # Capture the inserted ids so the UI can look them up quickly.
        try:
            type_memory_id = self.add_type_memory(json.dumps(payload, ensure_ascii=False), tags=tags, success=success)
        except Exception:
            type_memory_id = None

        try:
            hive_memory_id = self.add_hive_memory(json.dumps(payload, ensure_ascii=False), tags=tags, success=success)
        except Exception:
            hive_memory_id = None

        # Link this trace to the *run* for the code-pipeline trace viewer.
        try:
            run_id = str(getattr(self.ctx, "run_id", "") or "").strip()
            if run_id:
                self.ctx.memory_store.add_run_trace(
                    run_id=run_id,
                    orchestrator=self.agent_type,
                    created_at=_now_iso(),
                    hive_memory_id=hive_memory_id,
                    type_memory_id=type_memory_id,
                )
        except Exception:
            pass

        try:
            await self.ctx.bus.publish("trace", payload)
        except Exception:
            pass

    def _sequence_learning_context(self, *, user_prompt: str) -> Dict[str, Any]:
        if not getattr(settings, "sequence_learning_enabled", True):
            return {"enabled": False}

        scan_limit = int(getattr(settings, "sequence_learning_scan_limit", 250) or 250)
        max_traces = int(getattr(settings, "sequence_learning_max_traces", 6) or 6)
        min_sim = float(getattr(settings, "sequence_learning_min_similarity", 0.08) or 0.08)

        try:
            self._trace_event(
                "memory_access",
                op="recent_all",
                scope="type,hive",
                limit=scan_limit,
                purpose="sequence_learning_scan",
            )
            recent = self.ctx.memory_store.recent_all(limit=scan_limit, scopes=["type", "hive"])
            parsed = extract_sequence_traces_from_memories(recent)
        except Exception as e:
            return {"enabled": False, "error": str(e)}

        try:
            similar = rank_similar_traces(
                query=user_prompt,
                traces=parsed,
                max_traces=max_traces,
                min_similarity=min_sim,
                orchestrator=self.agent_type,
            )
            sequences: List[List[str]] = []
            summaries: List[Dict[str, Any]] = []
            for t in similar:
                sequences.append([str(x) for x in (t.get("agent_sequence") or []) if isinstance(x, str) and x.strip()])
                summaries.append(
                    {
                        "query": t.get("query"),
                        "signature": t.get("signature"),
                        "success": t.get("success"),
                        "finished_at": t.get("finished_at") or t.get("_memory_created_at"),
                    }
                )
            recommended = recommend_most_common_sequence(sequences)
            self._trace_event("sequence_learning", enabled=True, similar_traces=len(similar), recommended_sequence=recommended)
            return {"enabled": True, "similar_traces": summaries, "recommended_sequence": recommended}
        except Exception as e:
            return {"enabled": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------



    async def _auto_install_missing_deps(
        self,
        *,
        project_root: Path,
        command_agent,
        error_text: str,
        phase: str,
    ) -> Dict[str, Any]:
        """Try to resolve missing Python deps by pip-installing them and recording requirements.

        This is a best-effort helper used in the *code pipeline* when py_compile/tests fail due to
        ModuleNotFoundError. It is controlled by AUTO_INSTALL_DEPS / PIP_INSTALL_TIMEOUT_S.
        """
        if not getattr(settings, "auto_install_deps", True):
            return {"enabled": False, "phase": phase, "installed": []}

        blob = error_text or ""
        try:
            from app.utils.dependency_utils import append_requirements, extract_missing_modules, module_to_pip_package, workspace_requirements_path
        except Exception as e:
            return {"enabled": False, "phase": phase, "installed": [], "error": str(e)}

        missing_mods = extract_missing_modules(blob)
        pkgs: List[str] = []
        for m in missing_mods:
            p = module_to_pip_package(m)
            if p and p not in pkgs:
                pkgs.append(p)

        if not pkgs:
            return {"enabled": True, "phase": phase, "installed": []}

        # Record requirements (both per-project and workspace-global) so work is reproducible.
        proj_req = Path(project_root) / "requirements.txt"
        ws_req = workspace_requirements_path()

        added_project = []
        added_workspace = []
        try:
            added_project = append_requirements(proj_req, pkgs)
        except Exception:
            added_project = []
        try:
            added_workspace = append_requirements(ws_req, pkgs)
        except Exception:
            added_workspace = []

        # Install into current environment.
        install_res = None
        try:
            timeout_s = int(getattr(settings, "pip_install_timeout_s", 600) or 600)
            install_res = command_agent.pip_install(pkgs, cwd=project_root, timeout_s=timeout_s)
        except Exception as e:
            return {
                "enabled": True,
                "phase": phase,
                "installed": pkgs,
                "requirements_added_project": added_project,
                "requirements_added_workspace": added_workspace,
                "pip_error": str(e),
            }

        return {
            "enabled": True,
            "phase": phase,
            "installed": pkgs,
            "requirements_added_project": added_project,
            "requirements_added_workspace": added_workspace,
            "pip_returncode": getattr(install_res, "returncode", None),
            "pip_stdout": getattr(install_res, "stdout", "")[-4000:],
            "pip_stderr": getattr(install_res, "stderr", "")[-4000:],
        }

    async def run(self, *, project_root: Path, user_prompt: str) -> Dict[str, Any]:
        self.set_state("orchestrating")
        self.log("starting orchestration")

        self._start_trace(user_prompt=user_prompt)
        self._trace_event("query_config", project_root=str(project_root))

        registry = self.ctx.registry
        if registry is None:
            raise RuntimeError("AgentContext.registry is not set")

        # Base crew
        command_agent = await registry.spawn("command")
        module_agent = await registry.spawn("module")
        logic_agent = await registry.spawn("logic")
        testing_agent = await registry.spawn("testing")
        compiling_agent = await registry.spawn("compiling")
        struct_agent = await registry.spawn("complex_compiling")
        docker_agent = await registry.spawn("docker")

        self._trace_event(
            "crew_spawn",
            agents=[
                command_agent.agent_type,
                module_agent.agent_type,
                logic_agent.agent_type,
                testing_agent.agent_type,
                compiling_agent.agent_type,
                struct_agent.agent_type,
                docker_agent.agent_type,
            ],
        )

        # Optional helpers
        local_ref_agent = None
        try:
            local_ref_agent = await registry.spawn("local_reference")
            self._trace_event("agent_spawn", agent_type="local_reference")
        except Exception:
            local_ref_agent = None

        success_flag: Optional[bool] = None
        error: Optional[str] = None
        result: Dict[str, Any] = {}

        try:
            self.log(f"Project root: {project_root}")
            await self.broadcast({"event": "run_started", "prompt": user_prompt})

            # ---- Context gathering (memory + local references + web research) ----
            self._trace_event("agent_call", agent_type=self.agent_type, method="_build_context")
            context = await self._build_context(project_root=project_root, user_prompt=user_prompt, local_ref_agent=local_ref_agent)

            # 1) Plan changes. All normal local coding requests flow through
            # ModuleAgent's prompt-driven local code engine rather than
            # demo-specific hardcoded templates.
            self._trace_event("agent_call", agent_type="module", method="make_plan")
            plan = await module_agent.make_plan(project_root=project_root, user_prompt=user_prompt, context=context)
            await self.ctx.bus.publish("plan", {"plan": plan})
            self.log(f"Plan summary: {plan.get('summary','')}")
            self.remember(json.dumps({"prompt": user_prompt, "plan": plan}), tags=["orchestrator", "plan"])

            # 2) Logic review
            self._trace_event("agent_call", agent_type="logic", method="review_plan")
            try:
                review = await logic_agent.review_plan(project_root=project_root, user_prompt=user_prompt, plan=plan, context=context)
            except Exception as e:
                # Never let a malformed local JSON review crash the code pipeline.
                # The final safety pass below still sanitizes every operation before writing files.
                review = {
                    "approved": True,
                    "issues": [
                        f"Logic review failed ({type(e).__name__}: {e}); continuing with sanitized plan operations."
                    ],
                    "recommended_plan_patch": {"operations": []},
                }
                self.log("Logic review failed; continuing with sanitized plan operations.")
            await self.ctx.bus.publish("logic_review", {"review": review})
            if not review.get("approved", True):
                self.log("LogicAgent did not approve plan; proceeding but recording issues.")
            issues = review.get("issues") or []
            for iss in issues[:10]:
                self.log(f"Logic issue: {iss}")

            # 3) Apply recommended patch operations (if any)
            patch_ops = (review.get("recommended_plan_patch") or {}).get("operations") or []
            if patch_ops:
                plan_ops = plan.get("operations") or []
                plan["operations"] = plan_ops + patch_ops
                self.log(f"Merged {len(patch_ops)} patch operations into plan.")

            # Final safety pass: never apply repeated-token write_file content.
            plan_ops_clean, dropped_plan_ops = sanitize_operations(plan.get("operations") or [])
            plan["operations"] = plan_ops_clean
            if dropped_plan_ops:
                self.log(f"Dropped {dropped_plan_ops} unsafe repeated-token operation(s) before apply.")

            # Generic file-quality gate before writing anything. This blocks the
            # failure mode seen in the attached projects: README prose/shell
            # commands accepted as .py, tests, requirements.txt, or .gitignore.
            quality_filtered_ops, pre_apply_quality_issues = filter_operations_for_quality(
                plan.get("operations") or [],
                prompt=user_prompt,
                project_name=project_root.name,
            )
            if pre_apply_quality_issues:
                plan["operations"] = quality_filtered_ops
                meta = plan.setdefault("metadata", {})
                if isinstance(meta, dict):
                    existing = meta.get("quality_issues") if isinstance(meta.get("quality_issues"), list) else []
                    meta["quality_issues"] = list(existing) + pre_apply_quality_issues[:20]
                    meta["quality_issues_count"] = len(meta["quality_issues"])
                summary = short_quality_issue_summary(pre_apply_quality_issues)
                plan["notes"] = (str(plan.get("notes") or "") + "\nPre-apply quality gate removed invalid generated files: " + summary).strip()
                self.log("Pre-apply quality gate removed invalid generated files: " + summary)

            fallback_plan = maybe_simple_python_plan_fallback(
                user_prompt=user_prompt,
                plan=plan,
                reason="no safe operations remained after planner/review sanitization",
                project_root=project_root,
                extra_signal_text=str(plan.get("notes") or ""),
            )
            if fallback_plan is not None:
                if plan_looks_like_repetition_noop(plan):
                    self.log("Planner returned repeated-token/no-op output; using deterministic Python fallback.")
                elif not plan_has_meaningful_operations(plan):
                    self.log("No safe plan operations remained; using deterministic Python fallback for known request.")
                plan = fallback_plan

            if is_local_backend(self.ctx.llm) and not plan_has_meaningful_operations(plan):
                self.log("No safe operations remained for local code pipeline; creating generic prompt-based Python scaffold.")
                plan = build_generic_python_scaffold_plan(
                    user_prompt,
                    reason="orchestrator safety fallback after planning/review sanitization",
                )

            # 4) Execute file operations
            cleanup_report = self._cleanup_new_project_failed_artifacts(
                project_root=project_root,
                user_prompt=user_prompt,
                plan=plan,
            )
            if cleanup_report.get("deleted"):
                self.log(
                    "Cleaned failed generated artifacts before create-project apply: "
                    + ", ".join(cleanup_report.get("deleted", [])[:8])
                )
            self._trace_event("agent_call", agent_type=self.agent_type, method="_apply_operations")
            applied = await self._apply_operations(project_root, plan.get("operations") or [], command_agent, user_prompt=user_prompt)
            await self.ctx.bus.publish("operations_applied", {"applied": applied})
            self.log(f"Applied operations: {applied['applied_count']}, skipped: {applied['skipped_count']}")
            if applied["errors"]:
                self.log(f"Operation errors: {len(applied['errors'])}")

            # 5) Compile check (py_compile)
            self._trace_event("agent_call", agent_type="compiling", method="compile_check")
            compile_report = await compiling_agent.compile_check(project_root=project_root, command_agent=command_agent)
            await self.ctx.bus.publish("compile_report", {"compile": compile_report})
            if not compile_report.get("ok", False):
                self.log("Compile check found errors.")
                for e in compile_report.get("errors", [])[:5]:
                    self.log(f"Compile error: {e}")

                # Attempt auto dependency install if errors indicate missing modules.
                try:
                    err_blob = ""
                    for e in compile_report.get("errors", [])[:25]:
                        err_blob += "\n" + str(e.get("stderr") or "") + "\n" + str(e.get("stdout") or "")
                    deps_fix_compile = await self._auto_install_missing_deps(
                        project_root=project_root,
                        command_agent=command_agent,
                        error_text=err_blob,
                        phase="compile",
                    )
                    if deps_fix_compile.get("installed"):
                        self.log(f"Auto-installed deps (compile): {deps_fix_compile.get('installed')}")
                        # Re-run compile check after installing deps
                        compile_report = await compiling_agent.compile_check(project_root=project_root, command_agent=command_agent)
                        await self.ctx.bus.publish("compile_report", {"compile": compile_report, "deps_fix": deps_fix_compile})
                except Exception:
                    pass

            # 5b) If compile still fails, let the local code engine edit files once.
            if not compile_report.get("ok", False):
                try:
                    repair_text = self._compile_error_text(compile_report)
                    repair_result = await self._attempt_local_code_repair(
                        phase="compile",
                        project_root=project_root,
                        user_prompt=user_prompt,
                        error_text=repair_text,
                        context=context,
                        plan=plan,
                        module_agent=module_agent,
                        command_agent=command_agent,
                    )
                    if (repair_result.get("applied") or {}).get("applied_count", 0) > 0:
                        compile_report = await compiling_agent.compile_check(project_root=project_root, command_agent=command_agent)
                        await self.ctx.bus.publish("compile_report", {"compile": compile_report, "repair": repair_result})
                        if compile_report.get("ok"):
                            self.log("Compile check passed after local code engine repair.")
                except Exception as e:
                    self.log(f"local code engine compile repair failed: {type(e).__name__}: {e}")

            # 6) Tests
            self._trace_event("agent_call", agent_type="testing", method="ensure_and_run_tests")
            test_report = await testing_agent.ensure_and_run_tests(
                project_root=project_root,
                user_prompt=user_prompt,
                plan=plan,
                command_agent=command_agent,
            )
            await self.ctx.bus.publish("test_report", {"tests": test_report})
            self.log(f"Test summary: {test_report.get('summary')}")

            # If tests failed due to missing deps, auto-install and retry once.
            if getattr(settings, "auto_install_deps", True) and test_report.get("summary") != "passed":
                try:
                    blob = ""
                    for r in (test_report.get("results") or [])[:10]:
                        blob += "\n" + str(r.get("stderr") or "") + "\n" + str(r.get("stdout") or "")
                    deps_fix_tests = await self._auto_install_missing_deps(
                        project_root=project_root,
                        command_agent=command_agent,
                        error_text=blob,
                        phase="tests",
                    )
                    if deps_fix_tests.get("installed"):
                        self.log(f"Auto-installed deps (tests): {deps_fix_tests.get('installed')}")
                        test_report = await testing_agent.ensure_and_run_tests(
                            project_root=project_root,
                            user_prompt=user_prompt,
                            plan=plan,
                            command_agent=command_agent,
                        )
                        await self.ctx.bus.publish("test_report", {"tests": test_report, "deps_fix": deps_fix_tests})
                        self.log(f"Test summary (after deps install): {test_report.get('summary')}")
                except Exception:
                    pass

            # 6b) If tests still fail, let the local code engine edit files once and rerun compile/tests.
            if test_report.get("summary") != "passed":
                try:
                    repair_text = self._test_error_text(test_report)
                    repair_result = await self._attempt_local_code_repair(
                        phase="tests",
                        project_root=project_root,
                        user_prompt=user_prompt,
                        error_text=repair_text,
                        context=context,
                        plan=plan,
                        module_agent=module_agent,
                        command_agent=command_agent,
                    )
                    if (repair_result.get("applied") or {}).get("applied_count", 0) > 0:
                        compile_report = await compiling_agent.compile_check(project_root=project_root, command_agent=command_agent)
                        await self.ctx.bus.publish("compile_report", {"compile": compile_report, "repair": repair_result})
                        test_report = await testing_agent.ensure_and_run_tests(
                            project_root=project_root,
                            user_prompt=user_prompt,
                            plan=plan,
                            command_agent=command_agent,
                        )
                        await self.ctx.bus.publish("test_report", {"tests": test_report, "repair": repair_result})
                        self.log(f"Test summary (after local code engine repair): {test_report.get('summary')}")
                except Exception as e:
                    self.log(f"local code engine test repair failed: {type(e).__name__}: {e}")

            # 7) Project structure assessment
            self._trace_event("agent_call", agent_type="complex_compiling", method="assess_structure")
            structure_report = await struct_agent.assess_structure(project_root=project_root, command_agent=command_agent)
            await self.ctx.bus.publish("structure_report", {"structure": structure_report})

            # 8) Docker assets
            self._trace_event("agent_call", agent_type="docker", method="docker_plan")
            docker_plan = await docker_agent.docker_plan(project_root=project_root, user_prompt=user_prompt)
            docker_ops_clean, dropped_docker_ops = sanitize_operations(docker_plan.get("operations") or [])
            docker_plan["operations"] = docker_ops_clean
            if dropped_docker_ops:
                self.log(f"Dropped {dropped_docker_ops} unsafe repeated-token Docker operation(s) before apply.")
            self._trace_event("agent_call", agent_type=self.agent_type, method="_apply_operations", phase="docker")
            docker_applied = await self._apply_operations(project_root, docker_plan.get("operations") or [], command_agent, user_prompt=user_prompt)
            await self.ctx.bus.publish("docker_plan", {"docker": docker_plan, "applied": docker_applied})

            # Optional: execute docker commands if present and allowed
            docker_cmds = docker_plan.get("docker_commands") or []
            docker_exec_results = []
            for cmd_str in docker_cmds:
                try:
                    import shlex

                    self._trace_event("agent_call", agent_type="command", method="exec", command=cmd_str)
                    res = command_agent.exec(shlex.split(cmd_str), cwd=project_root, timeout_s=900)
                    docker_exec_results.append(
                        {
                            "command": cmd_str,
                            "returncode": res.returncode,
                            "stdout": res.stdout[-4000:],
                            "stderr": res.stderr[-4000:],
                        }
                    )
                except Exception as e:
                    docker_exec_results.append({"command": cmd_str, "error": str(e)})

            result = {
                "prompt": user_prompt,
                "context": context,
                "plan": plan,
                "logic_review": review,
                "applied_operations": applied,
                "compile_report": compile_report,
                "test_report": test_report,
                "structure_report": structure_report,
                "docker_plan": docker_plan,
                "docker_exec_results": docker_exec_results,
                "project_root": str(project_root),
            }

            self.latest_result = result

            # Persist a compact run summary to orchestrator type memory (for learning).
            compact = {
                "prompt": user_prompt,
                "plan_summary": plan.get("summary", ""),
                "notes": plan.get("notes", ""),
                "web_sources": (context.get("web_research") or {}).get("sources"),
                "tests_summary": test_report.get("summary"),
            }
            success_flag = test_report.get("summary") == "passed"
            try:
                self.add_type_memory(json.dumps(compact), tags=["run_summary"], success=success_flag)
            except Exception:
                pass

            # Promote a compact successful memory to hive if tests passed.
            if success_flag:
                self.add_hive_memory(json.dumps(compact), tags=["successful_run"], success=True)

                # Record small code-learning examples so the local LoRA worker can
                # learn from the hive's successful coding work over time.
                try:
                    if getattr(settings, "code_training_enabled", True):
                        ops_all = (plan.get("operations") or []) + (docker_plan.get("operations") or [])
                        rec = self._record_code_training_examples(user_prompt=user_prompt, operations=ops_all)
                        self._trace_event("code_training_examples", **rec)
                except Exception:
                    pass

            try:
                lesson_rec = self._record_code_failure_lesson(
                    user_prompt=user_prompt,
                    project_root=project_root,
                    plan=plan,
                    applied=applied,
                    compile_report=compile_report,
                    test_report=test_report,
                    success=success_flag,
                )
                if lesson_rec.get("recorded"):
                    self._trace_event("code_generation_lesson", **lesson_rec)
            except Exception as e:
                self.log(f"code generation lesson recording skipped: {type(e).__name__}: {e}")

            self._trace_event(
                "query_end",
                success=success_flag,
                compile_ok=compile_report.get("ok"),
                tests_summary=test_report.get("summary"),
                web_sources_count=len((context.get("web_research") or {}).get("sources") or []),
            )

            await self._commit_trace(
                success=success_flag,
                extra={
                    "project_root": str(project_root),
                    "tests_summary": test_report.get("summary"),
                    "compile_ok": compile_report.get("ok"),
                },
            )

            self.log("orchestration done")
            self.set_state("idle")
            return result

        except Exception as e:
            error = str(e)
            success_flag = False
            self.log(f"orchestration error: {e}")
            self._trace_event("error", error=error)
            try:
                await self._commit_trace(success=False, extra={"error": error, "project_root": str(project_root)})
            except Exception:
                pass
            raise

        finally:
            # Terminate worker agents (flush memories + free slots).
            to_term = [docker_agent, struct_agent, compiling_agent, testing_agent, logic_agent, module_agent, command_agent]
            if local_ref_agent is not None:
                to_term.append(local_ref_agent)
            for a in to_term:
                try:
                    await registry.terminate(a.agent_id)
                    self._trace_event("agent_terminate", agent_type=a.agent_type)
                except Exception:
                    pass

    def _compile_error_text(self, compile_report: Dict[str, Any]) -> str:
        parts: List[str] = []
        for e in (compile_report or {}).get("errors", [])[:20]:
            if not isinstance(e, dict):
                parts.append(str(e))
                continue
            if e.get("file"):
                parts.append(f"FILE: {e.get('file')}")
            if e.get("error"):
                parts.append(str(e.get("error")))
            if e.get("stderr"):
                parts.append(str(e.get("stderr")))
            if e.get("stdout"):
                parts.append(str(e.get("stdout")))
        return "\n".join(parts)[-20000:]

    def _test_error_text(self, test_report: Dict[str, Any]) -> str:
        parts: List[str] = []
        for r in (test_report or {}).get("results", [])[:20]:
            if not isinstance(r, dict):
                parts.append(str(r))
                continue
            if r.get("command"):
                parts.append(f"COMMAND: {r.get('command')}")
            if r.get("error"):
                parts.append(str(r.get("error")))
            if r.get("stderr"):
                parts.append(str(r.get("stderr")))
            if r.get("stdout"):
                parts.append(str(r.get("stdout")))
        return "\n".join(parts)[-24000:]

    def _cleanup_new_project_failed_artifacts(
        self,
        *,
        project_root: Path,
        user_prompt: str,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Remove obvious failed local-generation artifacts before a create run.

        This is intentionally conservative.  By default it only runs under a
        sample_projects folder, where a create-project prompt is expected to own
        the folder.  It prevents bad files from previous attempts (for example
        ``python main.py``, ``file_1.java``, invalid ``main.py`` or stale
        ``__pycache__``) from making every later run fail compile before the new
        valid files can help.
        """

        raw_enabled = os.getenv("LOCAL_CODE_ENGINE_CLEAN_NEW_PROJECT_ARTIFACTS", "1").strip().lower()
        if raw_enabled in {"0", "false", "no", "off"}:
            return {"deleted": [], "skipped": [], "reason": "disabled"}

        meta = plan.get("metadata") if isinstance(plan, dict) else {}
        mode = str((meta or {}).get("mode") or "").lower()
        planner_kind = str((meta or {}).get("planner_kind") or "").lower()
        q = str(user_prompt or "").strip().lower()
        looks_create = mode == "create_project" or (
            any(q.startswith(prefix) for prefix in ("create ", "make ", "build ", "generate ", "write "))
            and "debug" not in q
            and "fix" not in q
        )
        if not looks_create:
            return {"deleted": [], "skipped": [], "reason": "not_create_project"}

        parts = {part.lower() for part in project_root.resolve().parts}
        allow_anywhere = os.getenv("LOCAL_CODE_ENGINE_CLEAN_NEW_PROJECT_ANYWHERE", "0").strip().lower() in {"1", "true", "yes", "on"}
        if not allow_anywhere and "sample_projects" not in parts:
            return {"deleted": [], "skipped": [], "reason": "outside_sample_projects"}

        deleted: List[str] = []
        skipped: List[str] = []
        protected = {".memory", ".workspace", ".git", ".venv", "venv", "node_modules"}

        def rel_of(path: Path) -> str:
            return str(path.relative_to(project_root)).replace("\\", "/")

        # Remove cache directories and binary cache artifacts first.
        for p in sorted(project_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            try:
                rel = rel_of(p)
            except Exception:
                continue
            if any(part in protected for part in p.parts):
                continue
            try:
                if p.is_dir() and p.name in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
                    shutil.rmtree(p, ignore_errors=True)
                    deleted.append(rel + "/")
                elif p.is_file() and p.suffix.lower() in {".pyc", ".pyo", ".pyd"}:
                    p.unlink(missing_ok=True)
                    deleted.append(rel)
            except Exception as exc:
                skipped.append(f"{rel}: {exc}")

        # Remove files that fail the same generic quality gate used for new
        # writes.  This clears previous failed generations without hardcoding
        # calculator/hello-world behavior.
        for p in sorted(project_root.rglob("*")):
            if not p.is_file():
                continue
            try:
                rel = rel_of(p)
            except Exception:
                continue
            rel_parts = set(rel.split("/"))
            if rel_parts & protected or "__pycache__" in rel_parts:
                continue
            try:
                if p.stat().st_size > 300_000:
                    continue
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            ok, issues = validate_generated_file_content(rel, text, prompt=user_prompt, project_name=project_root.name)
            if ok:
                continue
            try:
                p.unlink(missing_ok=True)
                deleted.append(rel)
            except Exception as exc:
                skipped.append(f"{rel}: {exc}; issues={issues[:3]}")

        # Remove empty generated directories left behind after deleting files.
        for p in sorted([x for x in project_root.rglob("*") if x.is_dir()], key=lambda item: len(item.parts), reverse=True):
            try:
                rel = rel_of(p)
            except Exception:
                continue
            if any(part in protected for part in p.parts):
                continue
            try:
                if not any(p.iterdir()):
                    p.rmdir()
                    deleted.append(rel + "/")
            except Exception:
                pass

        return {"deleted": deleted[:200], "skipped": skipped[:50], "planner_kind": planner_kind, "mode": mode}

    async def _attempt_local_code_repair(
        self,
        *,
        phase: str,
        project_root: Path,
        user_prompt: str,
        error_text: str,
        context: Dict[str, Any],
        plan: Dict[str, Any],
        module_agent,
        command_agent,
    ) -> Dict[str, Any]:
        if not bool(getattr(settings, "local_code_engine_repair_enabled", True)):
            return {"attempted": False, "reason": "disabled"}
        if not str(error_text or "").strip():
            return {"attempted": False, "reason": "no_error_text"}

        self.log(f"attempting local code engine repair after {phase} failure")
        repair_plan = await module_agent.make_repair_plan(
            project_root=project_root,
            user_prompt=user_prompt,
            error_text=error_text,
            phase=phase,
            prior_plan=plan,
            context=context,
        )
        repair_ops, dropped = sanitize_operations(repair_plan.get("operations") or [])
        repair_plan["operations"] = repair_ops
        if dropped:
            self.log(f"Dropped {dropped} unsafe repeated-token repair operation(s) before apply.")
        repair_ops, repair_quality_issues = filter_operations_for_quality(
            repair_plan.get("operations") or [],
            prompt=user_prompt,
            project_name=project_root.name,
        )
        repair_plan["operations"] = repair_ops
        if repair_quality_issues:
            meta = repair_plan.setdefault("metadata", {})
            if isinstance(meta, dict):
                existing = meta.get("quality_issues") if isinstance(meta.get("quality_issues"), list) else []
                meta["quality_issues"] = list(existing) + repair_quality_issues[:20]
                meta["quality_issues_count"] = len(meta["quality_issues"])
            self.log("Repair quality gate removed invalid generated files: " + short_quality_issue_summary(repair_quality_issues))
        if not repair_ops:
            self.log("local code engine repair produced no safe file operations")
            return {"attempted": True, "applied": {"applied_count": 0, "skipped_count": 0, "errors": []}, "repair_plan": repair_plan}

        applied = await self._apply_operations(project_root, repair_ops, command_agent, user_prompt=user_prompt)
        await self.ctx.bus.publish("operations_applied", {"phase": f"repair_{phase}", "applied": applied, "repair_plan": repair_plan})
        self.log(f"Repair operations applied: {applied['applied_count']}, skipped: {applied['skipped_count']}")

        # Keep the final run result/training examples aware of repair edits.
        plan.setdefault("operations", [])
        plan["operations"] = (plan.get("operations") or []) + repair_ops
        if repair_plan.get("test_commands"):
            plan["test_commands"] = repair_plan.get("test_commands") or plan.get("test_commands") or []
        return {"attempted": True, "applied": applied, "repair_plan": repair_plan}

    def _record_code_failure_lesson(
        self,
        *,
        user_prompt: str,
        project_root: Path,
        plan: Dict[str, Any],
        applied: Dict[str, Any],
        compile_report: Dict[str, Any],
        test_report: Dict[str, Any],
        success: bool,
    ) -> Dict[str, Any]:
        """Store compact negative/quality lessons in .memory for future local runs.

        This does not fine-tune immediately and it does not train on bad file
        contents. It records what went wrong so the next code-engine prompt can
        retrieve the lesson from memory, and the trainer can continue to use
        successful prompt->file examples only.
        """

        if not bool(getattr(settings, "local_code_learning_lessons_enabled", True)):
            return {"recorded": False, "reason": "disabled"}

        meta = plan.get("metadata") if isinstance(plan, dict) else {}
        quality_issues = []
        if isinstance(meta, dict) and isinstance(meta.get("quality_issues"), list):
            quality_issues = list(meta.get("quality_issues") or [])

        compile_ok = bool((compile_report or {}).get("ok", False))
        tests_passed = (test_report or {}).get("summary") == "passed"
        applied_errors = (applied or {}).get("errors") or []
        should_record = bool(quality_issues or applied_errors or not compile_ok or not tests_passed or not success)
        if not should_record:
            return {"recorded": False, "reason": "no_failure_or_quality_issue"}

        compile_summary = self._compile_error_text(compile_report) if not compile_ok else "compile passed"
        test_summary = self._test_error_text(test_report) if not tests_passed else "tests passed"
        if applied_errors:
            quality_issues.append({"path": "[apply_operations]", "issues": [str(applied_errors)[:1200]]})

        payload = build_quality_lesson_payload(
            prompt=user_prompt,
            project_name=project_root.name,
            quality_issues=quality_issues,
            compile_summary=compile_summary,
            test_summary=test_summary,
        )
        payload.update(
            {
                "success": bool(success),
                "compile_ok": compile_ok,
                "tests_summary": (test_report or {}).get("summary"),
                "plan_summary": str((plan or {}).get("summary") or "")[:1200],
                "plan_notes": str((plan or {}).get("notes") or "")[:2000],
            }
        )

        # Add explicit general lessons for validation failures even if the quality
        # issue list is empty but compile/tests failed.
        if not payload.get("lessons"):
            lessons: List[str] = []
            if not compile_ok:
                lessons.append("Generated source files must pass syntax/compile checks before being considered complete.")
            if not tests_passed:
                lessons.append("Generated projects should include safe non-interactive tests or smoke checks that validate the requested behavior.")
            payload["lessons"] = lessons

        content = json.dumps(payload, ensure_ascii=False)[: int(getattr(settings, "local_code_failure_max_chars", 12000) or 12000)]
        tags = ["code_generation_lesson", "local_code_engine", "negative_example", "code_pipeline"]
        recorded = 0
        try:
            self.add_type_memory(content, tags=tags, success=False)
            recorded += 1
        except Exception:
            pass
        try:
            self.add_hive_memory(content, tags=tags, success=False)
            recorded += 1
        except Exception:
            pass
        if recorded:
            self.log("Recorded code-generation lesson for future local runs.")
        return {"recorded": bool(recorded), "records": recorded, "quality_issues": len(quality_issues)}

    def _record_code_training_examples(self, *, user_prompt: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Persist prompt->file-content examples from successful runs.

        These examples are stored in the training_examples table and are consumed
        by the background LoRA worker. This helps the local model improve at
        coding based on what the hive actually built.
        """

        max_examples = int(getattr(settings, "code_training_max_examples_per_run", 8) or 8)
        max_chars = int(getattr(settings, "code_training_max_chars", 12000) or 12000)

        recorded = 0
        skipped = 0
        for op in operations:
            if recorded >= max_examples:
                break
            if not isinstance(op, dict):
                skipped += 1
                continue

            if op.get("op") != "write_file":
                continue

            rel = str(op.get("path") or "").strip()
            content = op.get("content")
            if not rel or not isinstance(content, str) or not content.strip():
                skipped += 1
                continue
            if len(content) > max_chars:
                skipped += 1
                continue

            prompt = (
                "Task: "
                + str(user_prompt).strip()
                + "\n\n"
                + f"Write the complete contents of the file `{rel}`. "
                + "Return ONLY the file contents (no markdown fences, no explanation)."
            )

            try:
                self.ctx.memory_store.add_training_example(
                    source="code_write_file",
                    prompt=prompt,
                    completion=content,
                    created_at=_now_iso(),
                )
                recorded += 1
            except Exception:
                skipped += 1

        return {"recorded": recorded, "skipped": skipped, "max_examples": max_examples}

    async def _build_context(self, *, project_root: Path, user_prompt: str, local_ref_agent) -> Dict[str, Any]:
        """Collect context to help downstream agents.

        Context includes:
        - relevant memories (hive + all type memories)
        - optional local reference snippets from other projects
        - optional web research summaries (can be sharded to sub-orchestrators)
        - sequence-learning hints (similar past orchestration traces)
        """
        self.set_state("gathering_context")
        self.log("gathering context")

        query = user_prompt.strip().lower()
        simple_request = len(query.split()) <= 15

        # Orchestrator has access to all memories. Even for short prompts, keep
        # code-generation lessons in context so the local assistant can learn
        # from previous mistakes instead of repeating them.
        memory_query = ("code_generation_lesson local_code_engine quality_gate " + user_prompt).strip()
        memory_relevant = self.ctx.memory_store.search_all(
            query=memory_query,
            limit=6 if simple_request else 8,
            scopes=["hive", "type", "agent"],
        )
        self._trace_event(
            "memory_access",
            op="search_all",
            scope="hive,type,agent",
            query=memory_query,
            limit=6 if simple_request else 8,
            result_count=len(memory_relevant),
        )

        memory_recent = self.ctx.memory_store.recent_all(limit=3 if simple_request else 5, scopes=["hive", "type", "agent"])
        self._trace_event("memory_access", op="recent_all", scope="hive,type,agent", limit=3 if simple_request else 5, result_count=len(memory_recent))

        # Do not return early for short prompts. Most local-code prompts are
        # short (for example, "create a calculator"), and those are exactly the
        # runs that need memory lessons and optional web research context.
        local_refs: Dict[str, Any] | None = None
        if (not simple_request) and local_ref_agent is not None:
            try:
                self._trace_event("agent_call", agent_type="local_reference", method="find_references")
                local_refs = await local_ref_agent.find_references(query=user_prompt, project_root=project_root)
            except Exception as e:
                local_refs = {"error": str(e)}

        self._trace_event("agent_call", agent_type=self.agent_type, method="_collect_web_research")
        try:
            web_research = await self._collect_web_research(project_root=project_root, user_prompt=user_prompt)
        except Exception as e:
            # Optional context: never let network/search failures stop local code generation.
            web_research = {"enabled": False, "error": str(e), "queries": [], "items": [], "sources": []}
            self.log(f"web research skipped/failed: {type(e).__name__}: {e}")

        seq_ctx = {} #self._sequence_learning_context(user_prompt=user_prompt)

        rag_context: Dict[str, Any] | None = None
        try:
            if bool(getattr(settings, "rag_enabled", True)):
                from app.rag import build_rag_context

                rag_context = build_rag_context(
                    query=user_prompt,
                    memory_store=self.ctx.memory_store,
                    project_root=project_root,
                    mode="code",
                )
                stats = (rag_context or {}).get("stats") if isinstance(rag_context, dict) else {}
                self._trace_event(
                    "memory_access",
                    op="rag",
                    scope="hive,type,agent,project",
                    query=user_prompt,
                    result_count=int((stats or {}).get("memory_matches", 0) or 0) + int((stats or {}).get("project_file_matches", 0) or 0),
                )
        except Exception as e:
            rag_context = {"enabled": False, "error": str(e)}

        ctx = {
            "memory_relevant": memory_relevant,
            "memory_recent": memory_recent,
            "local_references": local_refs,
            "web_research": web_research,
            "sequence_learning": seq_ctx,
            "rag": rag_context,
        }

        # Include workspace modules so the code-pipeline can optionally
        # generate helpers that live outside the repo (under WORKSPACE_ROOT).
        try:
            from app.workspace_modules.manager import WorkspaceModuleManager

            mgr = WorkspaceModuleManager()
            ctx["workspace_modules"] = [m.to_dict() for m in mgr.list_modules()][:50]
        except Exception:
            ctx["workspace_modules"] = []

        self.set_state("idle")
        return ctx

    async def _collect_web_research(self, *, project_root: Path, user_prompt: str) -> Dict[str, Any]:
        if not settings.web_research_enabled:
            return {"enabled": False, "queries": [], "items": [], "sources": []}

        llm_client = getattr(getattr(self, "ctx", None), "llm", None)
        active_backend = str(getattr(llm_client, "backend", getattr(settings, "llm_backend", "local")) or "local").lower()
        if active_backend == "local" and not bool(getattr(settings, "local_code_web_research_enabled", False)):
            return {
                "enabled": False,
                "skipped": "disabled_for_local_backend",
                "queries": [],
                "items": [],
                "sources": [],
            }

        registry = self.ctx.registry
        if registry is None:
            return {"enabled": False, "error": "registry not set", "queries": [], "items": [], "sources": []}

        queries = await self._generate_web_queries(project_root=project_root, user_prompt=user_prompt)
        queries = _uniq(queries)
        if active_backend == "local":
            try:
                max_queries = max(1, int(getattr(settings, "local_code_web_research_max_queries", 3)))
            except Exception:
                max_queries = 3
            queries = queries[:max_queries]

        if not queries:
            return {"enabled": True, "queries": [], "items": [], "sources": []}

        self._trace_event("web_research", queries_count=len(queries), backend=active_backend)

        # If query count is high, try to shard to sub-orchestrators (self-scaling).
        items: List[Dict[str, Any]] = []
        assistants: List[Any] = []

        try:
            if len(queries) > settings.orchestrator_scale_threshold and getattr(registry, "max_orchestrators", 1) > 1:
                batches = _chunk(queries, max(1, settings.orchestrator_query_batch_size))
                primary_batch = batches[0]
                other_batches = batches[1:]

                self.log(f"web research: {len(queries)} queries -> {len(batches)} batches (attempt scaling)")

                # Spawn assistant orchestrators
                for _ in range(min(len(other_batches), max(0, int(getattr(registry, "max_orchestrators", 1)) - 1))):
                    try:
                        orch = await registry.spawn("orchestrator")
                        assistants.append(orch)
                        self._trace_event("orchestrator_spawn", orchestrator_type="orchestrator")
                    except Exception:
                        break

                tasks: List[asyncio.Task] = []
                tasks.append(asyncio.create_task(self.assist_web_research(project_root=project_root, queries=primary_batch)))

                for orch, batch in zip(assistants, other_batches[: len(assistants)]):
                    self._trace_event("orchestrator_consult", orchestrator_type="orchestrator", batch_size=len(batch))
                    tasks.append(asyncio.create_task(orch.assist_web_research(project_root=project_root, queries=batch)))

                leftover = other_batches[len(assistants) :]
                for batch in leftover:
                    tasks.append(asyncio.create_task(self.assist_web_research(project_root=project_root, queries=batch)))

                parts = await asyncio.gather(*tasks)
                for p in parts:
                    items.extend((p.get("items") or []))

            else:
                self.log(f"web research: {len(queries)} queries (no scaling)")
                timeout_s = float(getattr(settings, "local_code_web_research_timeout_s", 25.0)) if active_backend == "local" else float(getattr(settings, "resource_call_timeout_s", 25.0))
                try:
                    part = await asyncio.wait_for(
                        self.assist_web_research(project_root=project_root, queries=queries),
                        timeout=max(3.0, timeout_s),
                    )
                except asyncio.TimeoutError:
                    return {"enabled": False, "error": "web_research_timeout", "queries": queries, "items": [], "sources": []}
                items = part.get("items") or []
        finally:
            # ensure assistant orchestrators get terminated
            for orch in assistants:
                try:
                    await registry.terminate(orch.agent_id)
                    self._trace_event("agent_terminate", agent_type="orchestrator")
                except Exception:
                    pass

        # Collect sources
        sources: List[str] = []
        for it in items:
            for r in it.get("results") or []:
                u = r.get("url")
                if u and u not in sources:
                    sources.append(u)

        return {"enabled": True, "queries": queries, "items": items, "sources": sources[:25]}

    def _fallback_code_web_queries(self, *, project_root: Path, user_prompt: str) -> List[str]:
        """Deterministic web queries for local code mode.

        Local models in this project have produced malformed JSON and repeated
        token loops, so local-code web-query generation deliberately avoids an
        LLM call. These queries are prompt-shaped and bias toward docs/examples.
        """
        q = re.sub(r"\s+", " ", str(user_prompt or "").strip())
        q_short = q[:160]
        lower = f"{project_root.name} {q}".lower()

        language = "python"
        if any(x in lower for x in ("react", "vite", "tsx", "typescript")):
            language = "typescript react"
        elif any(x in lower for x in ("javascript", "node", "express", "npm", "html", "css", "website", "web app")):
            language = "javascript"
        elif any(x in lower for x in ("fastapi", "flask", "django", "pytest", "unittest", "python", "_py")):
            language = "python"
        elif any(x in lower for x in ("java", "spring")):
            language = "java"
        elif "rust" in lower:
            language = "rust"
        elif re.search(r"(?:^|\W)go(?:\W|$)|golang", lower):
            language = "go"

        queries: List[str] = []
        if q_short:
            queries.append(f"{language} official docs example {q_short}")
            queries.append(f"{language} project structure tests best practices {q_short}")

        # Add framework-specific official-docs queries when the prompt names one.
        framework_queries = [
            ("fastapi", "FastAPI official docs tutorial project structure testing"),
            ("flask", "Flask official docs tutorial testing application structure"),
            ("django", "Django official docs tutorial testing project layout"),
            ("react", "React official docs forms components Vite project structure"),
            ("vite", "Vite official docs create project React TypeScript"),
            ("express", "Express official docs routing middleware testing"),
            ("pytest", "pytest official docs getting started tests"),
            ("unittest", "Python unittest official docs examples"),
        ]
        for key, query in framework_queries:
            if key in lower:
                queries.insert(0, query)

        if not queries:
            queries.append(f"{language} official documentation examples")
        return queries

    async def _generate_web_queries(self, *, project_root: Path, user_prompt: str) -> List[str]:
        llm_client = getattr(getattr(self, "ctx", None), "llm", None)
        active_backend = str(getattr(llm_client, "backend", getattr(settings, "llm_backend", "local")) or "local").lower()
        if active_backend == "local":
            return self._fallback_code_web_queries(project_root=project_root, user_prompt=user_prompt)

        # If we cannot call a hosted LLM, use deterministic prompt-shaped queries.
        if not self.llm_available():
            return self._fallback_code_web_queries(project_root=project_root, user_prompt=user_prompt)

        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_prompt": user_prompt,
                        "project_root": str(project_root),
                        "hints": {
                            "language": "python",
                            "focus": [
                                "agent orchestration patterns",
                                "multi-agent web research / crawling",
                                "memory store patterns (sqlite / vector / retrieval)",
                                "load balancing / spawning orchestrators",
                            ],
                        },
                    },
                    indent=2,
                ),
            }
        ]
        try:
            out = await self.ctx.llm.chat_json_async(system=WEB_QUERY_SYSTEM, messages=messages, temperature=0.2)
            if isinstance(out, dict):
                qs = out.get("queries") or []
                if isinstance(qs, list):
                    return [str(x) for x in qs if str(x).strip()]
        except Exception as e:
            self.log(f"web query generation failed; using deterministic code queries: {e}")
        return self._fallback_code_web_queries(project_root=project_root, user_prompt=user_prompt)

    async def assist_web_research(self, *, project_root: Path, queries: List[str]) -> Dict[str, Any]:
        """An orchestrator 'worker mode' that only runs web research queries.

        This is used by the primary orchestrator to scale out when there are many queries.
        """
        registry = self.ctx.registry
        if registry is None:
            return {"items": [], "error": "registry not set"}

        self.set_state("assisting_web")
        self.log(f"assist_web_research: {len(queries)} queries")

        # Spawn a pool of web_research agents
        web_agents: List[Any] = []
        pool_target = min(int(settings.web_research_pool_size), max(1, len(queries)))
        for _ in range(pool_target):
            try:
                a = await registry.spawn("web_research")
                web_agents.append(a)
                self._trace_event("agent_spawn", agent_type="web_research")
            except Exception:
                break

        if not web_agents:
            self.log("no web_research agents available (limit reached?)")
            self.set_state("idle")
            return {"items": [], "error": "no web_research agents available"}

        try:
            tasks: List[asyncio.Task] = []
            for i, q in enumerate(queries):
                agent = web_agents[i % len(web_agents)]
                self._trace_event("agent_call", agent_type="web_research", method="research", query=q)
                tasks.append(asyncio.create_task(agent.research(query=q)))
            results = await asyncio.gather(*tasks)
            return {"queries": queries, "items": results}
        finally:
            for a in web_agents:
                try:
                    await registry.terminate(a.agent_id)
                    self._trace_event("agent_terminate", agent_type="web_research")
                except Exception:
                    pass
            self.set_state("idle")

    async def _apply_operations(
        self,
        project_root: Path,
        operations: List[Dict[str, Any]],
        command_agent,
        *,
        user_prompt: str = "",
    ) -> Dict[str, Any]:
        applied = 0
        skipped = 0
        errors: List[Dict[str, Any]] = []
        project_name = project_root.name

        for op in operations:
            try:
                if not isinstance(op, dict):
                    skipped += 1
                    continue
                kind = op.get("op")
                path = str(op.get("path") or "")
                if not kind or not path:
                    skipped += 1
                    continue

                # Validate path is within root and is not a hallucinated command/cache path.
                _ = safe_resolve(project_root, path)
                ok_path, path_issues = validate_generated_path_for_prompt(path, prompt=user_prompt, project_name=project_name)
                if not ok_path:
                    skipped += 1
                    errors.append({"op": {"op": kind, "path": path}, "error": "unsafe generated path: " + "; ".join(path_issues[:5])})
                    continue

                if kind == "mkdir":
                    command_agent.mkdir(project_root, path)
                    applied += 1
                elif kind == "write_file":
                    content = op.get("content")
                    if not isinstance(content, str):
                        raise ValueError("write_file requires string content")
                    content = sanitize_operation_content(content)
                    ok_file, file_issues = validate_generated_file_content(path, content, prompt=user_prompt, project_name=project_name)
                    if not ok_file:
                        skipped += 1
                        errors.append({"op": {"op": kind, "path": path}, "error": "unsafe generated file content: " + "; ".join(file_issues[:5])})
                        continue
                    command_agent.write_file(project_root, path, content)
                    applied += 1
                elif kind == "delete_file":
                    command_agent.delete_file(project_root, path)
                    applied += 1
                else:
                    skipped += 1
            except Exception as e:
                errors.append({"op": op, "error": str(e)})
        return {"applied_count": applied, "skipped_count": skipped, "errors": errors}
