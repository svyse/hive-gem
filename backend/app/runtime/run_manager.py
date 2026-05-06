from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.api.schemas import AgentStatus, RunStatusResponse
from app.runtime.workspace import Workspace, create_workspace
from app.runtime.bus import MessageBus
from app.agents.registry import AgentRegistry
from app.memory.store import get_memory_store


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunRecord:
    run_id: str
    created_at: str
    updated_at: str
    status: str
    project_root: str
    logs: List[str] = field(default_factory=list)
    agent_statuses: Dict[str, AgentStatus] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None
    input_mode: str = "text"


class RunManager:
    def __init__(self) -> None:
        self._runs: Dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def start_run(self, project_path: str, prompt: str, copy_project_to_workspace: bool, input_mode: str = "text") -> str:
        run_id = uuid.uuid4().hex[:12]

        ws = create_workspace(run_id, project_path, copy_project_to_workspace)

        record = RunRecord(
            run_id=run_id,
            created_at=_now_iso(),
            updated_at=_now_iso(),
            status="running",
            project_root=str(ws.project_root),
            input_mode=input_mode or "text",
        )

        async with self._lock:
            self._runs[run_id] = record

        bus = MessageBus()
        memory = get_memory_store()

        # Persist the prompt into hive memory immediately (supports "voice" input as well).
        try:
            memory.add(
                scope="hive",
                content=json.dumps({"event": "run_prompt", "run_id": run_id, "prompt": prompt, "input_mode": record.input_mode}),
                tags=["prompt", record.input_mode],
                success=None,
                created_at=_now_iso(),
            )
        except Exception:
            pass

        run_logger = self._make_logger(run_id, bus)
        registry = AgentRegistry(
            bus=bus,
            memory_store=memory,
            run_id=run_id,
            run_logger=run_logger,
            status_reporter=lambda st: self.update_agent_status(run_id, st),
        )

        # Start orchestrator pipeline as a background task (within the server process).
        record.task = asyncio.create_task(self._run_pipeline(record, ws, prompt, registry))
        return run_id

    def _make_logger(self, run_id: str, bus: MessageBus):
        def log(line: str) -> None:
            # called from async and sync code
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            msg = f"[{ts}] {line}"
            rec = self._runs.get(run_id)
            if rec:
                rec.logs.append(msg)
                rec.updated_at = _now_iso()
                # keep logs bounded
                if len(rec.logs) > 2000:
                    rec.logs = rec.logs[-2000:]

            # Emit to bus so the LoggingAgent can persist a full text log.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(bus.publish("logline", msg))
            except RuntimeError:
                # No running loop (rare in this server)
                pass
            except Exception:
                pass

        return log

    async def _run_pipeline(self, record: RunRecord, ws: Workspace, prompt: str, registry: AgentRegistry) -> None:
        log = registry.run_logger
        logging_agent = None
        try:
            # Start logging agent FIRST, so it captures everything.
            try:
                logging_agent = await registry.spawn("logging")
                await logging_agent.start(log_file=Path(ws.run_root) / "hive.log")
            except Exception as e:
                log(f"logging agent unavailable: {e}")

            log(f"Workspace created: {ws.run_root}")
            log(f"Project root: {ws.project_root}")

            orchestrator = await registry.spawn("orchestrator")
            await orchestrator.run(project_root=ws.project_root, user_prompt=prompt)

            result = orchestrator.latest_result or {}
            record.result = result
            record.status = "succeeded"
            record.updated_at = _now_iso()
            log("Run finished successfully.")
        except Exception as e:
            record.status = "failed"
            record.error = f"{type(e).__name__}: {e}"
            record.updated_at = _now_iso()
            log("Run failed.")
            log(record.error)
            log(traceback.format_exc())
        finally:
            # Ensure all agents flush memory on shutdown
            try:
                await registry.shutdown_all()
            except Exception:
                log("Failed to shutdown registry cleanly.")
                log(traceback.format_exc())

    def update_agent_status(self, run_id: str, status: AgentStatus) -> None:
        rec = self._runs.get(run_id)
        if not rec:
            return
        rec.agent_statuses[status.agent_id] = status
        rec.updated_at = _now_iso()

    def get_status(self, run_id: str) -> Optional[RunStatusResponse]:
        rec = self._runs.get(run_id)
        if not rec:
            return None

        return RunStatusResponse(
            run_id=rec.run_id,
            status=rec.status,
            created_at=rec.created_at,
            updated_at=rec.updated_at,
            project_root=rec.project_root,
            logs=rec.logs,
            agent_statuses=list(rec.agent_statuses.values()),
            result=rec.result,
            error=rec.error,
        )


_RUN_MANAGER: Optional[RunManager] = None


def get_run_manager() -> RunManager:
    global _RUN_MANAGER
    if _RUN_MANAGER is None:
        _RUN_MANAGER = RunManager()
    return _RUN_MANAGER
