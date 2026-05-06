from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent


class LoggingAgent(BaseAgent):
    """Logging-only agent.

    This agent subscribes to selected MessageBus topics and appends received events
    into a run-scoped text file.

    Primary intent: capture *everything* (logs, Q&A events, broadcasts) in a durable
    artifact per run.

    Notes:
    - It performs no LLM calls.
    - It should be spawned at the start of a run.
    """

    agent_type = "logging"

    def __init__(self, *, agent_id: str, ctx):
        super().__init__(agent_id=agent_id, ctx=ctx)
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def start(self, *, log_file: Path, topics: Optional[List[str]] = None) -> None:
        """Start background logging."""
        if self._task is not None and not self._task.done():
            return

        topics = topics or ["logline", "qa", "trace", "broadcast"]
        log_file.parent.mkdir(parents=True, exist_ok=True)

        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop(log_file=log_file, topics=topics))
        self.set_state("logging")
        self.log(f"logging started -> {log_file}")

    async def _run_loop(self, *, log_file: Path, topics: List[str]) -> None:
        queues: Dict[str, asyncio.Queue] = {}
        pending: Dict[asyncio.Task, str] = {}

        # Subscribe
        for t in topics:
            try:
                q = await self.ctx.bus.subscribe(t)
                queues[t] = q
                pending[asyncio.create_task(q.get())] = t
            except Exception as e:
                # Best-effort; don't crash if subscribe fails.
                self.ctx.run_logger(f"logging:{self.agent_id} | failed to subscribe topic={t}: {e}")

        # aiofiles is convenient but optional. Import lazily and fall back to
        # synchronous file writes if aiofiles isn't installed.
        try:
            import aiofiles  # type: ignore

            async with aiofiles.open(log_file, "a", encoding="utf-8") as f:  # type: ignore
                await f.write(f"--- logging_agent started (run_id={self.ctx.run_id}) ---\n")
                await f.flush()

                try:
                    while not self._stop.is_set():
                        if not pending:
                            await asyncio.sleep(0.2)
                            continue

                        done, _ = await asyncio.wait(
                            list(pending.keys()),
                            timeout=0.5,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if not done:
                            continue

                        for task in done:
                            topic = pending.pop(task, "unknown")
                            try:
                                msg = task.result()
                            except Exception as e:
                                await f.write(f"[error] topic={topic} task_error={e}\n")
                                continue

                            # Re-arm
                            if topic in queues:
                                pending[asyncio.create_task(queues[topic].get())] = topic

                            payload = getattr(msg, "payload", msg)
                            line = self._format_line(topic=topic, payload=payload)
                            await f.write(line + "\n")

                        await f.flush()
                finally:
                    await f.write("--- logging_agent stopped ---\n")
                    await f.flush()

        except ModuleNotFoundError:
            # Fallback mode (no aiofiles). This may block briefly on disk IO,
            # but keeps the system functional even in minimal installs.
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"--- logging_agent started (run_id={self.ctx.run_id}) ---\n")
                f.flush()
                try:
                    while not self._stop.is_set():
                        if not pending:
                            await asyncio.sleep(0.2)
                            continue

                        done, _ = await asyncio.wait(
                            list(pending.keys()),
                            timeout=0.5,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if not done:
                            continue

                        for task in done:
                            topic = pending.pop(task, "unknown")
                            try:
                                msg = task.result()
                            except Exception as e:
                                f.write(f"[error] topic={topic} task_error={e}\n")
                                continue

                            # Re-arm
                            if topic in queues:
                                pending[asyncio.create_task(queues[topic].get())] = topic

                            payload = getattr(msg, "payload", msg)
                            line = self._format_line(topic=topic, payload=payload)
                            f.write(line + "\n")

                        f.flush()
                finally:
                    f.write("--- logging_agent stopped ---\n")
                    f.flush()

        # Unsubscribe
        for t, q in queues.items():
            try:
                await self.ctx.bus.unsubscribe(t, q)
            except Exception:
                pass

        # Cancel any pending queue.get() tasks
        for task in list(pending.keys()):
            try:
                task.cancel()
            except Exception:
                pass

    def _format_line(self, *, topic: str, payload: Any) -> str:
        try:
            if isinstance(payload, str):
                body = payload
            else:
                body = json.dumps(payload, ensure_ascii=False)
        except Exception:
            body = str(payload)
        return f"[{topic}] {body}"

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except Exception:
                try:
                    self._task.cancel()
                except Exception:
                    pass
        self.set_state("idle")

    async def terminate(self) -> None:
        try:
            await self.stop()
        except Exception:
            pass
        await super().terminate()
