from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.registry import AgentRegistry
from app.core.config import settings
from app.memory.store import get_memory_store
from app.runtime.bus import MessageBus
from app.runtime.project_paths import resolve_project_path
from app.llm.factory import active_backend

import traceback


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_project_root(project_path: Optional[str]) -> Optional[Path]:
    if not project_path:
        return None

    try:
        p = resolve_project_path(project_path, create_if_missing=False, prefer_sample_projects=True)
        return p if p.exists() and p.is_dir() else None
    except Exception:
        return None


_ERRORISH_ANSWER_PHRASES = (
    "internal error in",
    "an internal error occurred",
    "internal error while answering",
    "please check the run log for details",
    "could not parse json from model output",
    "q&a agent output must be json object",
    "repeated-token loop",
    "repeated token",
    "generation collapsed",
    "generated repeated-token output",
    "skipped generated operations because the local model repeated tokens",
)


def _looks_like_error_answer(text: Any) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return True
    return any(p in lowered for p in _ERRORISH_ANSWER_PHRASES)


@dataclass
class QARunResult:
    conversation_id: str
    turn_id: str
    started_at: str
    finished_at: str
    log_file: str
    result: Dict[str, Any]
    logs: List[str]
    messages: List[Dict[str, Any]]


async def run_qa(
    *,
    question: str,
    orchestrator: Optional[str] = None,
    project_path: Optional[str] = None,
    use_web: bool = True,
    use_local_refs: bool = True,
    input_mode: str = "text",
    # Multi-turn chat
    conversation_id: Optional[str] = None,
    conversation_title: Optional[str] = None,
    conversation_max_messages: int = 50,
) -> QARunResult:
    """Run a Q&A request through the HiveMasterOrchestrator.

    Multi-turn support:
    - If conversation_id is provided, the turn is appended to that conversation.
    - If conversation_id is omitted, a new conversation is created.

    Logging:
    - A LoggingAgent appends logs into a conversation-scoped file:
      <WORKSPACE_ROOT>/qa_conversations/<conversation_id>/hive.log

    Memory:
    - Each user turn and assistant answer is committed to hive memory
      (tagged with conversation_id + input_mode).
    """

    started_at = _now_iso()
    turn_id = f"turn-{uuid.uuid4().hex[:12]}"

    bus = MessageBus()
    memory = get_memory_store()

    # Create or load conversation
    if not conversation_id:
        conversation_id = f"conv-{uuid.uuid4().hex[:12]}"
        try:
            memory.create_conversation(
                conversation_id=conversation_id,
                orchestrator=orchestrator,
                title=conversation_title,
                metadata={"kind": "qa"},
                created_at=_now_iso(),
            )
        except Exception:
            pass
    else:
        # Ensure it exists
        try:
            existing = memory.get_conversation(conversation_id=conversation_id)
            if existing is None:
                memory.create_conversation(
                    conversation_id=conversation_id,
                    orchestrator=orchestrator,
                    title=conversation_title,
                    metadata={"kind": "qa"},
                    created_at=_now_iso(),
                )
        except Exception:
            pass

    # Conversation directory lives under workspace_root.
    # On some setups (Windows permissions / unusual working dirs) this can fail,
    # which previously surfaced as a 500 and the frontend showed "unable to fetch".
    # We fall back to a local .workspace folder under the current working dir.
    try:
        conv_dir = Path(settings.workspace_root) / "qa_conversations" / conversation_id
        conv_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        conv_dir = Path.cwd() / ".workspace" / "qa_conversations" / conversation_id
        conv_dir.mkdir(parents=True, exist_ok=True)

    log_file = conv_dir / "hive.log"

    # Resolve project_root if provided
    project_root = _resolve_project_root(project_path)

    logs: List[str] = []

    def status_reporter(_st):
        return None

    def run_logger(line: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        msg = f"[{ts}] {line}"
        logs.append(msg)
        if len(logs) > 2500:
            del logs[: len(logs) - 2500]

        # publish to bus so LoggingAgent can write
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(bus.publish("logline", msg))
        except Exception:
            pass

    # Persist user message into conversation store first (so orchestrators can see it)
    try:
        memory.add_conversation_message(
            conversation_id=conversation_id,
            role="user",
            content=question,
            orchestrator=orchestrator,
            input_mode=input_mode,
            meta={"turn_id": turn_id, "project_path": project_path},
            created_at=_now_iso(),
        )
        memory.update_conversation(
            conversation_id=conversation_id,
            updated_at=_now_iso(),
            metadata_patch={"last_turn_id": turn_id, "last_input_mode": input_mode or "text"},
        )
    except Exception:
        pass

    # Persist user message to hive memory
    try:
        memory.add(
            scope="hive",
            content=json.dumps(
                {
                    "event": "conversation_message",
                    "conversation_id": conversation_id,
                    "turn_id": turn_id,
                    "role": "user",
                    "orchestrator": orchestrator,
                    "question": question,
                    "input_mode": input_mode,
                    "project_path": project_path,
                },
                ensure_ascii=False,
            ),
            tags=["conversation", "qa", conversation_id, "user", input_mode or "text"],
            success=None,
            created_at=_now_iso(),
        )
    except Exception:
        pass

    registry = AgentRegistry(
        bus=bus,
        memory_store=memory,
        run_id=turn_id,
        run_logger=run_logger,
        status_reporter=status_reporter,
    )

    try:
        # Start logging agent
        try:
            logger_agent = await registry.spawn("logging")
            await logger_agent.start(log_file=log_file)
        except Exception as e:
            run_logger(f"logging agent unavailable: {e}")

        effective_backend = active_backend()
        run_logger(f"Q&A turn started: {turn_id} (conversation_id={conversation_id}, backend={effective_backend})")
        if project_root:
            run_logger(f"project_root: {project_root}")

        # Get a compact thread history (used in orchestrator context)
        try:
            conversation_history = memory.get_conversation_messages(
                conversation_id=conversation_id,
                limit=max(30, int(conversation_max_messages or 50)),
            )
        except Exception:
            conversation_history = []

        result: Dict[str, Any] = {}

        # Spawn + run the master orchestrator. Any exception here used to bubble
        # up and become a 500 ("internal error") which left the frontend with
        # no usable response. We convert failures into a structured error result
        # and still return a valid QAAskResponse.
        try:
            master = await registry.spawn("hive_master_orchestrator")
            result_any = await master.answer(
                question=question,
                project_root=project_root,
                target_orchestrator=orchestrator,
                use_web=use_web,
                use_local_refs=use_local_refs,
                conversation_id=conversation_id,
                conversation_history=conversation_history,
            )
            result = result_any if isinstance(result_any, dict) else {"answer": str(result_any)}
            if isinstance(result, dict):
                result.setdefault("backend", active_backend())
        except (Exception, asyncio.CancelledError) as e:
            # NOTE: asyncio.CancelledError inherits from BaseException on Py3.11.
            # We catch it explicitly so a cancelled task doesn't explode into a
            # 500 response in the frontend.
            tb = traceback.format_exc()
            run_logger(f"ERROR: Q&A run failed: {type(e).__name__}: {e}")
            for ln in tb.rstrip().splitlines():
                run_logger(ln)
            result = {
                "domain": "Hive Master",
                "question": question,
                "answer": "Internal error while answering. Please check the run log for details.",
                "error": f"{type(e).__name__}: {e}",
                "backend": active_backend(),
            }

        # Persist assistant message into conversation store
        answer_text = str(result.get("answer", "")) if isinstance(result, dict) else ""
        answer_is_error = _looks_like_error_answer(answer_text)
        try:
            memory.add_conversation_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer_text,
                orchestrator="hive_master_orchestrator",
                input_mode=None,
                meta={"turn_id": turn_id, "result": result, "exclude_from_training": bool(answer_is_error)},
                created_at=_now_iso(),
            )
            memory.update_conversation(
                conversation_id=conversation_id,
                updated_at=_now_iso(),
                metadata_patch={
                    "last_answer_domain": (result.get("domain") if isinstance(result, dict) else None),
                    "last_orchestrators_used": (result.get("orchestrators_used") if isinstance(result, dict) else None),
                },
            )
        except Exception:
            pass

        # Persist assistant message to hive memory only when it is a usable answer.
        if not answer_is_error:
            try:
                memory.add(
                    scope="hive",
                    content=json.dumps(
                        {
                            "event": "conversation_message",
                            "conversation_id": conversation_id,
                            "turn_id": turn_id,
                            "role": "assistant",
                            "orchestrator": "hive_master_orchestrator",
                            "answer": answer_text,
                            "result": result,
                        },
                        ensure_ascii=False,
                    ),
                    tags=["conversation", "qa", conversation_id, "assistant"],
                    success=True,
                    created_at=_now_iso(),
                )
            except Exception:
                pass

        finished_at = _now_iso()
        run_logger("Q&A turn finished")

        # Return messages (latest N). If DB retrieval fails for any reason,
        # return an in-memory fallback thread so the frontend still shows
        # a response bubble for this turn.
        try:
            messages = memory.get_conversation_messages(
                conversation_id=conversation_id,
                limit=max(20, int(conversation_max_messages or 50)),
            )
        except Exception:
            messages = []

        # Fallback message thread for UI robustness
        fallback_messages: List[Dict[str, Any]] = [
            {
                "id": -1,
                "conversation_id": conversation_id,
                "role": "user",
                "content": question,
                "orchestrator": orchestrator,
                "input_mode": input_mode,
                "meta": {"turn_id": turn_id, "project_path": project_path},
                "created_at": started_at,
            },
            {
                "id": -2,
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": answer_text,
                "orchestrator": "hive_master_orchestrator",
                "input_mode": None,
                "meta": {"turn_id": turn_id},
                "created_at": finished_at,
            },
        ]

        if not messages:
            messages = fallback_messages
        else:
            # If the assistant message wasn't persisted (e.g. transient DB lock),
            # append it so the UI shows a response.
            has_turn_assistant = False
            for m in messages:
                try:
                    if m.get("role") == "assistant" and (m.get("meta") or {}).get("turn_id") == turn_id:
                        has_turn_assistant = True
                        break
                except Exception:
                    continue
            if not has_turn_assistant:
                messages.append(fallback_messages[-1])

        return QARunResult(
            conversation_id=conversation_id,
            turn_id=turn_id,
            started_at=started_at,
            finished_at=finished_at,
            log_file=str(log_file),
            result=result if isinstance(result, dict) else {},
            logs=logs,
            messages=messages,
        )

    finally:
        try:
            await registry.shutdown_all()
        except Exception:
            pass
