from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.memory.store import MemoryStore


_ERRORISH = (
    "internal error in",
    "an internal error occurred",
    "internal error while answering",
    "please check the run log for details",
    "could not parse json from model output",
    "q&a agent output must be json object",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any, *, limit: int = 20000) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[:limit] + "\n...[truncated]"
    return text


def looks_bad_training_text(value: Any) -> bool:
    lowered = str(value or "").strip().lower()
    if not lowered:
        return True
    return any(p in lowered for p in _ERRORISH)


def reward_from_score(score: int, corrected_response: str = "") -> float:
    try:
        s = int(score)
    except Exception:
        s = 0
    if corrected_response.strip():
        return 1.0 if s >= 0 else 0.75
    if s > 0:
        return 1.0
    if s < 0:
        return -1.0
    return 0.0


def training_prompt_for_feedback(
    *,
    target_type: str,
    prompt: str,
    comment: str = "",
    project_path: str = "",
) -> str:
    target = (target_type or "qa").strip().lower()
    base = clean_text(prompt, limit=8000)
    note = clean_text(comment, limit=2000)
    project = clean_text(project_path, limit=1000)

    if target == "code":
        parts = [
            "You are updating or fixing code for an Agentic Hive Studio code-pipeline request.",
            "Original user task:",
            base,
        ]
        if project:
            parts.extend(["Project path:", project])
        if note:
            parts.extend(["User feedback:", note])
        parts.append("Return the corrected final answer, code, or patch content requested by the feedback.")
        return "\n\n".join(parts).strip()

    parts = [base]
    if note:
        parts.extend(["User feedback:", note])
    return "\n\n".join(parts).strip()


def resolve_qa_context(store: MemoryStore, *, conversation_id: str, turn_id: str) -> Dict[str, Any]:
    """Best-effort lookup of a Q&A turn's user prompt and assistant response."""
    out: Dict[str, Any] = {}
    if not conversation_id or not turn_id:
        return out
    try:
        messages = store.get_conversation_messages(conversation_id=conversation_id, limit=500)
    except Exception:
        return out

    for msg in messages:
        try:
            meta = msg.get("meta") or {}
            if not isinstance(meta, dict) or meta.get("turn_id") != turn_id:
                continue
            role = str(msg.get("role") or "").lower()
            if role == "user" and not out.get("prompt"):
                out["prompt"] = msg.get("content") or ""
                out["input_mode"] = msg.get("input_mode") or None
                out["project_path"] = (meta.get("project_path") if isinstance(meta, dict) else None) or None
            elif role == "assistant" and not out.get("response"):
                out["response"] = msg.get("content") or ""
                out["orchestrator"] = msg.get("orchestrator") or None
        except Exception:
            continue
    return out


def submit_feedback(
    *,
    store: MemoryStore,
    target_type: str,
    target_id: str,
    score: int,
    prompt: str = "",
    response: str = "",
    corrected_response: str = "",
    comment: str = "",
    backend: str = "",
    project_path: str = "",
    mode: str = "",
    input_mode: str = "",
    conversation_id: Optional[str] = None,
    run_id: Optional[str] = None,
    add_to_training: bool = True,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target_type = (target_type or "").strip().lower() or "qa"
    target_id = clean_text(target_id, limit=500)
    prompt = clean_text(prompt, limit=20000)
    response = clean_text(response, limit=20000)
    corrected_response = clean_text(corrected_response, limit=20000)
    comment = clean_text(comment, limit=5000)
    backend = clean_text(backend, limit=200)
    project_path = clean_text(project_path, limit=2000)
    mode = clean_text(mode or target_type, limit=100)
    input_mode = clean_text(input_mode, limit=100)
    created_at = now_iso()

    reward = reward_from_score(score, corrected_response)

    feedback_id = store.add_model_feedback(
        target_type=target_type,
        target_id=target_id,
        conversation_id=conversation_id,
        run_id=run_id,
        score=int(score),
        reward=reward,
        prompt=prompt,
        response=response,
        corrected_response=corrected_response,
        comment=comment,
        backend=backend,
        project_path=project_path,
        mode=mode,
        input_mode=input_mode,
        meta=meta or {},
        created_at=created_at,
    )

    sentiment = "positive" if reward > 0 else ("negative" if reward < 0 else "neutral")
    memory_payload = {
        "event": "model_feedback",
        "feedback_id": feedback_id,
        "target_type": target_type,
        "target_id": target_id,
        "conversation_id": conversation_id,
        "run_id": run_id,
        "score": int(score),
        "reward": reward,
        "sentiment": sentiment,
        "backend": backend,
        "project_path": project_path,
        "prompt": prompt[:4000],
        "response": response[:4000],
        "corrected_response": corrected_response[:4000],
        "comment": comment[:2000],
        "add_to_training": bool(add_to_training),
    }
    memory_id = store.add(
        scope="hive",
        content=json.dumps(memory_payload, ensure_ascii=False),
        tags=["feedback", "rlhf", target_type, sentiment, backend or "unknown"],
        success=(reward > 0),
        created_at=created_at,
    )

    training_example_id: Optional[int] = None
    train_completion = ""
    source = ""
    if add_to_training:
        if corrected_response.strip():
            train_completion = corrected_response
            source = f"rlhf_correction_{target_type}"
        elif reward > 0 and response.strip():
            train_completion = response
            source = f"rlhf_positive_{target_type}"

    if train_completion and prompt.strip() and not looks_bad_training_text(train_completion):
        training_prompt = training_prompt_for_feedback(
            target_type=target_type,
            prompt=prompt,
            comment=comment,
            project_path=project_path,
        )
        if training_prompt and not looks_bad_training_text(training_prompt):
            training_example_id = store.add_training_example(
                source=source,
                prompt=training_prompt,
                completion=train_completion,
                created_at=created_at,
            )

    if training_example_id:
        msg = "Feedback saved and queued for the manual RLHF/LoRA trainer."
    elif reward < 0:
        msg = "Negative feedback saved as a reward signal and hive memory. Add a correction to train a preferred replacement."
    else:
        msg = "Feedback saved as hive memory."

    return {
        "feedback_id": feedback_id,
        "reward": reward,
        "memory_id": memory_id,
        "training_example_id": training_example_id,
        "message": msg,
    }
