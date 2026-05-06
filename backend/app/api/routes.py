from __future__ import annotations

import json
import uuid
import traceback
import logging
from datetime import datetime, timezone
from typing import Any, Dict
import asyncio

from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from app.api.schemas import (
    ConversationTraceGetResponse,
    ConversationTraceListResponse,
    RunTraceGetResponse,
    RunTraceListResponse,
    MemorySearchResponse,
    ModelBackendSetRequest,
    ModelBackendStatusResponse,
    ModelFeedbackStatsResponse,
    ModelFeedbackListResponse,
    ModelFeedbackResponse,
    ModelFeedbackRequest,
    QAAskRequest,
    QAAskResponse,
    QAConversationCreateRequest,
    QAConversationCreateResponse,
    QAConversationGetResponse,
    QAConversationListResponse,
    RunCreateRequest,
    RunCreateResponse,
    RunStatusResponse,
    UploadCreateResponse,
    UploadListResponse,
    UploadGetResponse,
    WorkspaceModuleInfo,
    WorkspaceModuleListResponse,
    WorkspaceModuleCreateRequest,
    WorkspaceModuleCreateResponse,
    WorkspaceModuleRunRequest,
    WorkspaceModuleRunResponse,
    SpeechFormatRequest,
    SpeechFormatResponse,
    SpeechFeedbackRequest,
    SpeechReplacementsResponse,
)
from app.memory.store import get_memory_store
from app.speech.formatting import format_transcript, extract_replacements
from app.runtime.qa_runner import run_qa
from app.runtime.run_manager import get_run_manager
from app.uploads.ingest import ingest_upload
from app.feedback.rl import resolve_qa_context, submit_feedback
from app.workspace_modules.manager import WorkspaceModuleManager
from app.workspace_modules.builtin import ensure_builtin_modules
from app.llm.factory import get_backend_status, get_local_runtime_status, set_active_backend, use_llm_backend

router = APIRouter()
log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.get("/health")
def health():
    return {"ok": True}


# ------------------------------
# Runtime LLM backend switching
# ------------------------------


@router.get("/model/backend", response_model=ModelBackendStatusResponse)
def get_model_backend():
    return get_backend_status()


@router.post("/model/backend", response_model=ModelBackendStatusResponse)
def set_model_backend(req: ModelBackendSetRequest):
    requested = str(req.backend or "").strip().lower()
    log.info("Model backend switch request received: %s", requested)
    try:
        status = set_active_backend(requested)
        log.info("Model backend switch request completed: active=%s", status.get("active_backend"))
        return status
    except ValueError as e:
        log.warning("Model backend switch rejected: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Model backend switch failed")
        raise HTTPException(status_code=500, detail=f"Failed to switch backend: {type(e).__name__}: {e}")


@router.get("/local/status")
def local_runtime_status(deep: bool = Query(False, description="If true, import torch and probe CUDA. Default is shallow/non-blocking.")) -> Dict[str, Any]:
    """Report local LLM/trainer status. Shallow by default; deep probes CUDA."""
    return get_local_runtime_status(deep=bool(deep))


# ------------------------------
# Model feedback / RLHF reward signals
# ------------------------------


@router.post("/feedback", response_model=ModelFeedbackResponse)
async def submit_model_feedback(req: ModelFeedbackRequest):
    """Store user feedback from Q&A or code-pipeline responses.

    Positive/corrected feedback is also queued for the manual local LoRA/RLHF-style
    trainer. Negative feedback is stored as a reward signal and hive memory; if the
    user supplies a corrected response, that corrected response becomes the preferred
    training target.
    """
    store = get_memory_store()
    target_type = (req.target_type or "qa").strip().lower()
    if target_type not in {"qa", "code"}:
        raise HTTPException(status_code=400, detail="target_type must be 'qa' or 'code'")

    target_id = (req.target_id or req.run_id or "").strip()
    prompt = req.prompt or ""
    response = req.response or ""
    project_path = req.project_path or ""
    input_mode = req.input_mode or ""
    meta = dict(req.meta or {})

    # Best-effort recovery when the frontend only sends ids.
    if target_type == "qa" and req.conversation_id and target_id and (not prompt or not response):
        ctx = resolve_qa_context(store, conversation_id=req.conversation_id, turn_id=target_id)
        prompt = prompt or str(ctx.get("prompt") or "")
        response = response or str(ctx.get("response") or "")
        project_path = project_path or str(ctx.get("project_path") or "")
        input_mode = input_mode or str(ctx.get("input_mode") or "")
        if ctx.get("orchestrator"):
            meta.setdefault("orchestrator", ctx.get("orchestrator"))

    if target_type == "code" and req.run_id and (not prompt or not response or not project_path or not input_mode):
        status = get_run_manager().get_status(req.run_id)
        if status is not None:
            prompt = prompt or str(getattr(status, "prompt", "") or "")
            input_mode = input_mode or str(getattr(status, "input_mode", "") or "")
            if status.result:
                prompt = prompt or str((status.result or {}).get("prompt") or "")
                response = response or json.dumps(status.result, ensure_ascii=False, indent=2)
                project_path = project_path or str((status.result or {}).get("project_root") or "")
            else:
                response = response or str(status.error or "")
                if not response and status.logs:
                    response = "\n".join((status.logs or [])[-120:])
            project_path = project_path or str(status.project_root or "")

    if not target_id:
        target_id = req.run_id or req.conversation_id or f"feedback-{uuid.uuid4().hex[:12]}"

    # Store feedback even if we cannot recover the original prompt. The feedback
    # stats/reward signal should still update. The manual trainer will simply skip
    # rows that lack a usable prompt.
    if not str(prompt or "").strip():
        meta.setdefault("warning", "original prompt was not available; feedback saved but not used as a supervised training pair")

    result = submit_feedback(
        store=store,
        target_type=target_type,
        target_id=target_id,
        conversation_id=req.conversation_id,
        run_id=req.run_id or (target_id if target_type == "code" else None),
        score=int(req.score),
        prompt=prompt,
        response=response,
        corrected_response=req.corrected_response or "",
        comment=req.comment or "",
        backend=req.backend or str(get_backend_status().get("active_backend") or ""),
        project_path=project_path,
        mode=req.mode or target_type,
        input_mode=input_mode,
        add_to_training=bool(req.add_to_training),
        meta=meta,
    )
    return ModelFeedbackResponse(**result)


@router.get("/feedback", response_model=ModelFeedbackListResponse)
def list_model_feedback(
    target_type: str | None = Query(None, description="Optional feedback target type: qa|code"),
    target_id: str | None = Query(None, description="Optional Q&A turn id or run id"),
    limit: int = Query(50, ge=1, le=500),
):
    store = get_memory_store()
    rows = store.list_model_feedback(limit=int(limit), target_type=target_type, target_id=target_id)
    return ModelFeedbackListResponse(feedback=rows)


@router.get("/feedback/stats", response_model=ModelFeedbackStatsResponse)
def model_feedback_stats():
    store = get_memory_store()
    return ModelFeedbackStatsResponse(**store.model_feedback_stats())


# ------------------------------
# Uploads (documents/images)
# ------------------------------

@router.post("/uploads", response_model=UploadCreateResponse)
async def upload_file(
    file: UploadFile = File(...),
    train: bool = Query(False, description="If true, create training examples from this upload."),
    add_to_hive: bool = Query(True, description="If true, add extracted content into hive memory for retrieval."),
):
    """Upload a document or image and ingest it into the hive.

    The backend extracts text (best-effort), chunks it into hive memory (so local HF, Gemini, and OpenAI
    can use it), and optionally creates SFT training examples for the background LoRA worker.
    """
    from app.core.config import settings

    data = await file.read()
    max_bytes = int(getattr(settings, "uploads_max_mb", 50) or 50) * 1024 * 1024
    if data is None:
        raise HTTPException(status_code=400, detail="No file received")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (>{max_bytes} bytes).")

    store = get_memory_store()
    result = await ingest_upload(
        store=store,
        data=data,
        filename=file.filename or "upload",
        content_type=file.content_type,
        add_to_hive_memory=bool(add_to_hive),
        create_training_examples=bool(train),
    )

    info = {
        "upload_id": result.upload_id,
        "filename": result.filename,
        "content_type": result.content_type,
        "size_bytes": result.size_bytes,
        "text_chars": result.text_chars,
        "chunks_added": result.chunks_added,
        "training_examples_added": result.training_examples_added,
        "created_at": _now_iso(),
    }

    return {
        "upload": info,
        "memory_ids": result.memory_ids,
        "message": f"Ingested {result.filename} (chunks={result.chunks_added}, training_examples={result.training_examples_added})",
    }


@router.get("/uploads", response_model=UploadListResponse)
def list_uploads(limit: int = Query(50, ge=1, le=500)):
    store = get_memory_store()
    rows = store.list_uploads(limit=int(limit))
    uploads = []
    for r in rows:
        uploads.append(
            {
                "upload_id": r["upload_id"],
                "filename": r["filename"],
                "content_type": r.get("content_type") or None,
                "size_bytes": int(r.get("size_bytes") or 0),
                "text_chars": int(r.get("text_chars") or 0),
                "chunks_added": int(r.get("chunks_added") or 0),
                "training_examples_added": int(r.get("training_examples_added") or 0),
                "created_at": r.get("created_at") or "",
            }
        )
    return {"uploads": uploads}


@router.get("/uploads/{upload_id}", response_model=UploadGetResponse)
def get_upload(upload_id: str):
    store = get_memory_store()
    r = store.get_upload(upload_id=upload_id)
    if not r:
        raise HTTPException(status_code=404, detail="Upload not found")
    info = {
        "upload_id": r["upload_id"],
        "filename": r["filename"],
        "content_type": r.get("content_type") or None,
        "size_bytes": int(r.get("size_bytes") or 0),
        "text_chars": int(r.get("text_chars") or 0),
        "chunks_added": int(r.get("chunks_added") or 0),
        "training_examples_added": int(r.get("training_examples_added") or 0),
        "created_at": r.get("created_at") or "",
    }
    return {"upload": info}


# ------------------------------
# Workspace Modules
# ------------------------------


@router.get("/modules", response_model=WorkspaceModuleListResponse)
def list_workspace_modules():
    mgr = WorkspaceModuleManager()
    try:
        ensure_builtin_modules(mgr)
    except Exception:
        pass
    mods = [WorkspaceModuleInfo(**m.to_dict()) for m in mgr.list_modules()]
    return {"modules": mods}


@router.post("/modules", response_model=WorkspaceModuleCreateResponse)
async def create_workspace_module(req: WorkspaceModuleCreateRequest):
    mgr = WorkspaceModuleManager()
    try:
        ensure_builtin_modules(mgr)
    except Exception:
        pass

    # Template short-circuit
    tpl = (req.template or "").strip().lower() if req.template else ""
    if tpl in {"calculator", "script_runner"}:
        ensure_builtin_modules(mgr)
        mod = mgr.get_module(tpl)
        if mod is None:
            raise HTTPException(status_code=500, detail="Failed to create built-in module")
        return {"module": WorkspaceModuleInfo(**mod.to_dict()), "message": f"Built-in module '{tpl}' is ready."}

    files: Dict[str, str] = dict(req.files or {})
    if req.code is not None and req.code.strip():
        files[req.entrypoint or "run.py"] = req.code

    if not files:
        raise HTTPException(status_code=400, detail="Provide either template, code, or files")

    try:
        mod = mgr.create_module(
            name=req.name,
            description=req.description,
            entrypoint=req.entrypoint,
            files=files,
            requirements=list(req.requirements or []),
            overwrite=bool(req.overwrite),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Publish to hive memory so agents can discover it.
    if req.publish_to_hive:
        try:
            store = get_memory_store()
            store.add(
                scope="hive",
                content=json.dumps(
                    {
                        "event": "workspace_module_created",
                        "name": mod.name,
                        "description": mod.description,
                        "entrypoint": mod.entrypoint,
                        "rel_path": mod.rel_path,
                    },
                    ensure_ascii=False,
                ),
                tags=["workspace_module", mod.name],
                success=True,
                created_at=_now_iso(),
            )
        except Exception:
            pass

    return {"module": WorkspaceModuleInfo(**mod.to_dict()), "message": f"Created module '{mod.name}'."}


@router.post("/modules/{module_name}/run", response_model=WorkspaceModuleRunResponse)
async def run_workspace_module(module_name: str, req: WorkspaceModuleRunRequest):
    mgr = WorkspaceModuleManager()
    try:
        ensure_builtin_modules(mgr)
    except Exception:
        pass

    mod = mgr.get_module(module_name)
    if mod is None:
        raise HTTPException(status_code=404, detail="Module not found")

    try:
        # subprocess.run is blocking; keep API responsive by offloading.
        res = await asyncio.to_thread(mgr.run_module, name=module_name, args=req.args or [], timeout_s=req.timeout_s)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "module": WorkspaceModuleInfo(**mod.to_dict()),
        "returncode": int(res.returncode),
        "stdout": (res.stdout or "")[-20000:],
        "stderr": (res.stderr or "")[-20000:],
    }



@router.post("/runs", response_model=RunCreateResponse)
async def create_run(req: RunCreateRequest):
    run_manager = get_run_manager()
    run_id = await run_manager.start_run(
        project_path=req.project_path,
        prompt=req.prompt,
        copy_project_to_workspace=req.copy_project_to_workspace,
        input_mode=req.input_mode,
        llm_backend=getattr(req, "backend", None),
    )
    return RunCreateResponse(run_id=run_id)


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str):
    run_manager = get_run_manager()
    status = run_manager.get_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="run_id not found")
    return status


# ------------------------------
# Code-pipeline trace viewer
# ------------------------------


@router.get("/runs/{run_id}/traces", response_model=RunTraceListResponse)
async def list_run_traces(
    run_id: str,
    orchestrator: str | None = Query(None, description="Optional orchestrator (agent_type) filter"),
    limit: int = Query(50, ge=1, le=500),
):
    """List stored sequence traces for a code-pipeline run."""

    store = get_memory_store()
    try:
        links = store.list_run_traces(run_id=run_id, orchestrator=orchestrator, limit=int(limit))
    except Exception:
        links = []

    enriched = []
    for l in links:
        mem = None
        mem_id = l.get("hive_memory_id") or l.get("type_memory_id")
        if mem_id is not None:
            mem = store.get_memory_by_id(memory_id=int(mem_id))

        trace_obj = {}
        if mem and isinstance(mem.get("content"), str):
            try:
                trace_obj = json.loads(mem["content"])
                if not isinstance(trace_obj, dict):
                    trace_obj = {}
            except Exception:
                trace_obj = {}

        summary = {
            "query": trace_obj.get("query"),
            "domain": trace_obj.get("domain"),
            "success": trace_obj.get("success"),
            "started_at": trace_obj.get("started_at"),
            "finished_at": trace_obj.get("finished_at"),
            "agent_sequence": trace_obj.get("agent_sequence") or [],
            "memory_sequence": trace_obj.get("memory_sequence") or [],
        }

        enriched.append({**l, "summary": summary})

    return RunTraceListResponse(run_id=run_id, traces=enriched)


@router.get("/runs/traces/{trace_id}", response_model=RunTraceGetResponse)
async def get_run_trace(trace_id: int):
    """Fetch a full code-pipeline trace payload by run trace id."""

    store = get_memory_store()
    link = store.get_run_trace(trace_id=int(trace_id))
    if link is None:
        raise HTTPException(status_code=404, detail="trace_id not found")

    mem = None
    mem_id = link.get("hive_memory_id") or link.get("type_memory_id")
    if mem_id is not None:
        mem = store.get_memory_by_id(memory_id=int(mem_id))
    if mem is None:
        raise HTTPException(status_code=404, detail="trace memory entry not found")

    trace_obj: dict = {}
    if isinstance(mem.get("content"), str):
        try:
            trace_obj = json.loads(mem["content"])
            if not isinstance(trace_obj, dict):
                trace_obj = {"raw": mem.get("content")}
        except Exception:
            trace_obj = {"raw": mem.get("content")}

    summary = {
        "query": trace_obj.get("query"),
        "domain": trace_obj.get("domain"),
        "success": trace_obj.get("success"),
        "started_at": trace_obj.get("started_at"),
        "finished_at": trace_obj.get("finished_at"),
        "agent_sequence": trace_obj.get("agent_sequence") or [],
        "memory_sequence": trace_obj.get("memory_sequence") or [],
    }

    link_with_summary = {**link, "summary": summary}
    return RunTraceGetResponse(link=link_with_summary, memory_entry=mem, trace=trace_obj)


# ------------------------------
# Q&A / multi-turn chat
# ------------------------------


@router.post("/qa/ask", response_model=QAAskResponse)
async def qa_ask(req: QAAskRequest):
    # Defensive wrapper: even if something goes wrong inside the Q&A runner,
    # we return a *structured* response instead of a 500. This prevents the
    # frontend from showing an empty window / "unable to fetch".
    try:
        with use_llm_backend(getattr(req, "backend", None)):
            qa_res = await run_qa(
                question=req.question,
                orchestrator=req.orchestrator,
                project_path=req.project_path,
                use_web=req.use_web,
                use_local_refs=req.use_local_refs,
                input_mode=req.input_mode,
                conversation_id=req.conversation_id,
                conversation_title=req.conversation_title,
            )
        answer_text = str((qa_res.result or {}).get("answer", ""))
        return QAAskResponse(
            run_id=qa_res.turn_id,
            turn_id=qa_res.turn_id,
            conversation_id=qa_res.conversation_id,
            answer=answer_text,
            result=qa_res.result,
            log_file=qa_res.log_file,
            logs=qa_res.logs,
            messages=qa_res.messages,
        )
    except Exception as e:
        conv_id = req.conversation_id or f"conv-error-{uuid.uuid4().hex[:12]}"
        turn_id = f"turn-error-{uuid.uuid4().hex[:12]}"
        tb = traceback.format_exc()
        answer_text = "Internal server error while answering. Check backend logs for details."
        result = {
            "domain": "Hive Master",
            "question": req.question,
            "answer": answer_text,
            "error": f"{type(e).__name__}: {e}",
        }
        logs = [f"ERROR: {type(e).__name__}: {e}"] + tb.rstrip().splitlines()[-80:]
        # Minimal message thread so the UI shows a response bubble.
        messages = [
            {
                "id": -1,
                "conversation_id": conv_id,
                "role": "user",
                "content": req.question,
                "orchestrator": req.orchestrator,
                "input_mode": req.input_mode,
                "meta": {"turn_id": turn_id, "project_path": req.project_path},
                "created_at": _now_iso(),
            },
            {
                "id": -2,
                "conversation_id": conv_id,
                "role": "assistant",
                "content": answer_text,
                "orchestrator": "hive_master_orchestrator",
                "input_mode": None,
                "meta": {"turn_id": turn_id},
                "created_at": _now_iso(),
            },
        ]
        return QAAskResponse(
            run_id=turn_id,
            turn_id=turn_id,
            conversation_id=conv_id,
            answer=answer_text,
            result=result,
            log_file="",
            logs=logs,
            messages=messages,
        )


@router.post("/qa/conversations", response_model=QAConversationCreateResponse)
async def qa_create_conversation(req: QAConversationCreateRequest):
    store = get_memory_store()
    conversation_id = f"conv-{uuid.uuid4().hex[:12]}"
    try:
        store.create_conversation(
            conversation_id=conversation_id,
            orchestrator=req.orchestrator,
            title=req.title,
            metadata={"kind": "qa"},
            created_at=_now_iso(),
        )
    except Exception:
        # best-effort
        pass
    return QAConversationCreateResponse(conversation_id=conversation_id)


@router.get("/qa/conversations", response_model=QAConversationListResponse)
async def qa_list_conversations(
    orchestrator: str | None = Query(None, description="Optional orchestrator filter"),
    limit: int = Query(25, ge=1, le=200),
):
    store = get_memory_store()
    try:
        items = store.list_conversations(limit=int(limit), orchestrator=orchestrator)
    except Exception:
        items = []
    return QAConversationListResponse(conversations=items)


@router.get("/qa/conversations/{conversation_id}", response_model=QAConversationGetResponse)
async def qa_get_conversation(
    conversation_id: str,
    limit: int = Query(100, ge=1, le=500),
):
    store = get_memory_store()
    convo = store.get_conversation(conversation_id=conversation_id)
    if convo is None:
        raise HTTPException(status_code=404, detail="conversation_id not found")
    msgs = store.get_conversation_messages(conversation_id=conversation_id, limit=int(limit))
    return QAConversationGetResponse(conversation=convo, messages=msgs)


@router.get("/qa/conversations/{conversation_id}/traces", response_model=ConversationTraceListResponse)
async def qa_list_conversation_traces(
    conversation_id: str,
    turn_id: str | None = Query(None, description="Optional turn_id filter"),
    orchestrator: str | None = Query(None, description="Optional orchestrator (agent_type) filter"),
    limit: int = Query(50, ge=1, le=500),
):
    """List stored per-turn sequence traces for a conversation.

    These traces are generated by orchestrators and include:
    - ordered agent calls
    - ordered memory access operations
    - sequence learning context (recommended order + similar traces)
    """

    store = get_memory_store()
    links = store.list_conversation_traces(
        conversation_id=conversation_id,
        turn_id=turn_id,
        orchestrator=orchestrator,
        limit=int(limit),
    )

    enriched = []
    for l in links:
        mem = None
        mem_id = l.get("hive_memory_id") or l.get("type_memory_id")
        if mem_id is not None:
            mem = store.get_memory_by_id(memory_id=int(mem_id))

        trace_obj = {}
        if mem and isinstance(mem.get("content"), str):
            try:
                trace_obj = json.loads(mem["content"])
                if not isinstance(trace_obj, dict):
                    trace_obj = {}
            except Exception:
                trace_obj = {}

        summary = {
            "query": trace_obj.get("query"),
            "domain": trace_obj.get("domain"),
            "success": trace_obj.get("success"),
            "started_at": trace_obj.get("started_at"),
            "finished_at": trace_obj.get("finished_at"),
            "agent_sequence": trace_obj.get("agent_sequence") or [],
            "memory_sequence": trace_obj.get("memory_sequence") or [],
        }

        enriched.append({**l, "summary": summary})

    return ConversationTraceListResponse(conversation_id=conversation_id, turn_id=turn_id, traces=enriched)


@router.get("/qa/traces/{trace_id}", response_model=ConversationTraceGetResponse)
async def qa_get_trace(trace_id: int):
    """Fetch a full trace payload by trace link id."""
    store = get_memory_store()
    link = store.get_conversation_trace(trace_id=int(trace_id))
    if link is None:
        raise HTTPException(status_code=404, detail="trace_id not found")

    mem = None
    mem_id = link.get("hive_memory_id") or link.get("type_memory_id")
    if mem_id is not None:
        mem = store.get_memory_by_id(memory_id=int(mem_id))
    if mem is None:
        raise HTTPException(status_code=404, detail="trace memory entry not found")

    trace_obj: dict = {}
    if isinstance(mem.get("content"), str):
        try:
            trace_obj = json.loads(mem["content"])
            if not isinstance(trace_obj, dict):
                trace_obj = {"raw": mem.get("content")}
        except Exception:
            trace_obj = {"raw": mem.get("content")}

    summary = {
        "query": trace_obj.get("query"),
        "domain": trace_obj.get("domain"),
        "success": trace_obj.get("success"),
        "started_at": trace_obj.get("started_at"),
        "finished_at": trace_obj.get("finished_at"),
        "agent_sequence": trace_obj.get("agent_sequence") or [],
        "memory_sequence": trace_obj.get("memory_sequence") or [],
    }

    link_with_summary = {**link, "summary": summary}
    return ConversationTraceGetResponse(link=link_with_summary, memory_entry=mem, trace=trace_obj)


# ------------------------------
# Memory search endpoints
# ------------------------------


@router.get("/memory/hive/search", response_model=MemorySearchResponse)
def search_hive_memory(q: str = Query(..., min_length=1)):
    store = get_memory_store()
    entries = store.search(scope="hive", query=q, limit=25)
    return MemorySearchResponse(query=q, entries=entries)


@router.get("/memory/type/{agent_type}/search", response_model=MemorySearchResponse)
def search_type_memory(agent_type: str, q: str = Query(..., min_length=1)):
    store = get_memory_store()
    entries = store.search(scope="type", agent_type=agent_type, query=q, limit=25)
    return MemorySearchResponse(query=q, entries=entries)


@router.get("/memory/all/search", response_model=MemorySearchResponse)
def search_all_memory(
    q: str = Query(..., min_length=1),
    scopes: str | None = Query(
        None,
        description="Optional comma-separated scopes to search (hive,type,agent). If omitted, searches all scopes.",
    ),
):
    store = get_memory_store()
    scopes_list = [s.strip() for s in scopes.split(",") if s.strip()] if scopes else None
    entries = store.search_all(query=q, limit=50, scopes=scopes_list)
    return MemorySearchResponse(query=q, entries=entries)



# ------------------------------
# Speech formatting + learning
# ------------------------------

@router.post("/speech/format", response_model=SpeechFormatResponse)
async def speech_format(req: SpeechFormatRequest):
    store = get_memory_store()
    replacements = store.list_speech_replacements(mode=req.mode, limit=200)
    formatted, meta = format_transcript(req.text, mode=req.mode, replacements=replacements, list_mode_active=False)
    return SpeechFormatResponse(formatted_text=formatted, meta=meta)


@router.get("/speech/replacements", response_model=SpeechReplacementsResponse)
async def speech_replacements(mode: str = Query("qa"), limit: int = Query(100)):
    store = get_memory_store()
    items = store.list_speech_replacements(mode=mode, limit=limit)
    return SpeechReplacementsResponse(mode=mode, items=items)


@router.post("/speech/feedback")
async def speech_feedback(req: SpeechFeedbackRequest):
    """Persist user corrections so formatting improves over time."""
    store = get_memory_store()
    now = datetime.now(timezone.utc).isoformat()

    # Extract conservative replacement pairs from the edit
    pairs = extract_replacements(req.raw_text, req.final_text)
    for src, dst in pairs:
        store.upsert_speech_replacement(mode=req.mode, src=src, dst=dst, updated_at=now)

    # Also feed into local training dataset (so local LLM can learn formatting + corrections)
    try:
        prompt = (
            f"Rewrite the following dictation transcript into clean, well-formatted text for mode={req.mode}. "
            "If the user says first/second/third point, format as bullet points. "
            "If the user dictates option A/B/C/D or option 1/2/3/4, format them as a clean list. "
            "Return only the final formatted text.\n\n"
            f"TRANSCRIPT:\n{req.raw_text.strip()}\n"
        )
        completion = req.final_text.strip()
        if completion:
            store.add_training_example(
                source="speech_feedback",
                prompt=prompt,
                completion=completion,
                created_at=now,
                upload_id=None,
            )
    except Exception:
        # Don't break the UI if training capture fails
        pass

    return {"ok": True, "pairs_learned": len(pairs)}
