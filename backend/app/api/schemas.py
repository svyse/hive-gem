from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class RunCreateRequest(BaseModel):
    project_path: str = Field(..., description="Path to a local project directory, or a sample_projects name such as hello.")
    prompt: str = Field(..., description="Natural language prompt describing desired feature/change.")
    copy_project_to_workspace: bool = Field(False, description="If true, operate on a workspace copy of non-sample projects. sample_projects are always edited in place.")
    input_mode: str = Field("text", description="How the prompt was provided: text|voice")
    backend: Optional[str] = Field(None, description="Optional per-request LLM backend override: local|openai|gemini")


class RunCreateResponse(BaseModel):
    run_id: str


class ModelBackendSetRequest(BaseModel):
    backend: str = Field(..., description="Runtime LLM backend: local|openai|gemini")


class ModelBackendStatusResponse(BaseModel):
    active_backend: str
    allowed_backends: List[str]
    available: Dict[str, bool]
    models: Dict[str, Optional[str]] = Field(default_factory=dict)
    fallback_enabled: bool = False
    fallback_backends: List[str] = Field(default_factory=list)
    message: Optional[str] = None


class AgentStatus(BaseModel):
    agent_id: str
    agent_type: str
    state: str
    last_update: Optional[str] = None


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    created_at: str
    updated_at: str
    project_root: str
    prompt: Optional[str] = None
    input_mode: Optional[str] = None
    logs: List[str]
    agent_statuses: List[AgentStatus]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MemoryEntry(BaseModel):
    id: int
    scope: str
    agent_type: Optional[str] = None
    agent_id: Optional[str] = None
    content: str
    tags: List[str] = []
    success: Optional[bool] = None
    created_at: str


class MemorySearchResponse(BaseModel):
    query: str
    entries: List[MemoryEntry]


# ------------------------------
# Uploads schemas
# ------------------------------


class UploadInfo(BaseModel):
    upload_id: str
    filename: str
    content_type: Optional[str] = None
    size_bytes: int
    text_chars: int
    chunks_added: int
    training_examples_added: int
    created_at: str


class UploadCreateResponse(BaseModel):
    upload: UploadInfo
    memory_ids: List[int] = []
    message: str = ""


class UploadListResponse(BaseModel):
    uploads: List[UploadInfo]


class UploadGetResponse(BaseModel):
    upload: UploadInfo


# ------------------------------
# Q&A / Chat schemas
# ------------------------------


class ConversationMessage(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    orchestrator: Optional[str] = None
    input_mode: Optional[str] = None
    meta: Dict[str, Any] = {}
    created_at: str


class QAAskRequest(BaseModel):
    question: str = Field(..., description="The question to ask the hive")
    orchestrator: Optional[str] = Field(
        None,
        description=(
            "Optional orchestrator alias or agent_type. Examples: "
            "interactive|hr|law|law_india|law_international|finance|economics|social|medicine|cyber "
            "or explicit agent_type like 'finance_orchestrator'."
        ),
    )
    project_path: Optional[str] = Field(None, description="Optional local project path for local reference lookups.")
    use_web: bool = Field(True, description="If true, allow web research agents to retrieve sources.")
    use_local_refs: bool = Field(True, description="If true, allow local-reference agent to search local projects.")
    input_mode: str = Field("text", description="How the question was provided: text|voice")
    backend: Optional[str] = Field(None, description="Optional per-request LLM backend override: local|openai|gemini")

    # Multi-turn conversation support
    conversation_id: Optional[str] = Field(
        None,
        description=(
            "If provided, appends this question as a new turn in the existing conversation. "
            "If omitted, a new conversation is created."
        ),
    )
    conversation_title: Optional[str] = Field(None, description="Optional title when creating a new conversation")


class QAAskResponse(BaseModel):
    # Backwards-compatible field name (represents a single turn)
    run_id: str
    # Explicit turn id
    turn_id: str
    # Persistent conversation id
    conversation_id: str

    answer: str
    result: Dict[str, Any]
    log_file: str
    logs: List[str]
    messages: List[ConversationMessage]


class QAConversationCreateRequest(BaseModel):
    orchestrator: Optional[str] = Field(
        None,
        description="Optional orchestrator alias or agent_type for this conversation.",
    )
    title: Optional[str] = Field(None, description="Optional title")


class QAConversationCreateResponse(BaseModel):
    conversation_id: str


class QAConversationGetResponse(BaseModel):
    conversation: Dict[str, Any]
    messages: List[ConversationMessage]


class QAConversationListResponse(BaseModel):
    conversations: List[Dict[str, Any]]




# ------------------------------
# Model feedback / RLHF schemas
# ------------------------------


class ModelFeedbackRequest(BaseModel):
    target_type: str = Field(..., description="Feedback target: qa or code")
    target_id: Optional[str] = Field(None, description="Q&A turn id or code run id")
    conversation_id: Optional[str] = Field(None, description="Conversation id for Q&A feedback")
    run_id: Optional[str] = Field(None, description="Run id for code-pipeline feedback")
    score: int = Field(..., ge=-1, le=1, description="-1=bad, 0=neutral/correction-only, 1=good")
    prompt: Optional[str] = Field(None, description="Original user prompt/question")
    response: Optional[str] = Field(None, description="Model response being rated")
    corrected_response: Optional[str] = Field(None, description="Preferred corrected answer/patch supplied by the user")
    comment: Optional[str] = Field(None, description="Optional user feedback notes")
    backend: Optional[str] = Field(None, description="Backend that produced the response: local|gemini|openai")
    project_path: Optional[str] = Field(None, description="Project/sample_projects path, if relevant")
    mode: Optional[str] = Field(None, description="UI mode: qa|code")
    input_mode: Optional[str] = Field(None, description="Prompt input mode: text|voice")
    add_to_training: bool = Field(True, description="If true, positive/corrected responses are added to manual trainer data")
    meta: Dict[str, Any] = Field(default_factory=dict)


class ModelFeedbackResponse(BaseModel):
    feedback_id: int
    reward: float
    memory_id: Optional[int] = None
    training_example_id: Optional[int] = None
    message: str


class ModelFeedbackListResponse(BaseModel):
    feedback: List[Dict[str, Any]]


class ModelFeedbackStatsResponse(BaseModel):
    total: int = 0
    positive: int = 0
    negative: int = 0
    corrections: int = 0
    max_id: int = 0
    by_backend: Dict[str, int] = Field(default_factory=dict)
    by_target_type: Dict[str, int] = Field(default_factory=dict)


# ------------------------------
# Trace viewer schemas
# ------------------------------


class ConversationTraceSummary(BaseModel):
    query: Optional[str] = None
    domain: Optional[str] = None
    success: Optional[bool] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    agent_sequence: List[str] = []
    memory_sequence: List[str] = []


class ConversationTraceLink(BaseModel):
    id: int
    conversation_id: str
    turn_id: str
    orchestrator: str
    hive_memory_id: Optional[int] = None
    type_memory_id: Optional[int] = None
    created_at: str
    summary: ConversationTraceSummary = ConversationTraceSummary()


class ConversationTraceListResponse(BaseModel):
    conversation_id: str
    turn_id: Optional[str] = None
    traces: List[ConversationTraceLink]


class ConversationTraceGetResponse(BaseModel):
    link: ConversationTraceLink
    memory_entry: MemoryEntry
    trace: Dict[str, Any]


# ------------------------------
# Run trace viewer schemas (code pipeline)
# ------------------------------


class RunTraceLink(BaseModel):
    id: int
    run_id: str
    orchestrator: str
    hive_memory_id: Optional[int] = None
    type_memory_id: Optional[int] = None
    created_at: str
    summary: ConversationTraceSummary = ConversationTraceSummary()


class RunTraceListResponse(BaseModel):
    run_id: str
    traces: List[RunTraceLink]


class RunTraceGetResponse(BaseModel):
    link: RunTraceLink
    memory_entry: MemoryEntry
    trace: Dict[str, Any]


# ------------------------------
# Workspace Modules schemas
# ------------------------------


class WorkspaceModuleInfo(BaseModel):
    name: str
    description: str = ""
    entrypoint: str = "run.py"
    rel_path: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    requirements: List[str] = []
    files: List[str] = []


class WorkspaceModuleListResponse(BaseModel):
    modules: List[WorkspaceModuleInfo]


class WorkspaceModuleCreateRequest(BaseModel):
    name: str = Field(..., description="Module name (letters/numbers plus _ or -)")
    description: str = Field("", description="Short description")
    entrypoint: str = Field("run.py", description="Entrypoint filename inside the module directory")

    # Either provide a template name, or provide code/files.
    template: Optional[str] = Field(None, description="Optional template name (e.g. calculator, script_runner)")
    code: Optional[str] = Field(None, description="Convenience: content for entrypoint file")
    files: Dict[str, str] = Field(default_factory=dict, description="Map of relative paths -> file contents")

    requirements: List[str] = Field(default_factory=list, description="Optional pip requirements (e.g. requests, numpy>=1.26)")

    overwrite: bool = Field(True, description="Overwrite existing module")
    publish_to_hive: bool = Field(True, description="If true, publish a memory entry so agents can discover it")


class WorkspaceModuleCreateResponse(BaseModel):
    module: WorkspaceModuleInfo
    message: str = ""


class WorkspaceModuleRunRequest(BaseModel):
    args: List[str] = Field(default_factory=list, description="CLI args passed to the module entrypoint")
    timeout_s: Optional[float] = Field(None, description="Optional timeout override")


class WorkspaceModuleRunResponse(BaseModel):
    module: WorkspaceModuleInfo
    returncode: int
    stdout: str = ""
    stderr: str = ""



# ------------------------------
# Speech formatting + learning
# ------------------------------

class SpeechFormatRequest(BaseModel):
    text: str
    mode: str = Field("qa", description="qa|code (affects formatting heuristics)")

class SpeechFormatResponse(BaseModel):
    formatted_text: str
    meta: Dict[str, Any] = Field(default_factory=dict)

class SpeechFeedbackRequest(BaseModel):
    mode: str = Field("qa", description="qa|code")
    raw_text: str
    formatted_text: str
    final_text: str

class SpeechReplacement(BaseModel):
    src: str
    dst: str
    count: int
    updated_at: str

class SpeechReplacementsResponse(BaseModel):
    mode: str
    items: List[SpeechReplacement]
