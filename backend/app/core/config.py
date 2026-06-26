from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _backend_root() -> Path:
    # backend/app/core/config.py -> .../backend
    return Path(__file__).resolve().parents[2]


def _default_data_dir() -> Path:
    # Keep workspace + sqlite DB *outside* the repo by default so uvicorn
    # --reload doesn't restart the server when logs/DB change.
    return Path.home() / ".agentic_hive_studio"


def _parse_csv(value: str | List[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    parts = [p.strip() for p in str(value).split(",")]
    return [p for p in parts if p]


def _parse_kv_csv(value: str | Dict[str, int] | None) -> Dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        out: Dict[str, int] = {}
        for k, v in value.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
        return out

    out: Dict[str, int] = {}
    for part in str(value).split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            out[k] = int(v)
        except Exception:
            continue
    return out


class Settings(BaseSettings):
    """App settings loaded from env vars and backend/.env."""

    model_config = SettingsConfigDict(
        env_file=str(_backend_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        # We parse comma-separated lists manually via field validators.
        # If this is True (default), pydantic-settings will try JSON parsing first
        # for list/dict fields, which breaks common ".env" patterns like "a,b,c".
        enable_decoding=False,
    )

    # ------------------------------
    # Server
    # ------------------------------
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    frontend_port: int = Field(default=5173, alias="FRONTEND_PORT")

    cors_allow_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"],
        alias="CORS_ALLOW_ORIGINS",
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ------------------------------
    # Workspace + memory DB
    # ------------------------------
    workspace_root: Path = Field(default_factory=lambda: _default_data_dir() / "workspace", alias="WORKSPACE_ROOT")
    memory_db_path: Path = Field(default_factory=lambda: _default_data_dir() / "memory.sqlite", alias="MEMORY_DB_PATH")

    # Workspace modules
    # Small utilities stored under <WORKSPACE_ROOT>/modules.
    workspace_module_run_timeout_s: float = Field(default=60.0, alias="WORKSPACE_MODULE_RUN_TIMEOUT_S")

    # Dependency auto-install (pip)
    # When True, the hive may automatically install missing Python packages
    # (via `python -m pip install ...`) when modules/projects fail due to
    # ModuleNotFoundError, and will record the packages into requirements.txt.
    auto_install_deps: bool = Field(default=True, alias="AUTO_INSTALL_DEPS")
    pip_install_timeout_s: int = Field(default=600, alias="PIP_INSTALL_TIMEOUT_S")

    # ------------------------------
    # Uploads (documents/images)
    # ------------------------------
    uploads_dir: Path = Field(default_factory=lambda: _default_data_dir() / "uploads", alias="UPLOADS_DIR")
    uploads_max_mb: int = Field(default=50, alias="UPLOAD_MAX_MB")
    uploads_chunk_size_chars: int = Field(default=1200, alias="UPLOAD_CHUNK_SIZE_CHARS")
    uploads_chunk_overlap_chars: int = Field(default=120, alias="UPLOAD_CHUNK_OVERLAP_CHARS")
    uploads_training_enabled: bool = Field(default=True, alias="UPLOAD_TRAINING_ENABLED")
    uploads_training_max_examples: int = Field(default=80, alias="UPLOAD_TRAINING_MAX_EXAMPLES")

    # ------------------------------
    # Retrieval-augmented generation (RAG)
    # ------------------------------
    # Dependency-free lexical RAG over hive/type/agent memory and current project files.
    # This keeps local responses faster and more grounded before LoRA training catches up.
    rag_enabled: bool = Field(default=True, alias="RAG_ENABLED")
    rag_memory_scan_limit: int = Field(default=350, alias="RAG_MEMORY_SCAN_LIMIT")
    rag_memory_limit: int = Field(default=8, alias="RAG_MEMORY_LIMIT")
    rag_project_file_limit: int = Field(default=10, alias="RAG_PROJECT_FILE_LIMIT")
    rag_max_chars: int = Field(default=9000, alias="RAG_MAX_CHARS")
    rag_max_file_bytes: int = Field(default=200000, alias="RAG_MAX_FILE_BYTES")
    rag_skip_dirs: List[str] = Field(
        default_factory=lambda: [
            ".git",
            ".workspace",
            ".workspaces",
            ".memory",
            "node_modules",
            ".venv",
            "venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            "dist",
            "build",
            ".next",
            "coverage",
        ],
        alias="RAG_SKIP_DIRS",
    )

    # ------------------------------
    # LLM backend selection
    # ------------------------------
    # "local" (HF transformers), "openai", "gemini", or "claude".
    llm_backend: str = Field(default="local", alias="LLM_BACKEND")

    # Enables provider fallback chaining.
    # Example with LLM_BACKEND=openai and LLM_FALLBACK_BACKENDS=openai,gemini,claude,local:
    #   OpenAI -> Gemini -> Claude -> local
    # The selected primary backend is automatically skipped in the fallback list.
    llm_fallback_to_openai: bool = Field(default=False, alias="LLM_FALLBACK_TO_OPENAI")
    llm_fallback_backends: List[str] = Field(
        default_factory=lambda: ["openai", "gemini", "claude", "local"],
        alias="LLM_FALLBACK_BACKENDS",
    )
    llm_fallback_min_chars: int = Field(default=0, alias="LLM_FALLBACK_MIN_CHARS")
    llm_fallback_phrases: List[str] = Field(
        default_factory=lambda: [
            "i don't know",
            "i do not know",
            "cannot answer",
            "can't answer",
            "not enough information",
        ],
        alias="LLM_FALLBACK_PHRASES",
    )

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    # Primary OpenAI model (used by default for both Q&A + code modes).
    # You can override this in .env via OPENAI_MODEL.
    openai_model: str = Field(default="gpt-5.2", alias="OPENAI_MODEL")

    # Optional fallback model if the primary errors (timeouts/5xx/overload/model-not-found/etc.).
    # This is used *within* OpenAI calls (separate from provider fallback chaining).
    openai_fallback_model: Optional[str] = Field(default="gpt-4.1", alias="OPENAI_FALLBACK_MODEL")

    # Optional per-purpose OpenAI routing.
    # - Q&A mode uses OPENAI_MODEL_QA (falls back to OPENAI_MODEL)
    # - Coding mode uses OPENAI_MODEL_CODE (falls back to OPENAI_MODEL)
    # NOTE: per your request, Codex is NOT forced. If OPENAI_MODEL_CODE is unset,
    # the code pipeline will just use OPENAI_MODEL.
    openai_model_qa: Optional[str] = Field(default=None, alias="OPENAI_MODEL_QA")
    openai_model_code: Optional[str] = Field(default=None, alias="OPENAI_MODEL_CODE")
    openai_fallback_model_qa: Optional[str] = Field(default=None, alias="OPENAI_FALLBACK_MODEL_QA")
    openai_fallback_model_code: Optional[str] = Field(default=None, alias="OPENAI_FALLBACK_MODEL_CODE")
    openai_api_key_file: Optional[str] = Field(default=None, alias="OPENAI_API_KEY_FILE")
    openai_quota_cooldown_s: float = Field(default=900.0, alias="OPENAI_QUOTA_COOLDOWN_S")
    openai_insufficient_quota_cooldown_s: int = Field(
        default=900,
        alias="OPENAI_INSUFFICIENT_QUOTA_COOLDOWN_S",
    )

    # Gemini
    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_api_key_file: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY_FILE", "GOOGLE_API_KEY_FILE"),
    )
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_fallback_model: Optional[str] = Field(default=None, alias="GEMINI_FALLBACK_MODEL")
    gemini_model_qa: Optional[str] = Field(default=None, alias="GEMINI_MODEL_QA")
    gemini_model_code: Optional[str] = Field(default=None, alias="GEMINI_MODEL_CODE")
    gemini_fallback_model_qa: Optional[str] = Field(default=None, alias="GEMINI_FALLBACK_MODEL_QA")
    gemini_fallback_model_code: Optional[str] = Field(default=None, alias="GEMINI_FALLBACK_MODEL_CODE")
    gemini_max_retries: int = Field(default=2, alias="GEMINI_MAX_RETRIES")
    gemini_retry_backoff_s: float = Field(default=1.5, alias="GEMINI_RETRY_BACKOFF_S")
    gemini_retry_max_backoff_s: float = Field(default=8.0, alias="GEMINI_RETRY_MAX_BACKOFF_S")
    llm_error_body_log_chars: int = Field(default=1200, alias="LLM_ERROR_BODY_LOG_CHARS")

    # Claude / Anthropic
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_api_key_file: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY_FILE")
    claude_model: str = Field(
        default="claude-sonnet-4-20250514",
        validation_alias=AliasChoices("CLAUDE_MODEL", "ANTHROPIC_MODEL"),
    )
    claude_fallback_model: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CLAUDE_FALLBACK_MODEL", "ANTHROPIC_FALLBACK_MODEL"),
    )
    claude_model_qa: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CLAUDE_MODEL_QA", "ANTHROPIC_MODEL_QA"),
    )
    claude_model_code: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CLAUDE_MODEL_CODE", "ANTHROPIC_MODEL_CODE"),
    )
    claude_fallback_model_qa: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CLAUDE_FALLBACK_MODEL_QA", "ANTHROPIC_FALLBACK_MODEL_QA"),
    )
    claude_fallback_model_code: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CLAUDE_FALLBACK_MODEL_CODE", "ANTHROPIC_FALLBACK_MODEL_CODE"),
    )
    claude_max_tokens: int = Field(default=1024, alias="CLAUDE_MAX_TOKENS")

    # Hugging Face (local)
    hf_home: Optional[Path] = Field(default=None, alias="HF_HOME")
    hf_hub_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"),
    )
    local_llm_model: str = Field(
        # Better default local model (small enough for RTX 3060 12GB with FP16 or CPU offload).
        # You can override this in .env via LOCAL_LLM_MODEL.
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        alias="LOCAL_LLM_MODEL",
        description="Hugging Face model id or local path.",
    )
    local_llm_device: str = Field(default="cuda:0", alias="LOCAL_LLM_DEVICE")
    cuda_visible_devices: str = Field(default="0", alias="CUDA_VISIBLE_DEVICES")
    local_llm_max_new_tokens: int = Field(default=384, alias="LOCAL_LLM_MAX_NEW_TOKENS")
    local_llm_temperature: float = Field(default=0.2, alias="LOCAL_LLM_TEMPERATURE")
    local_llm_top_p: float = Field(default=0.95, alias="LOCAL_LLM_TOP_P")
    local_llm_top_k: int = Field(default=50, alias="LOCAL_LLM_TOP_K")
    local_llm_do_sample: bool = Field(default=True, alias="LOCAL_LLM_DO_SAMPLE")
    local_llm_repetition_penalty: float = Field(default=1.25, alias="LOCAL_LLM_REPETITION_PENALTY")
    local_llm_no_repeat_ngram_size: int = Field(default=6, alias="LOCAL_LLM_NO_REPEAT_NGRAM_SIZE")
    local_llm_repeat_stop_enabled: bool = Field(default=True, alias="LOCAL_LLM_REPEAT_STOP_ENABLED")
    local_llm_repeat_stop_min_tokens: int = Field(default=20, alias="LOCAL_LLM_REPEAT_STOP_MIN_TOKENS")
    local_llm_repeat_token_limit: int = Field(default=10, alias="LOCAL_LLM_REPEAT_TOKEN_LIMIT")
    local_llm_repeat_ngram_limit: int = Field(default=4, alias="LOCAL_LLM_REPEAT_NGRAM_LIMIT")
    local_llm_repeat_max_ngram: int = Field(default=8, alias="LOCAL_LLM_REPEAT_MAX_NGRAM")
    local_llm_load_in_4bit: bool = Field(default=True, alias="LOCAL_LLM_LOAD_IN_4BIT")
    # Some models (rare) require `trust_remote_code=True`. Default is False for safety
    # and to avoid 404s on repos that advertise custom modules but don't ship them.
    local_llm_trust_remote_code: bool = Field(default=False, alias="LOCAL_LLM_TRUST_REMOTE_CODE")
    # Optional: timeout for normal local LLM calls after the model is already loaded.
    # If unset, uses LLM_TIMEOUT_S.
    local_llm_timeout_s: Optional[float] = Field(default=None, alias="LOCAL_LLM_TIMEOUT_S")
    # First local call often includes model download / weight load, which can take
    # much longer than a normal generation call. We give that path a larger timeout
    # so the app does not fall back or error while the model is still warming up.
    local_llm_initial_timeout_s: Optional[float] = Field(default=180.0, alias="LOCAL_LLM_INITIAL_TIMEOUT_S")
    local_llm_adapter_dir: Path = Field(
        default_factory=lambda: _default_data_dir() / "local_llm" / "adapters" / "latest",
        alias="LOCAL_LLM_ADAPTER_DIR",
    )
    local_llm_concurrency: int = Field(default=1, alias="LOCAL_LLM_CONCURRENCY")
    # Purpose-specific caps keep local generations short enough for 12 GB GPUs.
    local_llm_qa_max_new_tokens: int = Field(default=160, alias="LOCAL_LLM_QA_MAX_NEW_TOKENS")
    local_llm_code_max_new_tokens: int = Field(default=256, alias="LOCAL_LLM_CODE_MAX_NEW_TOKENS")
    # Project/file generation needs more room than strict JSON planning. This is
    # used by the local code-engine path that creates full project structures
    # from NLP prompts without requiring JSON output.
    local_llm_code_project_max_new_tokens: int = Field(default=1024, alias="LOCAL_LLM_CODE_PROJECT_MAX_NEW_TOKENS")
    local_llm_json_max_new_tokens: int = Field(default=192, alias="LOCAL_LLM_JSON_MAX_NEW_TOKENS")
    local_llm_max_input_tokens: int = Field(default=2048, alias="LOCAL_LLM_MAX_INPUT_TOKENS")
    local_llm_empty_cache_after_generate: bool = Field(default=False, alias="LOCAL_LLM_EMPTY_CACHE_AFTER_GENERATE")

    # Local code engine controls. This path asks the local model for normal
    # manifest/file-content text instead of strict JSON so it can create and edit
    # real project files from natural-language prompts.
    local_code_engine_enabled: bool = Field(default=True, alias="LOCAL_CODE_ENGINE_ENABLED")
    local_code_engine_max_files: int = Field(default=12, alias="LOCAL_CODE_ENGINE_MAX_FILES")
    local_code_engine_max_context_files: int = Field(default=24, alias="LOCAL_CODE_ENGINE_MAX_CONTEXT_FILES")
    local_code_engine_max_file_chars: int = Field(default=4500, alias="LOCAL_CODE_ENGINE_MAX_FILE_CHARS")
    local_code_engine_repair_enabled: bool = Field(default=True, alias="LOCAL_CODE_ENGINE_REPAIR_ENABLED")
    # Validate each generated file before writing it. This catches generic bad
    # local-model output such as shell commands inside .py files, README prose in
    # tests, or git/pip commands in requirements/ignore files.
    local_code_engine_validate_files: bool = Field(default=True, alias="LOCAL_CODE_ENGINE_VALIDATE_FILES")
    local_code_engine_file_retries: int = Field(default=2, alias="LOCAL_CODE_ENGINE_FILE_RETRIES")
    # Legacy demo templates are opt-in only. The default code path uses the
    # prompt-driven local code engine for every project type instead of
    # hardcoding hello-world/calculator projects.
    local_code_engine_deterministic_examples_enabled: bool = Field(
        default=False, alias="LOCAL_CODE_ENGINE_DETERMINISTIC_EXAMPLES_ENABLED"
    )
    # Keep the old one-shot local file-block/JSON fallbacks disabled by default.
    # They can turn code-fence guesses or commands into files such as file_1.java
    # or python main.py when a small local model drifts.
    local_code_engine_allow_legacy_nlp_fallback: bool = Field(
        default=False, alias="LOCAL_CODE_ENGINE_ALLOW_LEGACY_NLP_FALLBACK"
    )
    # For create-project runs under sample_projects, remove failed generated
    # artifacts from previous attempts before applying the new validated files.
    local_code_engine_clean_new_project_artifacts: bool = Field(
        default=True, alias="LOCAL_CODE_ENGINE_CLEAN_NEW_PROJECT_ARTIFACTS"
    )
    local_code_engine_clean_new_project_anywhere: bool = Field(
        default=False, alias="LOCAL_CODE_ENGINE_CLEAN_NEW_PROJECT_ANYWHERE"
    )
    # Store compact lessons from failures/quality-gate rejections in .memory so
    # future local code runs can retrieve and avoid the same mistakes.
    local_code_learning_lessons_enabled: bool = Field(default=True, alias="LOCAL_CODE_LEARNING_LESSONS_ENABLED")
    local_code_failure_max_chars: int = Field(default=12000, alias="LOCAL_CODE_FAILURE_MAX_CHARS")


    # Prefer the largest available CUDA GPU when device=auto.
    local_llm_prefer_largest_gpu: bool = Field(default=True, alias="LOCAL_LLM_PREFER_LARGEST_GPU")
    # If model load fails due to CUDA OOM, retry with CPU offload via device_map=auto.
    local_llm_offload_if_oom: bool = Field(default=True, alias="LOCAL_LLM_OFFLOAD_IF_OOM")
    local_llm_max_gpu_mb: int = Field(default=11000, alias="LOCAL_LLM_MAX_GPU_MB")
    local_llm_max_cpu_mb: int = Field(default=24000, alias="LOCAL_LLM_MAX_CPU_MB")
    local_llm_offload_folder: Path = Field(
        default_factory=lambda: _default_data_dir() / "local_llm" / "offload",
        alias="LOCAL_LLM_OFFLOAD_FOLDER",
    )

    # ------------------------------
    # Background local training (LoRA)
    # ------------------------------
    local_training_enabled: bool = Field(default=False, alias="LOCAL_TRAINING_ENABLED")
    local_training_autostart: bool = Field(default=False, alias="LOCAL_TRAINING_AUTOSTART")
    local_training_poll_s: int = Field(default=900, alias="LOCAL_TRAINING_POLL_S")
    local_training_device: str = Field(default="cuda:0", alias="LOCAL_TRAINING_DEVICE")
    local_training_model: Optional[str] = Field(default=None, alias="LOCAL_TRAINING_MODEL")
    # Some models require remote code for training as well. Default False.
    local_training_trust_remote_code: bool = Field(default=False, alias="LOCAL_TRAINING_TRUST_REMOTE_CODE")
    # Upload-derived examples are often noisier than direct conversation/code pairs.
    # Keep them opt-in for the trainer by default.
    local_training_include_upload_examples: bool = Field(
        default=False, alias="LOCAL_TRAINING_INCLUDE_UPLOAD_EXAMPLES"
    )
    local_training_min_new_pairs: int = Field(default=25, alias="LOCAL_TRAINING_MIN_NEW_PAIRS")
    local_training_max_pairs: int = Field(default=1500, alias="LOCAL_TRAINING_MAX_PAIRS")
    local_training_max_steps: int = Field(default=200, alias="LOCAL_TRAINING_MAX_STEPS")
    # Accept both LOCAL_TRAINING_LEARNING_RATE (preferred) and the older LOCAL_TRAINING_LR.
    local_training_learning_rate: float = Field(
        default=5e-5,
        validation_alias=AliasChoices("LOCAL_TRAINING_LEARNING_RATE", "LOCAL_TRAINING_LR"),
    )
    local_training_lora_r: int = Field(default=16, alias="LOCAL_TRAINING_LORA_R")
    local_training_lora_alpha: int = Field(default=32, alias="LOCAL_TRAINING_LORA_ALPHA")
    local_training_lora_dropout: float = Field(default=0.05, alias="LOCAL_TRAINING_LORA_DROPOUT")
    local_training_save_every_steps: int = Field(default=200, alias="LOCAL_TRAINING_SAVE_EVERY_STEPS")
    # Keep the effective sequence length conservative by default so Qwen/Qwen2 LoRA
    # training fits on 12 GB consumer GPUs (for example an RTX 3060) even on Windows,
    # where bitsandbytes/4-bit training is often unavailable.
    local_training_max_seq_len: int = Field(default=512, alias="LOCAL_TRAINING_MAX_SEQ_LEN")
    # When BF16 is unavailable, still prefer loading the frozen base model in FP16 on
    # CUDA to reduce VRAM. LoRA trainable weights are promoted back to FP32 later.
    local_training_force_fp16_base: bool = Field(default=True, alias="LOCAL_TRAINING_FORCE_FP16_BASE")
    # Auto-enable gradient checkpointing on small GPUs even if the env file disabled it.
    local_training_force_gradient_checkpointing_on_small_gpu: bool = Field(
        default=True, alias="LOCAL_TRAINING_FORCE_GRADIENT_CHECKPOINTING_ON_SMALL_GPU"
    )
    # Skip samples that leave too little supervised target text after truncation.
    local_training_min_target_tokens: int = Field(default=8, alias="LOCAL_TRAINING_MIN_TARGET_TOKENS")
    # Conservative gradient clipping helps avoid NaN gradients on small/dirty datasets.
    local_training_max_grad_norm: float = Field(default=0.5, alias="LOCAL_TRAINING_MAX_GRAD_NORM")

    # Training numerics controls. Default to conservative settings because
    # mixed precision + gradient checkpointing can produce NaN gradients on
    # consumer GPUs with small LoRA runs.
    local_training_use_amp: bool = Field(default=False, alias="LOCAL_TRAINING_USE_AMP")
    local_training_bf16: bool = Field(default=False, alias="LOCAL_TRAINING_BF16")
    local_training_abort_on_nonfinite: bool = Field(
        default=True, alias="LOCAL_TRAINING_ABORT_ON_NONFINITE"
    )

    # GPU/CPU memory controls for training. These are used when loading a model
    # with CPU offload (device_map=auto) to keep VRAM usage within limits.
    local_training_gradient_checkpointing: bool = Field(
        default=True, alias="LOCAL_TRAINING_GRADIENT_CHECKPOINTING"
    )
    local_training_max_gpu_mb: int = Field(default=11000, alias="LOCAL_TRAINING_MAX_GPU_MB")
    local_training_max_cpu_mb: int = Field(default=24000, alias="LOCAL_TRAINING_MAX_CPU_MB")
    local_training_offload_folder: Path = Field(
        default_factory=lambda: _default_data_dir() / "local_llm" / "offload",
        alias="LOCAL_TRAINING_OFFLOAD_FOLDER",
    )

    # ------------------------------
    # Code-learning signals
    # ------------------------------
    # When enabled, successful code runs (software-factory pipeline and workspace
    # modules) will record small "prompt -> file content" examples into the
    # training_examples table. The background LoRA worker can then learn from
    # these examples so the local model improves at coding over time.
    code_training_enabled: bool = Field(default=True, alias="CODE_TRAINING_ENABLED")
    code_training_max_examples_per_run: int = Field(default=8, alias="CODE_TRAINING_MAX_EXAMPLES_PER_RUN")
    code_training_max_chars: int = Field(default=12000, alias="CODE_TRAINING_MAX_CHARS")

    # User feedback / RLHF-style learning. Feedback is stored immediately and
    # consumed by the manual local LoRA trainer with extra weight so corrections
    # influence the local model more strongly than passive chat logs.
    feedback_training_enabled: bool = Field(default=True, alias="FEEDBACK_TRAINING_ENABLED")
    local_training_feedback_weight: int = Field(default=4, alias="LOCAL_TRAINING_FEEDBACK_WEIGHT")

    # ------------------------------
    # Agent spawn limits
    # ------------------------------
    agent_max_per_type: int = Field(default=6, alias="AGENT_MAX_PER_TYPE")
    max_orchestrators: int = Field(default=2, alias="MAX_ORCHESTRATORS")
    agent_type_limits: Dict[str, int] = Field(default_factory=dict, alias="AGENT_TYPE_LIMITS")

    # ------------------------------
    # Web research
    # ------------------------------
    web_research_enabled: bool = Field(default=True, alias="WEB_RESEARCH_ENABLED")
    web_research_pool_size: int = Field(default=4, alias="WEB_RESEARCH_POOL_SIZE")
    web_search_max_results: int = Field(default=6, alias="WEB_SEARCH_MAX_RESULTS")
    web_fetch_top_n: int = Field(default=3, alias="WEB_FETCH_TOP_N")
    web_request_timeout_s: float = Field(default=12.0, alias="WEB_REQUEST_TIMEOUT_S")
    web_user_agent: str = Field(default="AgenticHiveBot/0.1", alias="WEB_USER_AGENT")
    # Local Q&A/code web research controls. Q&A remains opt-in; code mode is
    # enabled by default because coding prompts benefit from current docs and
    # examples. Failures are handled as optional context and never stop local
    # generation.
    local_qa_web_research_enabled: bool = Field(default=False, alias="LOCAL_QA_WEB_RESEARCH_ENABLED")
    local_code_web_research_enabled: bool = Field(default=True, alias="LOCAL_CODE_WEB_RESEARCH_ENABLED")
    local_code_web_research_max_queries: int = Field(default=3, alias="LOCAL_CODE_WEB_RESEARCH_MAX_QUERIES")
    local_code_web_research_timeout_s: float = Field(default=25.0, alias="LOCAL_CODE_WEB_RESEARCH_TIMEOUT_S")
    local_code_web_research_max_context_chars: int = Field(default=5000, alias="LOCAL_CODE_WEB_RESEARCH_MAX_CONTEXT_CHARS")
    local_code_web_research_llm_summary_enabled: bool = Field(default=False, alias="LOCAL_CODE_WEB_RESEARCH_LLM_SUMMARY_ENABLED")

    orchestrator_scale_threshold: int = Field(default=6, alias="ORCH_SCALE_THRESHOLD")
    orchestrator_query_batch_size: int = Field(default=4, alias="ORCH_QUERY_BATCH_SIZE")

    # ------------------------------
    # Timeouts
    # ------------------------------
    resource_call_timeout_s: float = Field(default=25.0, alias="RESOURCE_CALL_TIMEOUT_S")
    qa_call_timeout_s: float = Field(default=120.0, alias="QA_CALL_TIMEOUT_S")
    orchestrator_call_timeout_s: float = Field(default=180.0, alias="ORCHESTRATOR_CALL_TIMEOUT_S")
    llm_timeout_s: float = Field(default=60.0, alias="LLM_TIMEOUT_S")

    # Provider retry / cooldown controls for hosted backends.
    llm_backend_retry_attempts: int = Field(default=2, alias="LLM_BACKEND_RETRY_ATTEMPTS")
    llm_backend_retry_backoff_s: float = Field(default=1.0, alias="LLM_BACKEND_RETRY_BACKOFF_S")
    llm_backend_error_cooldown_s: float = Field(default=300.0, alias="LLM_BACKEND_ERROR_COOLDOWN_S")
    llm_backend_quota_cooldown_s: float = Field(default=1800.0, alias="LLM_BACKEND_QUOTA_COOLDOWN_S")

    # ------------------------------
    # Command Agent safety
    # ------------------------------
    command_allowlist: List[str] = Field(
        default_factory=lambda: [
            "python",
            "python3",
            "py",
            "pytest",
            "pip",
            "uvicorn",
            "git",
            "docker",
            "node",
            "npm",
            "npx",
            "go",
            "cargo",
        ],
        alias="COMMAND_ALLOWLIST",
    )
    docker_exec_disabled: bool = Field(default=True, alias="DOCKER_EXEC_DISABLED")

    # ------------------------------
    # Local reference projects
    # ------------------------------
    reference_project_roots: List[str] = Field(default_factory=list, alias="REFERENCE_PROJECT_ROOTS")
    reference_project_exclude_dirs: List[str] = Field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            ".venv",
            "venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            "dist",
            "build",
        ],
        alias="REFERENCE_PROJECT_EXCLUDE_DIRS",
    )

    # ------------------------------
    # Sequence learning
    # ------------------------------
    sequence_learning_enabled: bool = Field(default=True, alias="SEQUENCE_LEARNING_ENABLED")
    sequence_learning_scan_limit: int = Field(default=250, alias="SEQUENCE_LEARNING_SCAN_LIMIT")
    sequence_learning_max_traces: int = Field(default=8, alias="SEQUENCE_LEARNING_MAX_TRACES")
    sequence_learning_min_similarity: float = Field(default=0.08, alias="SEQUENCE_LEARNING_MIN_SIMILARITY")

    # ------------------------------
    # Pre-validators for common .env patterns
    # ------------------------------

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _v_cors_allow_origins(cls, v):
        return _parse_csv(v)

    @field_validator("command_allowlist", mode="before")
    @classmethod
    def _v_command_allowlist(cls, v):
        return _parse_csv(v)

    @field_validator("llm_fallback_backends", mode="before")
    @classmethod
    def _v_llm_fallback_backends(cls, v):
        return _parse_csv(v)

    @field_validator("llm_fallback_phrases", mode="before")
    @classmethod
    def _v_llm_fallback_phrases(cls, v):
        return _parse_csv(v)

    @field_validator("reference_project_roots", mode="before")
    @classmethod
    def _v_reference_project_roots(cls, v):
        return _parse_csv(v)

    @field_validator("reference_project_exclude_dirs", mode="before")
    @classmethod
    def _v_reference_project_exclude_dirs(cls, v):
        return _parse_csv(v)

    @field_validator("rag_skip_dirs", mode="before")
    @classmethod
    def _v_rag_skip_dirs(cls, v):
        return _parse_csv(v)

    @field_validator("agent_type_limits", mode="before")
    @classmethod
    def _v_agent_type_limits(cls, v):
        return _parse_kv_csv(v)

    def model_post_init(self, __context: object) -> None:
        # Normalize list-like env vars (comma-separated)
        self.cors_allow_origins = _parse_csv(self.cors_allow_origins)
        self.command_allowlist = _parse_csv(self.command_allowlist)
        self.llm_fallback_backends = _parse_csv(self.llm_fallback_backends)
        self.reference_project_roots = _parse_csv(self.reference_project_roots)
        self.reference_project_exclude_dirs = _parse_csv(self.reference_project_exclude_dirs)
        self.rag_skip_dirs = _parse_csv(self.rag_skip_dirs)
        self.agent_type_limits = _parse_kv_csv(self.agent_type_limits)

        # Resolve + ensure dirs
        self.workspace_root = Path(self.workspace_root).expanduser()
        self.memory_db_path = Path(self.memory_db_path).expanduser()
        self.uploads_dir = Path(self.uploads_dir).expanduser()
        self.local_llm_adapter_dir = Path(self.local_llm_adapter_dir).expanduser()
        self.local_training_offload_folder = Path(self.local_training_offload_folder).expanduser()
        self.local_llm_offload_folder = Path(self.local_llm_offload_folder).expanduser()
        if not self.workspace_root.is_absolute():
            self.workspace_root = (_backend_root() / self.workspace_root).resolve()
        if not self.memory_db_path.is_absolute():
            self.memory_db_path = (_backend_root() / self.memory_db_path).resolve()

        try:
            self.workspace_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            # workspace is best-effort; fallback handled elsewhere
            pass

        try:
            self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            self.uploads_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            self.local_training_offload_folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            self.local_llm_offload_folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Optional: support OPENAI_API_KEY_FILE
        if (not self.openai_api_key) and self.openai_api_key_file:
            try:
                p = Path(self.openai_api_key_file).expanduser()
                if p.exists():
                    key = p.read_text(encoding="utf-8").strip()
                    if key:
                        self.openai_api_key = key
            except Exception:
                pass

        # Optional: support GEMINI_API_KEY_FILE / GOOGLE_API_KEY_FILE
        if (not self.gemini_api_key) and self.gemini_api_key_file:
            try:
                p = Path(self.gemini_api_key_file).expanduser()
                if p.exists():
                    key = p.read_text(encoding="utf-8").strip()
                    if key:
                        self.gemini_api_key = key
            except Exception:
                pass

        # Optional: support ANTHROPIC_API_KEY_FILE
        if (not self.anthropic_api_key) and self.anthropic_api_key_file:
            try:
                p = Path(self.anthropic_api_key_file).expanduser()
                if p.exists():
                    key = p.read_text(encoding="utf-8").strip()
                    if key:
                        self.anthropic_api_key = key
            except Exception:
                pass

        # Optional: HF token
        if self.hf_hub_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", self.hf_hub_token)

        if self.hf_home:
            os.environ.setdefault("HF_HOME", str(Path(self.hf_home).expanduser()))


settings = Settings()
