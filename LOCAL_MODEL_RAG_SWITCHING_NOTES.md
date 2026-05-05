# Local model, RAG, dashboard switching, and sample_projects changes

## Dashboard hot swap

The dashboard now has a **Model backend** selector for `Local`, `Gemini`, and `OpenAI`. It calls:

- `GET /api/model/backend` for status
- `POST /api/model/backend` with `{ "backend": "local|gemini|openai" }` to switch providers

The switch is process-local and affects new Q&A/code requests without restarting FastAPI. Existing in-flight runs keep the client they already started with.

## sample_projects source of truth

User-entered paths now resolve as follows:

- `hello` -> `<repo>/sample_projects/hello`
- `sample_projects/hello` -> `<repo>/sample_projects/hello`
- `~.\sample_projects\hello` -> `<repo>/sample_projects/hello`

In code mode, missing sample project folders are created automatically. `sample_projects` are always edited in place and are not copied into `.workspace` / `.workspaces`; those folders remain for logs and runtime metadata.

## RAG

A dependency-free lexical RAG layer now ranks relevant current project files plus hive/type/agent memory and injects that context into both Q&A and code-pipeline prompts. Tunables are in `backend/.env`:

```env
RAG_ENABLED=true
RAG_MEMORY_SCAN_LIMIT=350
RAG_MEMORY_LIMIT=8
RAG_PROJECT_FILE_LIMIT=10
RAG_MAX_CHARS=9000
RAG_MAX_FILE_BYTES=200000
```

## Local RTX 3060 OOM safeguards

The local HF client now defaults to GPU-preferred, 4-bit where available, smaller output caps, prompt token caps, CUDA fragmentation mitigation, and a generation retry with fewer tokens on CUDA OOM.

Important settings:

```env
LOCAL_LLM_DEVICE=auto
LOCAL_LLM_LOAD_IN_4BIT=true
LOCAL_LLM_MAX_GPU_MB=10240
LOCAL_LLM_MAX_INPUT_TOKENS=3072
LOCAL_LLM_QA_MAX_NEW_TOKENS=256
LOCAL_LLM_CODE_MAX_NEW_TOKENS=384
LOCAL_QA_WEB_RESEARCH_ENABLED=false
LOCAL_CODE_WEB_RESEARCH_ENABLED=false
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If bitsandbytes is unavailable on Windows, the app logs a warning and falls back to FP16/offload instead of failing immediately.

## Manual local training

After collecting conversations/code runs from Gemini or OpenAI, run one manual LoRA training pass:

PowerShell:

```powershell
.\scripts\manual_train_local.ps1 -Force
```

Bash:

```bash
./scripts/manual_train_local.sh --force
```

Training reads from the configured memory database and writes the latest adapter under `LOCAL_LLM_ADAPTER_DIR`. The live local client hot-reloads the adapter on later requests.

## Speech-to-text improvements

The frontend speech control now defaults to `en-IN`, lets you switch recognition language, requests multiple browser alternatives, applies code/path dictation fixes, and keeps the existing learned correction loop.

## API keys

`backend/.env` in this returned archive has API key values blanked. Re-add your own `OPENAI_API_KEY` and `GEMINI_API_KEY` locally before using hosted providers.
