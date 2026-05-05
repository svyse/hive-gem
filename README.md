# Agentic Hive Studio

React UI + FastAPI backend for a multi-agent "hive" that can:

- run a **software-factory pipeline** (plan → apply edits → test/compile → Docker artifacts)
- answer questions via a **multi-orchestrator Q&A hive** (domain routers + specialist agents)
- persist **agent/type/hive memories** and **multi-turn conversations** in SQLite

This fork adds:

- a **local Hugging Face Transformers LLM backend** (default)
- optional **OpenAI / Gemini / Claude hosted backends** with ordered provider fallbacks
- optional **background LoRA fine-tuning** that learns from your app’s stored data
- an **uploads feature (documents + images)** that is usable by both local and OpenAI backends and can also be added into the training set
- **workspace modules**: small reusable tools stored under `WORKSPACE_ROOT/modules` that can be created/listed/run by the hive

---

Create a project hello_py, the project prompts the user for their name and then responds with "Greetings" and the name provided. The entire project has to run in a jupyter notebook, set up the ipykernel and venv accordingly

## What changed in this version

### ✅ Local Transformers LLM (default)
The backend supports a local Hugging Face model via `transformers`:

- Inference runs through `backend/app/llm/local_hf_client.py`
- Default: `LLM_BACKEND=local`
- Optional: `LLM_BACKEND=openai`, `LLM_BACKEND=gemini`, or `LLM_BACKEND=claude`
- Optional fallback chain: `OpenAI -> Gemini -> Claude -> local` (or any order you configure)
- **Agent memories still work the same way** (memories are injected into prompts regardless of backend)

### ✅ Upload documents + images
You can upload PDFs/docs/text/images from the UI.

Uploads are:

- saved to disk under `~/.agentic_hive_studio/uploads` (configurable)
- extracted into text (best-effort)
- chunked and added to **hive memory** so both backends can use them
- optionally converted into **training examples** for the background LoRA worker

### ✅ Optional background fine-tuning (LoRA)
An optional training worker (`python -m app.training.worker`) can fine-tune a LoRA adapter from:

- stored conversations
- uploaded-file training examples

The adapter is saved under a model-specific subdirectory rooted at `LOCAL_LLM_ADAPTER_DIR` and is automatically loaded by the local HF client.

In addition to Q&A conversations and uploads, the hive also records **coding artifacts** (e.g., file
contents written by the software-factory pipeline and workspace modules) into the training examples
table. This helps the local model gradually learn your coding style and common project patterns.

---

## Prerequisites

### Required
- **Python 3.10+**
- **Node 18+** (for the React/Vite frontend)

### Recommended (GPU)
- NVIDIA driver installed
- **CUDA-enabled PyTorch**

Notes:
- PyTorch wheels typically ship with their own CUDA runtime. Your **NVIDIA driver** must be new enough for the wheel you install.
- If you have **CUDA Toolkit 11.7** installed (`nvcc` reports 11.7) but `nvidia-smi` shows a newer CUDA runtime (e.g. 12.x), that's normal: **the driver version is what matters** for PyTorch wheels.
- On Windows, `bitsandbytes` is often unavailable; the app automatically disables 4-bit quantization if it cannot import it.

---

## API keys / secrets

You can run the app **fully locally with no API keys**.

Optional keys:
- `OPENAI_API_KEY` — only required if `LLM_BACKEND=openai` or `openai` appears in `LLM_FALLBACK_BACKENDS`
- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) — only required if `LLM_BACKEND=gemini` or `gemini` appears in `LLM_FALLBACK_BACKENDS`
- `ANTHROPIC_API_KEY` — only required if `LLM_BACKEND=claude` or `claude` appears in `LLM_FALLBACK_BACKENDS`
- `HUGGINGFACE_HUB_TOKEN` — only required if you use a **gated** HF model

---

## Quickstart

### 1) Backend

```bash
cd backend
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# macOS/Linux:
#   source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env if you want to change the local model, enable training, etc.

uvicorn app.main:app --host localhost --port 8000 --reload --host 0.0.0.0 


pip install torch==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torchvision==0.15.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Backend: `http://localhost:8000`

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:5173`

The frontend proxies `/api/*` to the backend (see `frontend/vite.config.js`).

---

## Local LLM configuration

The backend reads config from `backend/.env` (see `backend/.env.example`).

Common settings:

- `LLM_BACKEND=local` (default), `LLM_BACKEND=openai`, `LLM_BACKEND=gemini`, or `LLM_BACKEND=claude`
- `LLM_FALLBACK_BACKENDS=openai,gemini,claude,local` (the selected primary backend is skipped automatically)
- `LOCAL_LLM_MODEL=Qwen/Qwen2.5-Coder-3B-Instruct` (recommended default)
- `LOCAL_LLM_DEVICE=auto` (or `cuda:0`, `cuda:1`, `cpu`)
- `LOCAL_LLM_PREFER_LARGEST_GPU=true` (use the biggest-VRAM GPU when multiple GPUs exist)

### Hosted model selection (optional)

If you enable a hosted provider (either by setting `LLM_BACKEND` to it or including it in
`LLM_FALLBACK_BACKENDS`), you can route models like this:

**OpenAI**
- `OPENAI_MODEL=gpt-5.2` (default primary)
- `OPENAI_FALLBACK_MODEL=gpt-4.1` (fallback if the OpenAI primary model errors)
- `LLM_BACKEND_QUOTA_COOLDOWN_S=1800` temporarily skips OpenAI after a `429 insufficient_quota`, so the app can move on to Gemini / Claude / local instead of burning retries on every request
- `OPENAI_MODEL_QA=...` overrides the model for Q&A agents
- `OPENAI_MODEL_CODE=...` overrides the model for the software-factory code pipeline

Per your request, **Codex is optional** — if `OPENAI_MODEL_CODE` is unset, the code pipeline uses
`OPENAI_MODEL`.

**Gemini**
- `GEMINI_MODEL=gemini-2.5-flash`
- `GEMINI_FALLBACK_MODEL=gemini-2.5-flash-lite` (recommended within-provider fallback)
- `GEMINI_MODEL_QA=...` and `GEMINI_MODEL_CODE=...`
- `LLM_BACKEND_RETRY_ATTEMPTS=2` and `LLM_BACKEND_RETRY_BACKOFF_S=1.0` retry transient 429/5xx overloads before failing over

**Claude / Anthropic**
- `CLAUDE_MODEL=claude-sonnet-4-20250514`
- `CLAUDE_FALLBACK_MODEL=claude-3-5-haiku-latest` (recommended within-provider fallback)
- `CLAUDE_MODEL_QA=...` and `CLAUDE_MODEL_CODE=...`
- `CLAUDE_MAX_TOKENS=1024`
- Gemini and Claude now log the HTTP error body for 400/429/5xx failures so you can see whether the issue is model access, payload shape, or transient overload

Additional useful toggles:

- `LOCAL_LLM_TRUST_REMOTE_CODE=false` (default)
  - Set this to `true` *only* for models that ship custom Python code in the HF repo.
- `LOCAL_LLM_TIMEOUT_S=180` and `LOCAL_LLM_INITIAL_TIMEOUT_S=240` (recommended for CPU local models)
  - Separate timeout for local calls (model load/download + generate). CPU cold starts for Qwen-sized models can take longer than a hosted API call, so a larger timeout avoids unnecessary fallbacks. If a call still times out and `LLM_FALLBACK_TO_OPENAI=true`, the app retries with the next provider in `LLM_FALLBACK_BACKENDS`.
  - The local HF loader now uses a first-load mutex so concurrent agent calls do not try to load the same model twice. This matters for generic STEM prompts, which can spawn more than one QA agent on the first request.
- `QA_CALL_TIMEOUT_S=240` and `ORCHESTRATOR_CALL_TIMEOUT_S=300` are safer defaults when local CPU fallback is enabled. The orchestrator now stretches its internal timeout when the local backend is still in cold-start mode so domain agents do not get marked as unavailable just because the model is still warming up.

### Prompt-length safety

Small local chat models often have ~2K context. The local HF client automatically truncates
overly-long prompts from the **left** (keeps the most recent context) to avoid
`Token indices sequence length ... > max` errors.

### Offload for big models / shared GPU
If a model fails to load due to CUDA OOM, the local backend can retry with CPU offload:

- `LOCAL_LLM_OFFLOAD_IF_OOM=true`
- `LOCAL_LLM_MAX_GPU_MB=11000`
- `LOCAL_LLM_MAX_CPU_MB=24000`

This uses HF `device_map="auto"` with a max-memory budget, splitting weights between GPU + system RAM.

---

## Uploads (documents/images)

### UI
In **QA mode**, use the **Upload documents / images** box.

- Default: uploads are added to hive memory **and** added to training examples
- You can toggle:
  - **Add to hive memory**
  - **Use for training**

### Supported formats (best-effort)
- Text: `.txt`, `.md`
- PDF: `.pdf` (via `pypdf`)
- Word: `.docx` (via `python-docx`)
- Images: `.png`, `.jpg`, `.jpeg` (captioned locally via a small BLIP model)

### How uploads are used
- The extracted text is chunked and stored in **hive memory**, so it is usable by:
  - the local HF model
  - OpenAI (if you switch backends)
- Training examples are inserted into the `training_examples` table so the background LoRA worker can learn from them.

---

## Workspace Modules (tools under `.workspace`)

The backend can create and manage small **workspace modules** (tiny utilities/apps) under:

`WORKSPACE_ROOT/modules/<module_name>/`

These modules are **outside the git repo** and persist across runs. They are useful for:

- reusable utilities (calculator, format converters, script runners)
- user-requested tools that orchestrators/agents can discover and invoke

### Built-in modules
On backend startup, these are created if missing:

- `calculator` — safe arithmetic expression evaluator (stdlib-only)
- `script_runner` — runs a Python script that lives under `WORKSPACE_ROOT` (useful for running things created in `WORKSPACE_ROOT/runs/...` like a snake game)

### Use from the chat (QA mode)

Ask things like:

- `list modules`
- `create module calculator`
- `run module calculator --expr 2+2`
- `create a module named my_tool that prints hello and accepts --name`

The Q&A orchestrator auto-adds the `workspace_module` agent for these requests.

### Use via API

- `GET /api/modules` — list modules
- `POST /api/modules` — create a module
- `POST /api/modules/{name}/run` — run a module (non-interactive; has a timeout)

### Use via CLI

```bash
cd backend
python -m app.workspace_modules.cli list
python -m app.workspace_modules.cli run calculator -- --expr "(1+2)*3"
python -m app.workspace_modules.cli run script_runner -- --latest --pattern "snake*.py"
```

### Configuration

```ini
# backend/.env
WORKSPACE_MODULE_RUN_TIMEOUT_S=60
```

If you plan to run **interactive** tools (like terminal games), prefer using the CLI.

### Dependencies (auto-install)

Workspace modules (and generated projects under `.workspace/runs/...`) can declare Python dependencies.

- Module dependencies can be recorded in:
  - `WORKSPACE_ROOT/modules/<name>/requirements.txt`
  - and in `module.json` as `requirements: [...]`
- If a module or project fails with `ModuleNotFoundError`, the orchestrator will attempt to:
  1) infer the missing package
  2) run `python -m pip install ...`
  3) append the package to `requirements.txt` (project + workspace)

Control this with:

```ini
# backend/.env
AUTO_INSTALL_DEPS=true
PIP_INSTALL_TIMEOUT_S=600
```

> Security note: auto-install only accepts **simple PyPI requirement strings** (no URLs / editable installs) by default.

## Background local training (LoRA)

### Option A: run the worker manually

```bash
cd backend
# activate venv
python -m app.training.worker
```

### Option B: auto-start training from the backend
In `backend/.env`:

```ini
LOCAL_TRAINING_ENABLED=true
LOCAL_TRAINING_AUTOSTART=true
LOCAL_TRAINING_MIN_NEW_PAIRS=5
```

The backend spawns the worker in a **separate process**, keeping the FastAPI server and UI responsive. The worker now inherits the parent process logs instead of writing into an unread pipe, which avoids background-training stalls. When the target model changes, the adapter and training-state files are isolated per base model.

### Code learning from agent work

If you want the local model to learn from the hive's **coding artifacts** (files written by the
software-factory pipeline and workspace modules), keep this enabled:

```ini
CODE_TRAINING_ENABLED=true
CODE_TRAINING_MAX_EXAMPLES_PER_RUN=8
CODE_TRAINING_MAX_CHARS=12000
```

### Splitting training between GPU + RAM (recommended on 12GB)
Training supports CPU offload to reduce VRAM usage:

```ini
# Memory budgets (MiB)
LOCAL_TRAINING_MAX_GPU_MB=10000
LOCAL_TRAINING_MAX_CPU_MB=24000
LOCAL_TRAINING_GRADIENT_CHECKPOINTING=true
LOCAL_TRAINING_MAX_SEQ_LEN=512
LOCAL_TRAINING_FORCE_FP16_BASE=true
```

---

## Optional: provider fallback chain

If you want hosted-first with Gemini and Claude before local:

```ini
LLM_BACKEND=openai
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
LLM_FALLBACK_TO_OPENAI=true
LLM_FALLBACK_BACKENDS=openai,gemini,claude,local
```

If you want local-first with hosted escape hatches:

```ini
LLM_BACKEND=local
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
LLM_FALLBACK_TO_OPENAI=true
LLM_FALLBACK_BACKENDS=openai,gemini,claude,local
```

Fallback triggers when:
- the current provider errors
- the current provider times out (see `LOCAL_LLM_TIMEOUT_S` / `LLM_TIMEOUT_S`)
- the current provider outputs a clear "can’t answer" message (configurable phrases)

The fallback heuristic is applied to both:
- `chat_text(_async)` outputs, and
- `chat_json(_async)` outputs (which is the main path used by the Q&A agents)

See `LLM_FALLBACK_*` in `backend/.env.example`.

---

## API overview

- `POST /api/runs` — start a code run
- `GET /api/runs/{run_id}` — run status + logs + agent statuses

- `POST /api/qa/ask` — Q&A (multi-turn via `conversation_id`)
- `POST /api/qa/conversations` — create conversation
- `GET /api/qa/conversations` — list conversations
- `GET /api/qa/conversations/{conversation_id}` — fetch full thread

- `POST /api/uploads` — upload a document/image (multipart)
- `GET /api/uploads` — list uploads
- `GET /api/uploads/{upload_id}` — upload details

- `GET /api/memory/hive/search?q=...`
- `GET /api/memory/type/{agent_type}/search?q=...`
- `GET /api/memory/all/search?q=...`

---

## Safety note

The **Command Agent** can write files and run commands. This repo uses a workspace sandbox and a command allowlist (`COMMAND_ALLOWLIST`), but you should still run it only on projects you trust (ideally in an isolated environment).
