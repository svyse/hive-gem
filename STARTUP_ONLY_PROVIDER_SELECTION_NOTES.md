# Startup-only provider selection

This build removes dashboard hot-swapping. The active provider is selected once when the FastAPI backend starts.

## Change provider

Edit `backend/.env`:

```env
LLM_BACKEND=local
```

Allowed values:

```env
LLM_BACKEND=local
LLM_BACKEND=gemini
LLM_BACKEND=openai
LLM_BACKEND=claude
```

Then fully restart the backend process. Do not rely only on the dashboard.

```powershell
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The dashboard shows the startup backend and model but does not switch providers at runtime.

## Local RTX 3060 path

The local runtime and trainer are configured to use the single visible CUDA device:

```env
CUDA_VISIBLE_DEVICES=0
LOCAL_LLM_DEVICE=cuda:0
LOCAL_TRAINING_DEVICE=cuda:0
```

Run this to verify CUDA visibility:

```powershell
.\scripts\check_local_cuda.ps1
```

Expected output includes:

```text
cuda_available: True
cuda:0: NVIDIA GeForce RTX 3060
```

## RLHF and training worker

The RLHF dashboard controls and stats remain enabled for Q&A and code-pipeline results. Positive feedback and corrections are stored as trainer-ready examples.

The supported training worker entrypoint is preserved:

```powershell
cd backend
python -m app.training.worker
```

The helper scripts still call that worker:

```powershell
.\scripts\manual_train_local.ps1 -Force
```

or:

```bash
./scripts/manual_train_local.sh --force
```
