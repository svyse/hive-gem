# Local Qwen working setup

Use this after switching local Q&A to `Qwen/Qwen2.5-1.5B-Instruct`.

## Files

- `backend.env` -> copy to `backend/.env`
- `prefetch_local_models.py` -> copy to `backend/scripts/prefetch_local_models.py`
- `prefetch_local_models.ps1` -> copy to `backend/scripts/prefetch_local_models.ps1`

## Why this is needed

Your log shows the first Q&A request was still downloading Hugging Face model files:
- tokenizer download took about 94 seconds
- `model.safetensors` is 3.09 GB and had just started downloading
- the Q&A request returned before the full model download/load completed

Prefetching moves the download outside the Q&A request.

## Commands

From PowerShell:

```powershell
cd C:\Users\shera\Desktop\gem_v9\backend
copy <downloaded>\backend.env .env

mkdir scripts -Force
copy <downloaded>\prefetch_local_models.py scripts\prefetch_local_models.py
copy <downloaded>\prefetch_local_models.ps1 scripts\prefetch_local_models.ps1

.\.my_rlhf_v6\Scripts\Activate.ps1
.\scripts\prefetch_local_models.ps1

# After download completes:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
