# Dashboard hot-swap + RLHF stats fix

This patch fixes two dashboard-level issues:

1. **Backend hot-swap while local is busy**
   - The backend selector is no longer disabled just because a Q&A/code request is running.
   - Switching is optimistic in the UI, so the selected provider changes immediately.
   - The backend client cache now uses an epoch and builds clients outside the global lock, so a slow local CUDA/Torch path cannot block the `/api/model/backend` switch endpoint.
   - Fallback providers are lazy. Listing `local` as a fallback no longer imports Torch/probes CUDA when Gemini/OpenAI is selected.
   - The selected dashboard backend is persisted to `.memory/runtime_backend.json`, so uvicorn reloads do not reset the UI back to the `.env` value.

2. **RLHF feedback/stats visibility**
   - The RLHF stats card now has manual refresh and automatic 5-second polling.
   - Feedback buttons update the visible stats optimistically and then reconcile from `/api/feedback/stats`.
   - Feedback is saved even when the frontend/backend cannot recover the original prompt. Such rows still count as reward signals and hive memory; they are skipped as supervised training pairs until a usable prompt/correction exists.
   - Code-pipeline feedback controls now appear for failed runs too, not only successful JSON results.
   - Run status responses now include the original code prompt and input mode, so code feedback can be tied back to the exact request.

## Quick checks

After restarting backend and frontend, use PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/model/backend" -ContentType "application/json" -Body '{"backend":"gemini"}'
Invoke-RestMethod "http://127.0.0.1:8000/api/model/backend"
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/feedback" -ContentType "application/json" -Body '{"target_type":"qa","target_id":"manual-test","score":1,"prompt":"test prompt","response":"test response","backend":"gemini"}'
Invoke-RestMethod "http://127.0.0.1:8000/api/feedback/stats"
```

Expected:
- backend status shows `active_backend: gemini`
- feedback stats total increments

A currently running local request may continue on local. Hot-swap means **new** Q&A/code requests use the selected backend without restarting the app.
