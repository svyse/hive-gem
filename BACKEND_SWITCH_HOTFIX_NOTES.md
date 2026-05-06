# Backend switching hotfix

This patch fixes a hang that could happen when switching from `local` to `gemini` or `openai`.

## What was happening

The dashboard switch endpoint (`POST /api/model/backend`) rebuilt the LLM client immediately. The rebuild path instantiated every provider client, including `LocalHFClient`, even when the selected target was Gemini. Constructing `LocalHFClient` probes Torch/CUDA. On Windows/CUDA systems, especially while a local model is loading or generating, that CUDA probe can block before FastAPI finishes the request. Uvicorn only prints access logs after the request returns, so the terminal could show no POST line and look stuck.

## What changed

- Backend switching is now lazy and non-blocking.
- Switching to Gemini/OpenAI no longer imports/probes local Torch/CUDA.
- Backend status is lightweight and no longer constructs every provider.
- The switch endpoint logs when it receives and completes a switch request.
- The frontend adds a 15-second timeout to the switch request so the UI cannot spin forever.
- The dashboard no longer calls local CUDA status before the backend status has loaded.

## Expected log when switching

```text
Model backend switch request received: gemini
LLM backend switched: local -> gemini (lazy reload; next request builds provider)
Model backend switch request completed: active=gemini
INFO: 127.0.0.1:xxxxx - "POST /api/model/backend HTTP/1.1" 200 OK
```

Restart both backend and frontend after replacing files.
