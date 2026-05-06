# Hot-swap stale-run and backend-switch fix

This patch fixes the dashboard behavior where switching from local to Gemini/OpenAI could appear to time out and then snap back to local.

## Changes

- The dashboard-selected backend is now persisted in browser localStorage and is the routing source of truth for new Q&A and code requests.
- Q&A and code requests continue to send `backend` explicitly, so Gemini/OpenAI can be used even if the server default backend sync is delayed.
- The backend switch path is pure in-memory and does not synchronously touch local Torch/CUDA, model locks, provider SDKs, or runtime-state file writes. Runtime backend persistence happens in a daemon thread.
- Backend status refresh no longer overwrites the user-selected dropdown with a stale server default.
- Stale code-run polling now stops after repeated `404 run_id not found` responses, which commonly occurs after `uvicorn --reload` restarts and loses in-memory run records.

## Expected behavior

If the server default update is slow, the dashboard may temporarily show:

`Selected: gemini • server default: local`

New Q&A/code requests still route to Gemini because the request body includes `backend: "gemini"`.
