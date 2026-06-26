# Note on local-only build history

This project was previously packaged as a local-only build while the dashboard hot-swap issue was being removed.

The current build uses **startup-only provider selection** instead:

```env
LLM_BACKEND=local
# or gemini / openai / claude, followed by a FastAPI restart
```

Dashboard hot-swapping remains disabled. See `STARTUP_ONLY_PROVIDER_SELECTION_NOTES.md` for current setup instructions.
