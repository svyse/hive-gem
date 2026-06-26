# Local backend repeated-token fix

This build adds safeguards for the local Hugging Face backend when the coding pipeline is run with `LLM_BACKEND=local`.

## What changed

- Added token-level `StoppingCriteria` during `model.generate()` so repeated token loops stop early instead of running to `max_new_tokens`.
- Added text-level guards for repeated punctuation, repeated words, repeated lines, and repeated phrase loops.
- Added safe JSON fallbacks for coding agents. If the local model collapses while generating a plan, review, tests, or Docker assets, the pipeline returns a no-op JSON plan instead of writing repeated-token output into files.
- Added write-file operation sanitization in the orchestrator so repeated-token content cannot be written to the project even if a bad JSON plan slips through.
- Added session-level LoRA adapter disabling. If a loaded adapter causes repeated-token collapse, it is disabled for the rest of the backend session and the base model will be reloaded on the next local call.
- Updated `backend/.env.example` with safer local decoding defaults.

## Recommended local settings

Copy these into `backend/.env` if your existing file does not already have them:

```env
LLM_BACKEND=local
LOCAL_LLM_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
LOCAL_LLM_DEVICE=auto
LOCAL_LLM_DO_SAMPLE=true
LOCAL_LLM_TEMPERATURE=0.25
LOCAL_LLM_TOP_P=0.90
LOCAL_LLM_TOP_K=40
LOCAL_LLM_REPETITION_PENALTY=1.25
LOCAL_LLM_NO_REPEAT_NGRAM_SIZE=6
LOCAL_LLM_REPEAT_STOP_ENABLED=true
LOCAL_LLM_REPEAT_STOP_MIN_TOKENS=20
LOCAL_LLM_REPEAT_TOKEN_LIMIT=10
LOCAL_LLM_REPEAT_NGRAM_LIMIT=4
LOCAL_LLM_REPEAT_MAX_NGRAM=8
LOCAL_LLM_CODE_MAX_NEW_TOKENS=384
LOCAL_LLM_JSON_MAX_NEW_TOKENS=192
```

If repetition started after local training, disable the adapter and restart the backend:

```env
LOCAL_LLM_DISABLE_ADAPTER=true
LOCAL_LLM_ENABLE_ADAPTER=false
```

## Files changed

- `backend/app/llm/local_hf_client.py`
- `backend/app/core/config.py`
- `backend/app/utils/repetition_guard.py`
- `backend/app/agents/orchestrator.py`
- `backend/app/agents/module_agent.py`
- `backend/app/agents/logic_agent.py`
- `backend/app/agents/testing_agent.py`
- `backend/app/agents/docker_agent.py`
- `backend/.env.example`
