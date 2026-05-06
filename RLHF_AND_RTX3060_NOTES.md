# RLHF Feedback + RTX 3060 Local Model Notes

This patch makes user feedback visible and functional in both dashboard modes.

## What changed

### Q&A mode

Every assistant answer now shows feedback controls:

- Good: stores a positive reward signal.
- Bad: stores a negative reward signal.
- Add correction: stores the preferred answer and queues it for the manual local trainer.

The feedback is written to the `model_feedback` table, added to hive memory, and corrected/positive examples are added to `training_examples`.

### Code pipeline mode

When a code run finishes and the result JSON is visible, the same feedback controls appear below the result.

For code feedback, the trainer receives the original project task, project path, optional user note, and the corrected/preferred response.

### Manual RL-style training

The local trainer now consumes explicit user feedback in addition to passive conversation and training examples. Feedback examples are weighted with:

```env
FEEDBACK_TRAINING_ENABLED=true
LOCAL_TRAINING_FEEDBACK_WEIGHT=4
```

This is an offline RLHF-style LoRA update path: feedback becomes reward-weighted preference/correction data and is applied when you manually run training. It intentionally does not mutate model weights immediately after each click, because online weight updates during normal chat can corrupt the model or cause CUDA OOM.

Run a manual training pass after collecting feedback:

```powershell
.\scripts\manual_train_local.ps1 -Force
```

or:

```bash
./scripts/manual_train_local.sh --force
```

The trainer writes adapter state under `LOCAL_LLM_ADAPTER_DIR`. The local backend hot-loads compatible LoRA adapters on new requests.

## Checking RTX 3060 usage

The dashboard now shows Local GPU status when the backend is `local`. It reports:

- whether CUDA is visible to PyTorch,
- which CUDA device was selected,
- GPU name and VRAM totals,
- whether the local model is loaded,
- trainer device settings and feedback weight.

There is also a backend endpoint:

```text
GET /api/local/status
```

Expected for an RTX 3060 system:

```text
cuda_available: true
resolved_device: cuda:0 or the CUDA index for the RTX 3060
cuda_devices[...].name: NVIDIA GeForce RTX 3060
```

If the dashboard says CUDA is not visible, install a CUDA-enabled PyTorch build in the backend environment. `Active: local` only selects the local provider; CUDA visibility depends on PyTorch, NVIDIA driver, and the Python environment.

## RTX 3060 safety settings

The included `.env` keeps local generation and training conservative for a 12 GB RTX 3060:

```env
LOCAL_LLM_DEVICE=auto
LOCAL_LLM_PREFER_LARGEST_GPU=true
LOCAL_LLM_OFFLOAD_IF_OOM=true
LOCAL_LLM_MAX_GPU_MB=10240
LOCAL_LLM_MAX_INPUT_TOKENS=3072
LOCAL_LLM_QA_MAX_NEW_TOKENS=256
LOCAL_LLM_CODE_MAX_NEW_TOKENS=384
LOCAL_LLM_JSON_MAX_NEW_TOKENS=256
LOCAL_LLM_INITIAL_TIMEOUT_S=300

LOCAL_TRAINING_DEVICE=auto
LOCAL_TRAINING_MAX_SEQ_LEN=512
LOCAL_TRAINING_GRADIENT_CHECKPOINTING=true
LOCAL_TRAINING_FORCE_GRADIENT_CHECKPOINTING_ON_SMALL_GPU=true
LOCAL_TRAINING_FORCE_FP16_BASE=true
LOCAL_TRAINING_MAX_GPU_MB=10240
```

On Windows, `bitsandbytes` 4-bit loading is often unavailable. The backend will still try to use GPU FP16/offload for `Qwen/Qwen2.5-Coder-1.5B-Instruct`.

## Quick CUDA check script

Before starting the backend, run:

```powershell
.\scripts\check_local_cuda.ps1
```

or:

```bash
./scripts/check_local_cuda.sh
```

Use the same Python/virtualenv that runs FastAPI. CUDA must be visible there, not just in another Python installation.
