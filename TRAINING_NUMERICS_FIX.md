Training numerics fix

What changed
- disabled AMP by default for local LoRA training on CUDA (`LOCAL_TRAINING_USE_AMP=false`)
- disabled gradient checkpointing by default in the shipped `.env`
- lowered default learning rate to `5e-5`
- lowered default max grad norm to `0.5`
- added `LOCAL_TRAINING_ABORT_ON_NONFINITE=true`
- patched the trainer to fail fast on non-finite loss or gradients
- patched the trainer to log actual NaN/Inf losses instead of masking them as `0.0`
- added input-grad enabling for PEFT + gradient checkpointing, and auto-disables checkpointing if that cannot be enabled safely
- added logging for usable samples and target-token stats

Why this was happening
- the visible `loss: 0.0` was almost certainly a logging artifact from Transformers filtering NaN/Inf losses
- `grad_norm: nan` is the real signal that training numerics were unstable
- your live run was still using `learning_rate=0.0002`, so the prior code default was being overridden by `.env`
- the stack trace ending in `rotate_half` was where you interrupted the run, not the primary bug
