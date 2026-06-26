from __future__ import annotations

import inspect
import json
import logging
import os
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Maintain the supported training entrypoint: python -m app.training.worker
# Mask to the single RTX 3060-visible CUDA device before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch


from app.core.config import settings
from app.llm.adapter_paths import resolve_model_adapter_dir, resolve_model_adapter_root, resolve_training_state_file


log = logging.getLogger(__name__)


def _from_pretrained_with_dtype(loader: Any, model_id: str, *, dtype: Any, **kwargs: Any) -> Any:
    """Prefer the newer `dtype=` kwarg, but keep compatibility with older Transformers."""

    try:
        sig = inspect.signature(loader.from_pretrained)
        if "dtype" in sig.parameters:
            return loader.from_pretrained(model_id, dtype=dtype, **kwargs)
    except Exception:
        pass

    try:
        return loader.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError as e:
        msg = str(e)
        if "dtype" not in msg and "unexpected keyword" not in msg:
            raise
    return loader.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


DEFAULT_SYSTEM_PROMPT = (
    "You are Agentic Hive Studio, a helpful assistant and software engineer. "
    "You can answer questions and write correct, runnable code when asked. "
    "Follow the user's instructions carefully."
)



_ERRORISH_TRAINING_PHRASES = (
    "internal error in",
    "an internal error occurred",
    "internal error while answering",
    "please check the run log for details",
    "could not parse json from model output",
    "q&a agent output must be json object",
)


def _include_upload_examples() -> bool:
    return bool(getattr(settings, "local_training_include_upload_examples", False))


def _feedback_training_enabled() -> bool:
    return bool(getattr(settings, "feedback_training_enabled", True))


def _feedback_weight() -> int:
    try:
        return max(1, min(8, int(getattr(settings, "local_training_feedback_weight", 4) or 4)))
    except Exception:
        return 4


def _looks_bad_training_text(text: Any) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return True
    return any(p in lowered for p in _ERRORISH_TRAINING_PHRASES)


def _pair_key(a: str, b: str) -> tuple[str, str]:
    norm_a = re.sub(r"\s+", " ", str(a or "").strip().lower())
    norm_b = re.sub(r"\s+", " ", str(b or "").strip().lower())
    return norm_a, norm_b


def _assistant_meta_excludes_training(meta_raw: Any) -> bool:
    try:
        meta = json.loads(meta_raw or "{}")
        if not isinstance(meta, dict):
            return False
    except Exception:
        return False
    if bool(meta.get("exclude_from_training")):
        return True
    result = meta.get("result")
    if isinstance(result, dict) and _looks_bad_training_text(result.get("answer")):
        return True
    return False


def _current_db_max_ids() -> tuple[int, int, int]:
    db_path = Path(settings.memory_db_path).expanduser()
    if not db_path.exists():
        return 0, 0, 0
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        cur = conn.cursor()
        row1 = cur.execute("SELECT COALESCE(MAX(id), 0) FROM conversation_messages").fetchone()
        row2 = cur.execute("SELECT COALESCE(MAX(id), 0) FROM training_examples").fetchone()
        try:
            row3 = cur.execute("SELECT COALESCE(MAX(id), 0) FROM model_feedback").fetchone()
        except sqlite3.OperationalError:
            row3 = [0]
        conn.close()
        return (
            int((row1 or [0])[0] or 0),
            int((row2 or [0])[0] or 0),
            int((row3 or [0])[0] or 0),
        )
    except Exception:
        return 0, 0, 0


def _normalize_state_ids(last_message_id: int, last_example_id: int, last_feedback_id: int = 0) -> tuple[int, int, int]:
    max_msg, max_ex, max_fb = _current_db_max_ids()
    if int(last_message_id or 0) > max_msg:
        last_message_id = 0
    if int(last_example_id or 0) > max_ex:
        last_example_id = 0
    if int(last_feedback_id or 0) > max_fb:
        last_feedback_id = 0
    return int(last_message_id or 0), int(last_example_id or 0), int(last_feedback_id or 0)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _backend_root() -> Path:
    # backend/app/training/worker.py -> .../backend
    return Path(__file__).resolve().parents[2]


def _target_model_id() -> str:
    return str(getattr(settings, "local_training_model", None) or settings.local_llm_model)


def _adapters_root(model_id: str | None = None) -> Path:
    return resolve_model_adapter_root(settings.local_llm_adapter_dir, model_id or _target_model_id())


def _training_state_file(model_id: str | None = None) -> Path:
    return resolve_training_state_file(settings.local_llm_adapter_dir, model_id or _target_model_id())


def _load_training_state_payload() -> Dict[str, Any]:
    p = _training_state_file()
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}



def _load_training_state() -> tuple[int, int, int]:
    """Return (last_message_id, last_training_example_id, last_feedback_id)."""

    data = _load_training_state_payload()
    try:
        last_msg = int(data.get("last_message_id") or 0)
        last_ex = int(data.get("last_training_example_id") or 0)
        last_fb = int(data.get("last_feedback_id") or 0)
        return _normalize_state_ids(last_msg, last_ex, last_fb)
    except Exception:
        return 0, 0, 0


def _write_state(*, last_message_id: int, last_training_example_id: int, last_feedback_id: int, stats: Dict[str, Any]) -> None:
    p = _training_state_file()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "last_message_id": int(last_message_id),
                    "last_training_example_id": int(last_training_example_id),
                    "last_feedback_id": int(last_feedback_id),
                    "updated_at": _now_iso(),
                    "stats": stats,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        log.warning("Failed writing training state: %s", e)



def _cuda_index(device: str | None) -> int:
    try:
        d = str(device or "cuda:0")
        if ":" in d:
            return int(d.split(":", 1)[1])
    except Exception:
        pass
    return 0


def _pick_largest_cuda_device() -> str | None:
    try:
        if not torch.cuda.is_available():
            return None
        best = 0
        best_mem = -1
        for i in range(torch.cuda.device_count()):
            try:
                mem = int(torch.cuda.get_device_properties(i).total_memory)
            except Exception:
                mem = 0
            if mem > best_mem:
                best_mem = mem
                best = i
        return f"cuda:{best}"
    except Exception:
        return None


def _cuda_supports_bf16(device: str | None) -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability(_cuda_index(device))
        return int(major) >= 8
    except Exception:
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return False


def _cuda_total_memory_mb(device: str | None) -> int:
    try:
        if not torch.cuda.is_available():
            return 0
        idx = _cuda_index(device)
        total = int(torch.cuda.get_device_properties(idx).total_memory)
        return max(0, total // (1024 * 1024))
    except Exception:
        return 0


def _resolve_device(device: str | None) -> str:
    d = (device or "auto").strip().lower()

    if d in {"", "auto"}:
        picked = _pick_largest_cuda_device()
        return picked or "cpu"

    if d in {"cuda", "gpu"}:
        picked = _pick_largest_cuda_device()
        return picked or "cuda:0"

    return device or "cpu"


def _has_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _has_peft() -> bool:
    try:
        import peft  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _count_new_items(last_message_id: int, last_example_id: int, last_feedback_id: int = 0) -> int:
    """Count new training signals since the last state.

    - assistant messages from conversations
    - rows from training_examples (uploads are opt-in)
    - model_feedback rows with corrections or positive reward
    """

    db_path = Path(settings.memory_db_path).expanduser()
    if not db_path.exists():
        return 0

    try:
        last_message_id, last_example_id, last_feedback_id = _normalize_state_ids(
            last_message_id, last_example_id, last_feedback_id
        )
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        row1 = cur.execute(
            """
            SELECT COUNT(*) AS n
            FROM conversation_messages
            WHERE id > ?
              AND role = 'assistant'
              AND content NOT LIKE 'Internal error in%'
              AND content NOT LIKE 'Internal error while answering%'
              AND content NOT LIKE 'An internal error occurred%'
            """,
            (int(last_message_id),),
        ).fetchone()
        if _include_upload_examples():
            row2 = cur.execute(
                "SELECT COUNT(*) AS n FROM training_examples WHERE id > ?",
                (int(last_example_id),),
            ).fetchone()
        else:
            row2 = cur.execute(
                "SELECT COUNT(*) AS n FROM training_examples WHERE id > ? AND lower(COALESCE(source, '')) != 'upload'",
                (int(last_example_id),),
            ).fetchone()
        row3 = {"n": 0}
        if _feedback_training_enabled():
            try:
                row3 = cur.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM model_feedback
                    WHERE id > ?
                      AND TRIM(COALESCE(prompt, '')) != ''
                      AND (
                        TRIM(COALESCE(corrected_response, '')) != ''
                        OR (reward > 0 AND TRIM(COALESCE(response, '')) != '')
                      )
                    """,
                    (int(last_feedback_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                row3 = {"n": 0}
        conn.close()
        n1 = int(row1["n"] if row1 else 0)
        n2 = int(row2["n"] if row2 else 0)
        n3 = int(row3["n"] if row3 else 0)
        return n1 + n2 + n3
    except Exception:
        return 0



def _load_recent_conversation_pairs(*, max_pairs: int) -> Tuple[List[Tuple[str, str]], int]:
    """Build (user, assistant) training pairs from conversation_messages.

    Returns:
      (pairs, max_assistant_message_id)
    """

    db_path = Path(settings.memory_db_path).expanduser()
    if not db_path.exists():
        return [], 0

    # Heuristic: pull more rows than pairs to account for user+assistant and other roles.
    limit_rows = max(200, int(max_pairs) * 6)

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        rows = cur.execute(
            """
            SELECT id, conversation_id, role, content, meta
            FROM conversation_messages
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit_rows),),
        ).fetchall()
    except sqlite3.OperationalError:
        # DB exists but schema not initialized yet
        conn.close()
        return [], 0
    conn.close()

    if not rows:
        return [], 0

    # Chronological
    rows = list(reversed(rows))

    last_user_by_conv: Dict[str, Tuple[int, str]] = {}
    pairs: List[Tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    max_assistant_id = 0

    for r in rows:
        mid = int(r["id"])
        conv = str(r["conversation_id"] or "")
        role = str(r["role"] or "").strip().lower()
        content = str(r["content"] or "").strip()
        if not conv or not role or not content:
            continue

        if role == "user":
            if _looks_bad_training_text(content):
                continue
            last_user_by_conv[conv] = (mid, content)
            continue

        if role == "assistant":
            if _looks_bad_training_text(content):
                continue
            if _assistant_meta_excludes_training(r["meta"]):
                continue
            prev = last_user_by_conv.get(conv)
            if not prev:
                continue
            _uid, utext = prev
            # Basic hygiene: avoid training on enormous tool dumps.
            if len(utext) > 4000 or len(content) > 8000:
                continue
            if _looks_bad_training_text(utext):
                continue
            key = _pair_key(utext, content)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((utext, content))
            max_assistant_id = max(max_assistant_id, mid)

    # Keep only the most recent pairs
    if len(pairs) > max_pairs:
        pairs = pairs[-max_pairs:]

    return pairs, max_assistant_id




def _load_recent_training_examples(*, max_pairs: int) -> Tuple[List[Tuple[str, str]], int]:
    """Load (prompt, completion) pairs from training_examples.

    Returns:
      (pairs, max_training_example_id)
    """

    db_path = Path(settings.memory_db_path).expanduser()
    if not db_path.exists():
        return [], 0

    # Pull more than needed to allow for filtering.
    limit_rows = max(200, int(max_pairs) * 3)

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        rows = cur.execute(
            """
            SELECT id, source, prompt, completion
            FROM training_examples
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit_rows),),
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return [], 0
    conn.close()

    if not rows:
        return [], 0

    rows = list(reversed(rows))

    pairs: List[Tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    max_ex_id = 0
    include_uploads = _include_upload_examples()

    for r in rows:
        ex_id = int(r["id"])
        source = str(r["source"] or "").strip().lower()
        prompt = str(r["prompt"] or "").strip()
        completion = str(r["completion"] or "").strip()
        if source == "upload" and not include_uploads:
            continue
        if not prompt or not completion:
            continue
        if _looks_bad_training_text(prompt) or _looks_bad_training_text(completion):
            continue
        if len(prompt) > 4000 or len(completion) > 8000:
            continue
        key = _pair_key(prompt, completion)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((prompt, completion))
        max_ex_id = max(max_ex_id, ex_id)

    if len(pairs) > max_pairs:
        pairs = pairs[-max_pairs:]

    return pairs, max_ex_id




def _load_recent_feedback_pairs(*, max_pairs: int) -> Tuple[List[Tuple[str, str]], int]:
    """Load preferred (prompt, completion) pairs from explicit user feedback.

    Corrections are preferred over original model responses. Positive feedback without
    a correction uses the original model response. Negative-only feedback is retained
    as a reward signal in the DB/hive memory, but is not used as a supervised target.
    """

    if not _feedback_training_enabled():
        return [], 0

    db_path = Path(settings.memory_db_path).expanduser()
    if not db_path.exists():
        return [], 0

    limit_rows = max(200, int(max_pairs) * 3)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        rows = cur.execute(
            """
            SELECT id, target_type, prompt, response, corrected_response, comment, project_path, reward
            FROM model_feedback
            WHERE TRIM(COALESCE(prompt, '')) != ''
              AND (
                TRIM(COALESCE(corrected_response, '')) != ''
                OR (reward > 0 AND TRIM(COALESCE(response, '')) != '')
              )
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit_rows),),
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return [], 0
    conn.close()

    if not rows:
        return [], 0

    rows = list(reversed(rows))
    pairs: List[Tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    max_feedback_id = 0

    try:
        from app.feedback.rl import training_prompt_for_feedback
    except Exception:
        training_prompt_for_feedback = None  # type: ignore[assignment]

    for r in rows:
        fb_id = int(r["id"])
        target_type = str(r["target_type"] or "qa").strip().lower()
        prompt = str(r["prompt"] or "").strip()
        corrected = str(r["corrected_response"] or "").strip()
        response = str(r["response"] or "").strip()
        comment = str(r["comment"] or "").strip()
        project_path = str(r["project_path"] or "").strip()
        try:
            reward = float(r["reward"] or 0.0)
        except Exception:
            reward = 0.0
        completion = corrected or (response if reward > 0 else "")
        if not prompt or not completion:
            continue
        if _looks_bad_training_text(prompt) or _looks_bad_training_text(completion):
            continue
        if len(prompt) > 8000 or len(completion) > 12000:
            continue
        if training_prompt_for_feedback is not None:
            try:
                prompt_for_training = training_prompt_for_feedback(
                    target_type=target_type,
                    prompt=prompt,
                    comment=comment,
                    project_path=project_path,
                )
            except Exception:
                prompt_for_training = prompt
        else:
            prompt_for_training = prompt
        if _looks_bad_training_text(prompt_for_training):
            continue
        key = _pair_key(prompt_for_training, completion)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((prompt_for_training, completion))
        max_feedback_id = max(max_feedback_id, fb_id)

    if len(pairs) > max_pairs:
        pairs = pairs[-max_pairs:]

    return pairs, max_feedback_id



def _trailing_pairs_window(pairs: List[Tuple[str, str]], length: int) -> List[Tuple[str, str]]:
    if length <= 0 or not pairs:
        return []
    return pairs[-min(int(length), len(pairs)):]



def _build_weighted_rolling_window(
    pairs: List[Tuple[str, str]], *, base_length: int
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """Compose the requested rolling-window mix.

    Vector A is the rolling window length calculated by the existing logic.
    Vector B is the most recent ceil(A / 2) pairs.
    Vector C is the most recent ceil(A / 3) pairs.
    Vector D is the concatenation A + B + C so newer pairs are repeated more often.
    """

    available = len(pairs)
    a_len = min(max(0, int(base_length)), available)
    if a_len <= 0:
        return [], {
            "available_pairs": available,
            "vector_a_pairs": 0,
            "vector_b_pairs": 0,
            "vector_c_pairs": 0,
            "vector_d_pairs": 0,
        }

    b_len = min((a_len + 1) // 2, available)
    c_len = min((a_len + 2) // 3, available)

    vector_a = _trailing_pairs_window(pairs, a_len)
    vector_b = _trailing_pairs_window(pairs, b_len)
    vector_c = _trailing_pairs_window(pairs, c_len)
    vector_d = vector_a + vector_b + vector_c

    return vector_d, {
        "available_pairs": available,
        "vector_a_pairs": len(vector_a),
        "vector_b_pairs": len(vector_b),
        "vector_c_pairs": len(vector_c),
        "vector_d_pairs": len(vector_d),
    }


def _guess_lora_targets(model: Any) -> List[str]:
    """Heuristic LoRA target modules by model_type / module names."""

    model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()
    if model_type in {"llama", "mistral", "qwen2", "gemma", "phi3", "phi", "falcon"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if model_type in {"gpt2"}:
        return ["c_attn", "c_proj"]

    # Fallback: inspect names
    names = [n for n, _ in model.named_modules()]
    common = ["q_proj", "v_proj", "k_proj", "o_proj"]
    if any(any(n.endswith(s) for s in common) for n in names):
        return common
    return ["c_attn", "c_proj"]


@dataclass
class _Example:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]



class _PairsDataset(torch.utils.data.Dataset):
    def __init__(self, *, pairs: List[Tuple[str, str]], tokenizer: Any, system_prompt: str, max_len: int | None = None) -> None:
        self.tok = tokenizer
        self.system = system_prompt
        try:
            tokenizer_cap = int(getattr(tokenizer, "model_max_length", 2048) or 2048)
        except Exception:
            tokenizer_cap = 2048
        configured_max_len = int(max_len or getattr(settings, "local_training_max_seq_len", 512) or 512)
        self.max_len = int(max(128, min(tokenizer_cap, configured_max_len, 2048)))
        self.min_target_tokens = max(1, int(getattr(settings, "local_training_min_target_tokens", 8) or 8))
        self.examples: List[Dict[str, Any]] = []
        for user, assistant in pairs:
            item = self._build_item(user=user, assistant=assistant)
            if item is not None:
                self.examples.append(item)

    def __len__(self) -> int:
        return len(self.examples)

    def _render(self, user: str, assistant: str) -> Tuple[str, str]:
        tok = self.tok
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            prompt_only = tok.apply_chat_template(
                [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            full = tok.apply_chat_template(
                [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            return prompt_only, full

        prompt_only = f"System: {self.system}\nUser: {user}\nAssistant:"
        full = prompt_only + f" {assistant}"
        return prompt_only, full

    def _build_item(self, *, user: str, assistant: str) -> Optional[Dict[str, Any]]:
        prompt_only, full = self._render(user, assistant)

        old_truncation_side = getattr(self.tok, "truncation_side", "right")
        try:
            self.tok.truncation_side = "left"
            prompt_ids = self.tok(
                prompt_only,
                truncation=True,
                max_length=self.max_len,
                add_special_tokens=True,
            )["input_ids"]

            enc = self.tok(
                full,
                truncation=True,
                max_length=self.max_len,
                add_special_tokens=True,
            )
        finally:
            try:
                self.tok.truncation_side = old_truncation_side
            except Exception:
                pass
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask") or [1] * len(input_ids)

        labels = list(input_ids)
        mask_upto = min(len(labels), len(prompt_ids))
        for i in range(mask_upto):
            labels[i] = -100

        target_tokens = sum(1 for x in labels if x != -100)
        if target_tokens < self.min_target_tokens:
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "target_tokens": target_tokens,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def _dataset_target_stats(dataset: Any) -> Dict[str, float]:
    target_counts: List[int] = []

    try:
        items = getattr(dataset, "examples", None)
        if items is None:
            items = [dataset[i] for i in range(len(dataset))]
    except Exception:
        items = []

    for item in items:
        try:
            n = int(item.get("target_tokens") if isinstance(item, dict) else 0)
        except Exception:
            n = 0
        if n <= 0 and isinstance(item, dict):
            try:
                labels = item.get("labels") or []
                n = sum(1 for x in labels if int(x) != -100)
            except Exception:
                n = 0
        if n > 0:
            target_counts.append(n)

    if not target_counts:
        return {"count": 0.0, "min": 0.0, "avg": 0.0, "max": 0.0}

    return {
        "count": float(len(target_counts)),
        "min": float(min(target_counts)),
        "avg": float(sum(target_counts) / len(target_counts)),
        "max": float(max(target_counts)),
    }


def _enable_input_require_grads(model: Any) -> bool:
    """Best-effort helper for PEFT + gradient checkpointing.

    Some HF/PEFT model stacks need input embeddings to require grads, otherwise
    checkpointed LoRA training can produce no trainable gradients.
    """

    try:
        fn = getattr(model, "enable_input_require_grads", None)
        if callable(fn):
            fn()
            return True
    except Exception:
        pass

    try:
        emb = model.get_input_embeddings()
    except Exception:
        emb = None
    if emb is None:
        return False

    try:
        old_handle = getattr(model, "_agentic_input_grad_hook", None)
        if old_handle is not None:
            try:
                old_handle.remove()
            except Exception:
                pass

        def _mark_requires_grad(_module: Any, _inputs: Any, output: Any) -> Any:
            try:
                if isinstance(output, tuple):
                    first = output[0]
                    if hasattr(first, "requires_grad_"):
                        first.requires_grad_(True)
                    return output
                if hasattr(output, "requires_grad_"):
                    output.requires_grad_(True)
            except Exception:
                pass
            return output

        model._agentic_input_grad_hook = emb.register_forward_hook(_mark_requires_grad)
        return True
    except Exception:
        return False


def _collate(tokenizer: Any, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    attn = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=int(pad_id))
    attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def _configure_logging() -> None:
    level = (settings.log_level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def main() -> int:
    _configure_logging()

    if not _has_peft():
        log.error("PEFT is not installed. Install `peft` to enable local LoRA training.")
        return 2

    state = _load_training_state_payload()
    last_msg_id, last_ex_id, last_fb_id = _load_training_state()
    last_msg_id, last_ex_id, last_fb_id = _normalize_state_ids(last_msg_id, last_ex_id, last_fb_id)
    min_new = max(1, int(getattr(settings, "local_training_min_new_pairs", 25) or 25))
    n_new = _count_new_items(last_msg_id, last_ex_id, last_fb_id)

    previous_pairs_used = 0
    previous_base_pairs_used = 0
    try:
        stats = state.get("stats") if isinstance(state, dict) else {}
        if isinstance(stats, dict):
            previous_pairs_used = int(stats.get("pairs_used") or 0)
            previous_base_pairs_used = int(
                stats.get("rolling_window_base_pairs")
                or stats.get("base_pairs_used")
                or stats.get("pairs_used")
                or 0
            )
    except Exception:
        previous_pairs_used = 0
        previous_base_pairs_used = 0

    max_pairs = max(50, int(getattr(settings, "local_training_max_pairs", 1500) or 1500))
    conv_pairs, max_assistant_id = _load_recent_conversation_pairs(max_pairs=max_pairs)
    ex_pairs, max_example_id = _load_recent_training_examples(max_pairs=max_pairs)
    feedback_pairs, max_feedback_id = _load_recent_feedback_pairs(max_pairs=max_pairs)
    feedback_weight = _feedback_weight() if feedback_pairs else 1
    weighted_feedback_pairs: List[Tuple[str, str]] = []
    for pair in feedback_pairs:
        weighted_feedback_pairs.extend([pair] * feedback_weight)

    all_pairs = conv_pairs + ex_pairs + weighted_feedback_pairs
    if len(all_pairs) > max_pairs:
        all_pairs = all_pairs[-max_pairs:]

    if len(all_pairs) < 10:
        log.info("Not enough training pairs to train (found=%s)", len(all_pairs))
        return 0

    base_target_pairs = max(10, previous_base_pairs_used or min_new)
    base_target_pairs = min(base_target_pairs, max_pairs)
    pairs, rolling_window_stats = _build_weighted_rolling_window(all_pairs, base_length=base_target_pairs)

    force_train = os.getenv("FORCE_TRAIN", "").strip().lower() in {"1", "true", "yes"}
    if n_new < min_new and not force_train:
        if previous_base_pairs_used > 0:
            log.info(
                "Fewer than %s new items (%s); retraining on rolling windows A=%s, B=%s, C=%s -> D=%s pairs.",
                min_new,
                n_new,
                rolling_window_stats["vector_a_pairs"],
                rolling_window_stats["vector_b_pairs"],
                rolling_window_stats["vector_c_pairs"],
                rolling_window_stats["vector_d_pairs"],
            )
        else:
            log.info(
                "Fewer than %s new items (%s); no prior training window recorded, training on rolling windows A=%s, B=%s, C=%s -> D=%s pairs.",
                min_new,
                n_new,
                rolling_window_stats["vector_a_pairs"],
                rolling_window_stats["vector_b_pairs"],
                rolling_window_stats["vector_c_pairs"],
                rolling_window_stats["vector_d_pairs"],
            )
    elif force_train:
        log.info(
            "FORCE_TRAIN enabled; training on rolling windows A=%s, B=%s, C=%s -> D=%s pairs.",
            rolling_window_stats["vector_a_pairs"],
            rolling_window_stats["vector_b_pairs"],
            rolling_window_stats["vector_c_pairs"],
            rolling_window_stats["vector_d_pairs"],
        )

    device = _resolve_device(getattr(settings, "local_training_device", "auto"))
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    model_id = getattr(settings, "local_training_model", None) or settings.local_llm_model

    if use_cuda:
        try:
            idx = _cuda_index(device)
            props = torch.cuda.get_device_properties(idx)
            torch.cuda.set_device(idx)
            log.info(
                "Local training CUDA selected: %s | %s | total_vram=%s MiB",
                device,
                getattr(props, "name", "CUDA GPU"),
                int(getattr(props, "total_memory", 0) or 0) // (1024 * 1024),
            )
        except Exception as e:
            log.warning("Training CUDA was selected but diagnostics/setup failed for %s: %s", device, e)
    else:
        log.warning(
            "Local training will run on CPU (configured=%s, resolved=%s, cuda_available=%s).",
            getattr(settings, "local_training_device", "auto"),
            device,
            bool(torch.cuda.is_available()),
        )

    log.info(
        "Training local LoRA adapter | model=%s | device=%s | weighted_pairs=%s | base_window=%s | A=%s | B=%s | C=%s | recent_pairs=%s (conv=%s, extra=%s, feedback_raw=%s, feedback_weight=%s, include_upload_examples=%s, new_items=%s, previous_pairs_used=%s, previous_base_pairs_used=%s)",
        model_id,
        device,
        len(pairs),
        base_target_pairs,
        rolling_window_stats["vector_a_pairs"],
        rolling_window_stats["vector_b_pairs"],
        rolling_window_stats["vector_c_pairs"],
        len(all_pairs),
        len(conv_pairs),
        len(ex_pairs),
        len(feedback_pairs),
        feedback_weight,
        _include_upload_examples(),
        n_new,
        previous_pairs_used,
        previous_base_pairs_used,
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer

    class _SingleDeviceTrainer(Trainer):
        """Keep training on the already-selected CUDA device instead of DataParallel.

        Transformers Trainer will automatically wrap the model in torch.nn.DataParallel
        whenever more than one CUDA device is visible. In mixed setups such as an RTX
        3060 plus a GTX 1650, that causes replication onto the smaller card and can OOM
        even when the main model fits on the 3060.
        """

        def _move_model_to_device(self, model, device):  # type: ignore[override]
            if getattr(self.args, "_force_single_device", False):
                return model
            return super()._move_model_to_device(model, device)

        def _wrap_model(self, model, training=True, dataloader=None):  # type: ignore[override]
            if getattr(self.args, "_force_single_device", False):
                return model
            return super()._wrap_model(model, training=training, dataloader=dataloader)

        def _prepare_input(self, data):  # type: ignore[override]
            selected_device = getattr(self.args, "_selected_device", None)
            if selected_device is not None and isinstance(data, torch.Tensor):
                try:
                    kwargs = {"device": selected_device}
                    if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                        kwargs["dtype"] = self.args.hf_deepspeed_config.dtype()
                    return data.to(**kwargs)
                except Exception:
                    pass
            return super()._prepare_input(data)

        def training_step(self, model, inputs):  # type: ignore[override]
            loss = super().training_step(model, inputs)
            if bool(getattr(settings, "local_training_abort_on_nonfinite", True)):
                try:
                    if isinstance(loss, torch.Tensor) and not torch.isfinite(loss.detach()).all():
                        raise RuntimeError(
                            "Non-finite training loss encountered. The forward pass produced NaN/Inf loss; use BF16 or FP32 base weights and ensure training inputs are on the selected GPU."
                        )
                except RuntimeError:
                    raise
                except Exception:
                    pass
                saw_grad = False
                bad_name = None
                for name, param in model.named_parameters():
                    if not getattr(param, "requires_grad", False):
                        continue
                    grad = getattr(param, "grad", None)
                    if grad is None:
                        continue
                    saw_grad = True
                    try:
                        if not torch.isfinite(grad.detach()).all():
                            bad_name = name
                            break
                    except Exception:
                        continue
                if bad_name is not None:
                    raise RuntimeError(f"Non-finite gradient detected in trainable parameter: {bad_name}")
                if not saw_grad:
                    raise RuntimeError(
                        "No trainable gradients were produced. For LoRA training this usually means gradient checkpointing is on without input grads enabled."
                    )
            return loss

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    trust_remote = bool(
        getattr(settings, "local_training_trust_remote_code", False)
        or getattr(settings, "local_llm_trust_remote_code", False)
    )

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    requested_fp16 = bool(use_cuda and getattr(settings, "local_training_use_amp", False))
    requested_bf16 = bool(use_cuda and getattr(settings, "local_training_bf16", False))
    bf16_supported = bool(use_cuda and _cuda_supports_bf16(device))
    auto_bf16 = bool(use_cuda and not requested_fp16 and not requested_bf16 and bf16_supported)
    if requested_bf16 and not bf16_supported:
        log.warning("BF16 requested but not supported on %s; falling back to FP16 or FP32 base weights.", device)
    if auto_bf16:
        log.info("Auto-enabling BF16 for local LoRA training on %s for better stability than pure FP16.", device)
    use_bf16 = bool((requested_bf16 or auto_bf16) and bf16_supported)
    use_fp16 = bool(requested_fp16 and not use_bf16)
    force_fp16_base = bool(use_cuda and getattr(settings, "local_training_force_fp16_base", True))
    auto_fp16_base = bool(use_cuda and not use_bf16 and force_fp16_base)
    if auto_fp16_base and not use_fp16:
        log.info(
            "Using FP16 base weights for local LoRA training on %s to stay within consumer GPU VRAM; LoRA trainable weights stay FP32.",
            device,
        )
    base_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if (use_fp16 or auto_fp16_base) else torch.float32)
    use_amp = bool(use_fp16 or use_bf16)

    quant_cfg = None
    if use_cuda and _has_bitsandbytes():
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=base_dtype,
            )
        except Exception as e:
            log.warning("4-bit quantization unavailable; falling back to non-quantized load: %s", e)
            quant_cfg = None

    load_kwargs = {}
    offload_kwargs = {}
    cuda_idx = 0
    if use_cuda:
        try:
            cuda_idx = _cuda_index(device)
        except Exception:
            cuda_idx = 0

        # Prefer keeping the whole trainable model on the selected GPU.
        # Only fall back to CPU offload if the initial load OOMs.
        load_kwargs = {"device_map": {"": device}}

        try:
            gpu_mb = int(getattr(settings, "local_training_max_gpu_mb", 11000) or 11000)
            cpu_mb = int(getattr(settings, "local_training_max_cpu_mb", 24000) or 24000)
        except Exception:
            gpu_mb, cpu_mb = 11000, 24000

        offload_dir = Path(
            getattr(
                settings,
                "local_training_offload_folder",
                Path(settings.local_llm_adapter_dir).expanduser().parent / "offload",
            )
        ).expanduser()
        try:
            offload_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        offload_kwargs = {
            "device_map": "auto",
            "max_memory": {cuda_idx: f"{gpu_mb}MiB", "cpu": f"{cpu_mb}MiB"},
            "offload_folder": str(offload_dir),
            "offload_state_dict": True,
            "low_cpu_mem_usage": True,
        }

    def _load_model(*, offload: bool) -> Any:
        kwargs = dict(offload_kwargs if (offload and use_cuda) else load_kwargs)
        if quant_cfg is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=trust_remote,
                quantization_config=quant_cfg,
                **kwargs,
            )
            model = prepare_model_for_kbit_training(model)
            return model

        model = _from_pretrained_with_dtype(
            AutoModelForCausalLM,
            model_id,
            trust_remote_code=trust_remote,
            dtype=base_dtype,
            **kwargs,
        )
        if use_cuda and not kwargs.get("device_map"):
            model.to(device)
        elif not use_cuda:
            model.to(device)
        return model

    try:
        model = _load_model(offload=False)
    except Exception as e:
        msg = str(e)
        is_oom = use_cuda and ("out of memory" in msg.lower() or ("cuda" in msg.lower() and "oom" in msg.lower()))
        if is_oom:
            log.warning(
                "CUDA OOM while loading %s for training on %s; retrying with CPU offload.",
                model_id,
                device,
            )
            model = _load_model(offload=True)
        else:
            raise

    try:
        tok_vocab = int(len(tok))
        emb = model.get_input_embeddings()
        model_vocab = int(getattr(getattr(emb, "weight", None), "shape", [0])[0] or 0)
        if tok_vocab > 0 and model_vocab > 0 and tok_vocab != model_vocab:
            log.info("Resizing token embeddings to match tokenizer vocab: %s -> %s", model_vocab, tok_vocab)
            model.resize_token_embeddings(tok_vocab)
    except Exception as e:
        log.warning("Failed to verify/resize token embeddings: %s", e)

    try:
        if getattr(model, "config", None) is not None and getattr(tok, "pad_token_id", None) is not None:
            model.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass

    # Gradient checkpointing reduces VRAM usage at the cost of training speed.
    configured_gc = bool(getattr(settings, "local_training_gradient_checkpointing", True))
    small_gpu_mb = _cuda_total_memory_mb(device)
    force_gc_small_gpu = bool(
        use_cuda
        and bool(getattr(settings, "local_training_force_gradient_checkpointing_on_small_gpu", True))
        and small_gpu_mb > 0
        and small_gpu_mb <= 12288
    )
    if force_gc_small_gpu and not configured_gc:
        log.info("Auto-enabling gradient checkpointing on %s (%s MiB VRAM) to avoid CUDA OOM during training.", device, small_gpu_mb)
    use_gc = bool(configured_gc or force_gc_small_gpu)
    if use_gc:
        enabled_input_grads = _enable_input_require_grads(model)
        if not enabled_input_grads:
            log.warning(
                "Could not enable input grads for gradient checkpointing; disabling gradient checkpointing for stability."
            )
            use_gc = False
    if use_gc:
        try:
            sig = inspect.signature(model.gradient_checkpointing_enable)
            if "gradient_checkpointing_kwargs" in sig.parameters:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                model.gradient_checkpointing_enable()
        except Exception:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                use_gc = False
        try:
            if getattr(model, "config", None) is not None:
                model.config.use_cache = False
        except Exception:
            pass

    targets = _guess_lora_targets(model)
    lora_cfg = LoraConfig(
        r=int(getattr(settings, "local_training_lora_r", 16) or 16),
        lora_alpha=int(getattr(settings, "local_training_lora_alpha", 32) or 32),
        lora_dropout=float(getattr(settings, "local_training_lora_dropout", 0.05) or 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_cfg)

    if use_gc:
        _enable_input_require_grads(model)

    # PEFT documents that mixing a model loaded in fp16 with Trainer(fp16=True) can
    # raise "Attempting to unscale FP16 gradients". Keep trainable LoRA weights in
    # fp32 and let AMP handle the mixed-precision forward/backward pass. This is
    # especially important on consumer GPUs such as an RTX 3060.
    if use_cuda:
        try:
            torch.cuda.set_device(cuda_idx)
        except Exception as e:
            log.warning("Failed to set active CUDA device to %s: %s", device, e)
        try:
            for param in model.parameters():
                if getattr(param, "requires_grad", False):
                    param.data = param.data.float()
        except Exception as e:
            log.warning("Failed to promote trainable adapter params to fp32: %s", e)

        if torch.cuda.device_count() > 1:
            log.info(
                "Multiple CUDA devices are visible (%s total); forcing single-device training on %s to avoid DataParallel OOM on smaller GPUs.",
                torch.cuda.device_count(),
                device,
            )

    train_max_len = int(getattr(settings, "local_training_max_seq_len", 512) or 512)
    if use_cuda and train_max_len > 512 and _cuda_total_memory_mb(device) <= 12288 and quant_cfg is None:
        log.info(
            "Capping local training sequence length to 512 tokens on %s to avoid Qwen/Qwen2 logits OOM on a 12 GB GPU.",
            device,
        )
        train_max_len = 512
    dataset = _PairsDataset(pairs=pairs, tokenizer=tok, system_prompt=DEFAULT_SYSTEM_PROMPT, max_len=train_max_len)
    if len(dataset) < 10:
        log.info("Not enough usable training samples after filtering/truncation (usable=%s, raw_pairs=%s)", len(dataset), len(pairs))
        return 0

    stats = _dataset_target_stats(dataset)
    log.info(
        "Usable training samples=%s | max_seq_len=%s | target_tokens min/avg/max=%.0f/%.1f/%.0f | rolling windows A=%s, B=%s, C=%s -> D=%s",
        int(stats["count"]),
        train_max_len,
        stats["min"],
        stats["avg"],
        stats["max"],
        rolling_window_stats["vector_a_pairs"],
        rolling_window_stats["vector_b_pairs"],
        rolling_window_stats["vector_c_pairs"],
        rolling_window_stats["vector_d_pairs"],
    )

    max_steps = int(getattr(settings, "local_training_max_steps", 200) or 200)
    lr = float(getattr(settings, "local_training_learning_rate", 5e-5) or 5e-5)
    max_grad_norm = float(getattr(settings, "local_training_max_grad_norm", 0.5) or 0.5)
    log.info(
        "Training numerics | lr=%s | max_grad_norm=%s | use_amp=%s | use_bf16=%s | use_fp16=%s | model_dtype=%s | gradient_checkpointing=%s",
        lr,
        max_grad_norm,
        use_amp,
        use_bf16,
        use_fp16,
        str(base_dtype).replace('torch.', ''),
        use_gc,
    )

    out_root = _adapters_root(model_id)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / ("run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))

    arg_kwargs = dict(
        output_dir=str(run_dir),
        disable_tqdm=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=max_steps,
        learning_rate=lr,
        warmup_steps=0,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=bool(use_cuda),
        fp16=bool(use_fp16),
        bf16=bool(use_bf16),
        max_grad_norm=max_grad_norm,
    )
    try:
        sig = inspect.signature(TrainingArguments.__init__)
        if "logging_nan_inf_filter" in sig.parameters:
            arg_kwargs["logging_nan_inf_filter"] = False
        if "optim" in sig.parameters:
            arg_kwargs["optim"] = "adamw_torch"
    except Exception:
        pass

    args = TrainingArguments(**arg_kwargs)

    if use_cuda and device.startswith("cuda:"):
        try:
            args._n_gpu = 1  # type: ignore[attr-defined]
            setattr(args, "_force_single_device", True)
            setattr(args, "_selected_device", torch.device(device))
        except Exception:
            pass

    trainer = _SingleDeviceTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda batch: _collate(tok, batch),
    )

    trainer.train()

    # Save adapter (PEFT) + tokenizer for reproducibility.
    model.save_pretrained(str(run_dir))
    try:
        tok.save_pretrained(str(run_dir))
    except Exception:
        pass

    # Atomically update the "latest" adapter directory.
    latest = resolve_model_adapter_dir(settings.local_llm_adapter_dir, model_id)
    tmp = latest.with_name("latest_tmp")
    try:
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(run_dir, tmp)
        if latest.exists():
            shutil.rmtree(latest)
        tmp.rename(latest)
    except Exception as e:
        log.warning("Failed updating latest adapter dir (%s): %s", latest, e)

    train_stats = {
        "pairs_used": len(pairs),
        "pairs_used_conv": len(conv_pairs),
        "pairs_used_extra": len(ex_pairs),
        "pairs_used_feedback_raw": len(feedback_pairs),
        "feedback_weight": feedback_weight,
        "feedback_training_enabled": _feedback_training_enabled(),
        "include_upload_examples": _include_upload_examples(),
        "new_items_since_last": n_new,
        "rolling_window_pairs": len(pairs),
        "rolling_window_base_pairs": rolling_window_stats["vector_a_pairs"],
        "rolling_window_vector_a_pairs": rolling_window_stats["vector_a_pairs"],
        "rolling_window_vector_b_pairs": rolling_window_stats["vector_b_pairs"],
        "rolling_window_vector_c_pairs": rolling_window_stats["vector_c_pairs"],
        "rolling_window_vector_d_pairs": rolling_window_stats["vector_d_pairs"],
        "available_recent_pairs": len(all_pairs),
        "usable_training_samples": int(stats["count"]),
        "model_id": model_id,
        "device": device,
        "max_steps": max_steps,
        "learning_rate": lr,
        "adapter_run_dir": str(run_dir),
        "max_feedback_id": max_feedback_id or last_fb_id,
    }
    _write_state(
        last_message_id=max_assistant_id or last_msg_id,
        last_training_example_id=max_example_id or last_ex_id,
        last_feedback_id=max_feedback_id or last_fb_id,
        stats=train_stats,
    )
    log.info("Training complete. Updated adapter in %s", latest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
