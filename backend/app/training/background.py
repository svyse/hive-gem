from __future__ import annotations

import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.llm.adapter_paths import resolve_training_state_file


log = logging.getLogger(__name__)


_SUPERVISOR: Optional["_TrainingSupervisor"] = None


def _backend_root() -> Path:
    # backend/app/training/background.py -> .../backend
    return Path(__file__).resolve().parents[2]


def _training_state_file() -> Path:
    # Keep a separate state file per base model so 1.5B and 3B adapters do not overwrite each other.
    model_id = str(getattr(settings, "local_training_model", None) or settings.local_llm_model)
    return resolve_training_state_file(settings.local_llm_adapter_dir, model_id)


def _load_training_state() -> tuple[int, int]:
    """Return (last_message_id, last_training_example_id).

    Stored in adapter_parent/training_state.json.
    """

    p = _training_state_file()
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            last_msg = int(data.get("last_message_id") or 0)
            last_ex = int(data.get("last_training_example_id") or 0)
            return last_msg, last_ex
    except Exception:
        return 0, 0
    return 0, 0


def _include_upload_examples() -> bool:
    return bool(getattr(settings, "local_training_include_upload_examples", False))


def _current_db_max_ids() -> tuple[int, int]:
    db_path = Path(settings.memory_db_path).expanduser()
    if not db_path.exists():
        return 0, 0
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        cur = conn.cursor()
        row1 = cur.execute("SELECT COALESCE(MAX(id), 0) FROM conversation_messages").fetchone()
        row2 = cur.execute("SELECT COALESCE(MAX(id), 0) FROM training_examples").fetchone()
        conn.close()
        return int((row1 or [0])[0] or 0), int((row2 or [0])[0] or 0)
    except Exception:
        return 0, 0


def _normalize_state_ids(last_message_id: int, last_example_id: int) -> tuple[int, int]:
    max_msg, max_ex = _current_db_max_ids()
    if int(last_message_id or 0) > max_msg:
        last_message_id = 0
    if int(last_example_id or 0) > max_ex:
        last_example_id = 0
    return int(last_message_id or 0), int(last_example_id or 0)


def _count_new_pairs(last_message_id: int, last_example_id: int) -> int:
    """Quick heuristic: count new assistant messages + training_examples."""

    db_path = Path(settings.memory_db_path).expanduser()
    if not db_path.exists():
        return 0

    try:
        last_message_id, last_example_id = _normalize_state_ids(last_message_id, last_example_id)
        conn = sqlite3.connect(str(db_path), timeout=5.0)
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

        conn.close()
        n1 = int(row1["n"] if row1 else 0)
        n2 = int(row2["n"] if row2 else 0)
        return n1 + n2
    except Exception:
        return 0


class _TrainingSupervisor:

    """Lightweight supervisor that spawns the LoRA worker in a separate process."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        p = self._proc
        if p and p.poll() is None:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                try:
                    p.terminate()
                except Exception:
                    pass

    def _spawn_worker(self) -> None:
        if self._proc and self._proc.poll() is None:
            return

        cmd = [sys.executable, "-m", "app.training.worker"]
        env = os.environ.copy()
        # Best-effort hinting for GPU selection (worker also reads LOCAL_TRAINING_DEVICE).
        if settings.local_training_device and str(settings.local_training_device).strip().lower() not in {"auto", ""}:
            env.setdefault("LOCAL_TRAINING_DEVICE", str(settings.local_training_device))

        try:
            env.setdefault("PYTHONUNBUFFERED", "1")
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(_backend_root()),
                env=env,
            )
            log.info("Started local training worker (pid=%s)", self._proc.pid)
        except Exception as e:
            log.warning("Failed to start training worker: %s", e)

    def _loop(self) -> None:
        poll_s = max(30, int(getattr(settings, "local_training_poll_s", 900) or 900))
        min_new = max(1, int(getattr(settings, "local_training_min_new_pairs", 25) or 25))

        while not self._stop.is_set():
            try:
                # If a worker is running, just sleep.
                if self._proc and self._proc.poll() is None:
                    time.sleep(5)
                    continue

                if self._proc is not None and self._proc.poll() is not None:
                    code = self._proc.returncode
                    if code not in (0, None):
                        log.warning("Local training worker exited with status %s", code)
                    self._proc = None

                last_msg_id, last_ex_id = _load_training_state()
                n_new = _count_new_pairs(last_msg_id, last_ex_id)

                if n_new >= min_new:
                    log.info("Training trigger: %s new items since state=(msg=%s, ex=%s) (>= %s)", n_new, last_msg_id, last_ex_id, min_new)
                    self._spawn_worker()
                else:
                    # Nothing to do.
                    time.sleep(poll_s)
            except Exception as e:
                log.warning("Training supervisor loop error: %s", e)
                time.sleep(poll_s)


def maybe_start_background_trainer() -> None:
    """Start the background training supervisor if enabled.

    This is intentionally best-effort: failures here should never prevent the API from starting.
    """

    global _SUPERVISOR

    if not bool(getattr(settings, "local_training_enabled", False)):
        return
    if not bool(getattr(settings, "local_training_autostart", False)):
        # Training can still be run manually via: python -m app.training.worker
        return

    if _SUPERVISOR is None:
        _SUPERVISOR = _TrainingSupervisor()
    try:
        _SUPERVISOR.start()
    except Exception as e:
        log.warning("Unable to start background trainer: %s", e)
