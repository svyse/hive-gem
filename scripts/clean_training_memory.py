from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

ERROR_PATTERNS = (
    "Internal error in",
    "Internal error while answering",
    "An internal error occurred",
    "Please check the run log for details",
    "Could not parse JSON from model output",
    "Q&A agent output must be JSON object",
)


def _like_sql(pattern: str) -> str:
    return f"%{pattern}%"


def _load_bad_turn_ids(cur: sqlite3.Cursor) -> set[str]:
    bad: set[str] = set()
    rows = cur.execute("SELECT meta FROM conversation_messages WHERE role = 'assistant'").fetchall()
    for (meta_raw,) in rows:
        try:
            meta = json.loads(meta_raw or "{}")
            if not isinstance(meta, dict):
                continue
        except Exception:
            continue
        turn_id = str(meta.get("turn_id") or "").strip()
        result = meta.get("result")
        answer = str(result.get("answer") if isinstance(result, dict) else "").strip()
        if turn_id and any(p.lower() in answer.lower() for p in ERROR_PATTERNS):
            bad.add(turn_id)
    return bad


def clean_db(db_path: Path) -> Path:
    out_path = db_path.with_name(db_path.stem + "_training_clean.sqlite")
    shutil.copy2(db_path, out_path)

    conn = sqlite3.connect(str(out_path))
    cur = conn.cursor()

    bad_turn_ids = _load_bad_turn_ids(cur)

    # Remove upload-derived training rows for a clean conversational/code-only experiment.
    cur.execute("DELETE FROM training_examples WHERE lower(COALESCE(source, '')) = 'upload'")
    cur.execute("UPDATE uploads SET training_examples_added = 0")

    # Remove unusable assistant responses from conversations.
    for pat in ERROR_PATTERNS:
        cur.execute(
            "DELETE FROM conversation_messages WHERE role = 'assistant' AND content LIKE ?",
            (_like_sql(pat),),
        )
        # Scrub stale error payloads that may remain in message metadata.
        cur.execute(
            "UPDATE conversation_messages SET meta = '{}' WHERE meta LIKE ?",
            (_like_sql(pat),),
        )

    # Remove noisy memory rows and failed-turn traces that refer to the deleted bad turns.
    for pat in ERROR_PATTERNS:
        cur.execute("DELETE FROM memories WHERE content LIKE ?", (_like_sql(pat),))
    for turn_id in bad_turn_ids:
        cur.execute("DELETE FROM conversation_traces WHERE turn_id = ?", (turn_id,))
        cur.execute("DELETE FROM memories WHERE content LIKE ?", (_like_sql(turn_id),))

    try:
        remaining = cur.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
        if int(remaining or 0) == 0:
            cur.execute("DELETE FROM sqlite_sequence WHERE name = 'training_examples'")
    except Exception:
        pass

    conn.commit()
    try:
        cur.execute("VACUUM")
    except Exception:
        pass
    conn.close()
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean noisy training data from memory.sqlite")
    ap.add_argument("db", type=Path, help="Path to memory.sqlite")
    args = ap.parse_args()
    out = clean_db(args.db.expanduser().resolve())
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
