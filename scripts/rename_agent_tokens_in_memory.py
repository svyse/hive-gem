from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

TEXT_COLUMNS = {
    'conversation_messages': ['content', 'meta', 'orchestrator', 'input_mode', 'conversation_id', 'role', 'created_at'],
    'conversations': ['conversation_id', 'title', 'orchestrator', 'metadata', 'created_at', 'updated_at'],
    'conversation_traces': ['conversation_id', 'turn_id', 'orchestrator', 'created_at'],
    'memories': ['scope', 'agent_type', 'agent_id', 'content', 'tags', 'created_at'],
    'run_traces': ['run_id', 'orchestrator', 'created_at'],
    'training_examples': ['source', 'upload_id', 'prompt', 'completion', 'created_at'],
    'uploads': ['upload_id', 'filename', 'content_type', 'saved_path', 'extracted_text_path', 'created_at'],
}


def replace_all(db_path: Path, old_token: str, new_token: str, old_file: str | None = None, new_file: str | None = None) -> None:
    backup = db_path.with_suffix(db_path.suffix + '.bak')
    shutil.copy2(db_path, backup)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    changed = 0
    for table, cols in TEXT_COLUMNS.items():
        for col in cols:
            try:
                if old_file and new_file:
                    cur.execute(
                        f"UPDATE {table} SET {col}=REPLACE(REPLACE({col}, ?, ?), ?, ?) WHERE {col} LIKE '%' || ? || '%' OR {col} LIKE '%' || ? || '%'",
                        (old_token, new_token, old_file, new_file, old_token, old_file),
                    )
                else:
                    cur.execute(
                        f"UPDATE {table} SET {col}=REPLACE({col}, ?, ?) WHERE {col} LIKE '%' || ? || '%'",
                        (old_token, new_token, old_token),
                    )
                changed += cur.rowcount if cur.rowcount is not None else 0
            except sqlite3.OperationalError:
                continue
    conn.commit()
    try:
        cur.execute('VACUUM')
    except Exception:
        pass
    conn.close()
    print(f'Updated rows/fields: {changed}')
    print(f'Backup written to: {backup}')


if __name__ == '__main__':
    import sys

    if len(sys.argv) not in {4, 6}:
        raise SystemExit(
            'Usage: python scripts/rename_agent_tokens_in_memory.py <memory.sqlite> <old_token> <new_token> [<old_file_token> <new_file_token>]'
        )
    db = Path(sys.argv[1]).expanduser().resolve()
    old_token = sys.argv[2]
    new_token = sys.argv[3]
    old_file = sys.argv[4] if len(sys.argv) == 6 else None
    new_file = sys.argv[5] if len(sys.argv) == 6 else None
    replace_all(db, old_token, new_token, old_file, new_file)
