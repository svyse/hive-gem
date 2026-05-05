from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings


class MemoryStore:
    """A lightweight sqlite-backed memory store.

    This store backs both:
    - long-term hive/type/agent memories (table: `memories`)
    - persistent multi-turn Q&A conversations (tables: `conversations`, `conversation_messages`)

    Memory scopes:
      - hive: accessible to all agents
      - type: shared among agents of a given agent_type
      - agent: per-agent instance memory
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        # SQLite defaults can be fragile under concurrent access (e.g. multiple
        # uvicorn workers / reload processes). We use a higher timeout and set
        # pragmatic defaults (WAL + busy_timeout) to reduce "database is locked"
        # errors, which were causing Q&A runs to fail mid-turn.
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()

            # Best-effort pragmas for fewer lock errors.
            # - WAL improves writer concurrency.
            # - busy_timeout makes sqlite wait instead of raising immediately.
            try:
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=NORMAL;")
                cur.execute("PRAGMA busy_timeout=30000;")
            except Exception:
                # Ignore on environments that don't support these pragmas.
                pass

            # --- memories ---
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    agent_type TEXT,
                    agent_id TEXT,
                    content TEXT NOT NULL,
                    tags TEXT,
                    success INTEGER,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_agent_type ON memories(agent_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_agent_id ON memories(agent_id);")

            # --- conversations (multi-turn Q&A threads) ---
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT,
                    orchestrator TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    orchestrator TEXT,
                    input_mode TEXT,
                    meta TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                );
                """
            )

            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_orch ON conversations(orchestrator);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_msg_conv ON conversation_messages(conversation_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_msg_role ON conversation_messages(role);"
            )

            # --- conversation traces (sequence-trace links for trace viewer UI) ---
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    orchestrator TEXT NOT NULL,
                    hive_memory_id INTEGER,
                    type_memory_id INTEGER,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_traces_conv_turn ON conversation_traces(conversation_id, turn_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_traces_orch ON conversation_traces(orchestrator);"
            )


            # --- run traces (code pipeline trace viewer support) ---
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS run_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    orchestrator TEXT NOT NULL,
                    hive_memory_id INTEGER,
                    type_memory_id INTEGER,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_run_traces_run ON run_traces(run_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_run_traces_orch ON run_traces(orchestrator);")

            # --- uploads (documents/images ingested into hive memory) ---
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS uploads (
                    upload_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_type TEXT,
                    size_bytes INTEGER,
                    saved_path TEXT NOT NULL,
                    extracted_text_path TEXT,
                    text_chars INTEGER,
                    chunks_added INTEGER,
                    training_examples_added INTEGER,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_uploads_created_at ON uploads(created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_uploads_filename ON uploads(filename);")

            # --- training examples (extra SFT pairs beyond conversation history) ---
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS training_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    upload_id TEXT,
                    prompt TEXT NOT NULL,
                    completion TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_train_ex_created_at ON training_examples(created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_train_ex_upload ON training_examples(upload_id);")

            # --- speech replacements (learned dictation corrections) ---
            # This schema must be created during store initialization, not at
            # import-time, otherwise uvicorn reload/subprocess spawning can crash.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS speech_replacements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL,
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_speech_repl_unique ON speech_replacements(mode, src, dst);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_speech_repl_mode_count ON speech_replacements(mode, count DESC);"
            )

            self._conn.commit()

    # ---------------------------------------------------------------------
    # Speech replacements API (learned dictation corrections)
    # ---------------------------------------------------------------------

    def upsert_speech_replacement(self, *, mode: str, src: str, dst: str, updated_at: str) -> None:
        """Insert or increment a speech dictation replacement rule."""
        src = (src or "").strip()
        dst = (dst or "").strip()
        mode = (mode or "qa").strip()
        if not src or not dst:
            return

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO speech_replacements(mode, src, dst, count, updated_at)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(mode, src, dst)
                DO UPDATE SET count = count + 1, updated_at = excluded.updated_at
                """,
                (mode, src, dst, updated_at),
            )
            self._conn.commit()

    def list_speech_replacements(self, *, mode: str, limit: int = 100) -> List[Dict[str, Any]]:
        mode = (mode or "qa").strip()
        limit = int(limit or 100)
        if limit < 1:
            limit = 1
        if limit > 500:
            limit = 500

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT src, dst, count, updated_at
                FROM speech_replacements
                WHERE mode = ?
                ORDER BY count DESC, LENGTH(src) DESC
                LIMIT ?
                """,
                (mode, limit),
            )
            rows = cur.fetchall()

        return [
            {"src": r[0], "dst": r[1], "count": int(r[2]), "updated_at": r[3]}
            for r in rows
        ]

    # ---------------------------------------------------------------------
    # Memories API
    # ---------------------------------------------------------------------

    def add(
        self,
        *,
        scope: str,
        content: str,
        created_at: str,
        agent_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        success: Optional[bool] = None,
    ) -> int:
        tags_json = json.dumps(tags or [])
        success_int = None if success is None else (1 if success else 0)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO memories(scope, agent_type, agent_id, content, tags, success, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (scope, agent_type, agent_id, content, tags_json, success_int, created_at),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_memory_by_id(self, *, memory_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single memory entry by id."""
        with self._lock:
            cur = self._conn.cursor()
            row = cur.execute("SELECT * FROM memories WHERE id = ?", (int(memory_id),)).fetchone()

        if row is None:
            return None

        return {
            "id": int(row["id"]),
            "scope": row["scope"],
            "agent_type": row["agent_type"],
            "agent_id": row["agent_id"],
            "content": row["content"],
            "tags": json.loads(row["tags"] or "[]"),
            "success": None if row["success"] is None else bool(int(row["success"])),
            "created_at": row["created_at"],
        }

    def search(
        self,
        *,
        scope: str,
        query: str,
        limit: int = 25,
        agent_type: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q = f"%{query}%"
        clauses = ["scope = ?", "content LIKE ?"]
        params: List[Any] = [scope, q]

        if agent_type is not None:
            clauses.append("agent_type = ?")
            params.append(agent_type)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)

        where = " AND ".join(clauses)
        sql = f"SELECT * FROM memories WHERE {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(sql, params).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "scope": r["scope"],
                    "agent_type": r["agent_type"],
                    "agent_id": r["agent_id"],
                    "content": r["content"],
                    "tags": json.loads(r["tags"] or "[]"),
                    "success": None if r["success"] is None else bool(int(r["success"])),
                    "created_at": r["created_at"],
                }
            )
        return out

    def search_all(
        self,
        *,
        query: str,
        limit: int = 25,
        scopes: Optional[List[str]] = None,
        agent_type: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search across multiple scopes.

        If scopes is None, searches across all scopes.
        """
        q = f"%{query}%"
        clauses = ["content LIKE ?"]
        params: List[Any] = [q]

        if scopes:
            placeholders = ",".join(["?"] * len(scopes))
            clauses.append(f"scope IN ({placeholders})")
            params.extend(scopes)

        if agent_type is not None:
            clauses.append("agent_type = ?")
            params.append(agent_type)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)

        where = " AND ".join(clauses)
        sql = f"SELECT * FROM memories WHERE {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(sql, params).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "scope": r["scope"],
                    "agent_type": r["agent_type"],
                    "agent_id": r["agent_id"],
                    "content": r["content"],
                    "tags": json.loads(r["tags"] or "[]"),
                    "success": None if r["success"] is None else bool(int(r["success"])),
                    "created_at": r["created_at"],
                }
            )
        return out

    def recent_all(
        self,
        *,
        limit: int = 25,
        scopes: Optional[List[str]] = None,
        agent_type: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return most recent memories across multiple scopes."""
        clauses: List[str] = []
        params: List[Any] = []

        if scopes:
            placeholders = ",".join(["?"] * len(scopes))
            clauses.append(f"scope IN ({placeholders})")
            params.extend(scopes)

        if agent_type is not None:
            clauses.append("agent_type = ?")
            params.append(agent_type)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM memories{where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(sql, params).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "scope": r["scope"],
                    "agent_type": r["agent_type"],
                    "agent_id": r["agent_id"],
                    "content": r["content"],
                    "tags": json.loads(r["tags"] or "[]"),
                    "success": None if r["success"] is None else bool(int(r["success"])),
                    "created_at": r["created_at"],
                }
            )
        return out

    def recent(self, *, scope: str, limit: int = 25, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses = ["scope = ?"]
        params: List[Any] = [scope]
        if agent_type is not None:
            clauses.append("agent_type = ?")
            params.append(agent_type)
        where = " AND ".join(clauses)
        sql = f"SELECT * FROM memories WHERE {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(sql, params).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "scope": r["scope"],
                    "agent_type": r["agent_type"],
                    "agent_id": r["agent_id"],
                    "content": r["content"],
                    "tags": json.loads(r["tags"] or "[]"),
                    "success": None if r["success"] is None else bool(int(r["success"])),
                    "created_at": r["created_at"],
                }
            )
        return out

    # ---------------------------------------------------------------------
    # Conversation API (persistent multi-turn chat threads)
    # ---------------------------------------------------------------------

    def create_conversation(
        self,
        *,
        conversation_id: str,
        orchestrator: Optional[str],
        created_at: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a conversation if it does not already exist."""
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR IGNORE INTO conversations(conversation_id, title, orchestrator, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, title, orchestrator, created_at, created_at, meta_json),
            )
            self._conn.commit()

    def update_conversation(
        self,
        *,
        conversation_id: str,
        updated_at: str,
        title: Optional[str] = None,
        orchestrator: Optional[str] = None,
        metadata_patch: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update conversation fields (best-effort).

        metadata_patch is shallow-merged into existing metadata.
        """
        with self._lock:
            cur = self._conn.cursor()
            row = cur.execute(
                "SELECT metadata FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            existing_meta: Dict[str, Any] = {}
            if row is not None:
                try:
                    existing_meta = json.loads(row["metadata"] or "{}")
                except Exception:
                    existing_meta = {}

            if metadata_patch:
                try:
                    existing_meta.update(metadata_patch)
                except Exception:
                    pass

            fields: List[str] = ["updated_at = ?"]
            params: List[Any] = [updated_at]

            if title is not None:
                fields.append("title = ?")
                params.append(title)
            if orchestrator is not None:
                fields.append("orchestrator = ?")
                params.append(orchestrator)

            fields.append("metadata = ?")
            params.append(json.dumps(existing_meta, ensure_ascii=False))

            params.append(conversation_id)
            sql = f"UPDATE conversations SET {', '.join(fields)} WHERE conversation_id = ?"
            cur.execute(sql, params)
            self._conn.commit()

    def get_conversation(self, *, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            row = cur.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            meta = json.loads(row["metadata"] or "{}")
        except Exception:
            meta = {}
        return {
            "conversation_id": row["conversation_id"],
            "title": row["title"],
            "orchestrator": row["orchestrator"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": meta,
        }

    def list_conversations(self, *, limit: int = 20, orchestrator: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if orchestrator:
            clauses.append("orchestrator = ?")
            params.append(orchestrator)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM conversations{where} ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(sql, params).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                meta = json.loads(r["metadata"] or "{}")
            except Exception:
                meta = {}
            out.append(
                {
                    "conversation_id": r["conversation_id"],
                    "title": r["title"],
                    "orchestrator": r["orchestrator"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "metadata": meta,
                }
            )
        return out

    def add_conversation_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        created_at: str,
        orchestrator: Optional[str] = None,
        input_mode: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO conversation_messages(conversation_id, role, content, orchestrator, input_mode, meta, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, orchestrator, input_mode, meta_json, created_at),
            )
            # update conversation updated_at
            cur.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (created_at, conversation_id),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_conversation_messages(self, *, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                """
                SELECT * FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()

        out: List[Dict[str, Any]] = []
        # reverse to chronological
        for r in reversed(rows):
            try:
                meta = json.loads(r["meta"] or "{}")
                # Response schemas expect meta to be a JSON object.
                # If older data stored a non-object (e.g. null/list/string),
                # coerce it to a dict to avoid response validation errors.
                if not isinstance(meta, dict):
                    meta = {"value": meta}
            except Exception:
                meta = {}
            out.append(
                {
                    "id": int(r["id"]),
                    "conversation_id": r["conversation_id"],
                    "role": r["role"],
                    "content": r["content"],
                    "orchestrator": r["orchestrator"],
                    "input_mode": r["input_mode"],
                    "meta": meta,
                    "created_at": r["created_at"],
                }
            )
        return out

    def get_conversation_thread(
        self,
        *,
        conversation_id: str,
        orchestrator: str,
        limit: int = 50,
        include_master: bool = True,
    ) -> List[Dict[str, Any]]:
        """Convenience: get messages relevant to a specific orchestrator thread.

        Policy:
        - always include user messages
        - include assistant messages where orchestrator matches
        - optionally include hive master assistant messages
        """
        msgs = self.get_conversation_messages(conversation_id=conversation_id, limit=limit)
        thread: List[Dict[str, Any]] = []
        for m in msgs:
            if m.get("role") == "user":
                thread.append(m)
                continue
            if m.get("role") == "assistant":
                orch = (m.get("orchestrator") or "").strip()
                if orch == orchestrator:
                    thread.append(m)
                elif include_master and orch == "hive_master_orchestrator":
                    thread.append(m)
        return thread

    # ---------------------------------------------------------------------
    # Conversation traces (sequence trace viewer support)
    # ---------------------------------------------------------------------

    def add_conversation_trace(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        orchestrator: str,
        created_at: str,
        hive_memory_id: Optional[int] = None,
        type_memory_id: Optional[int] = None,
    ) -> int:
        """Record a link from a conversation turn to a stored sequence trace.

        We keep the trace content in the `memories` table (scope hive/type)
        and store only the IDs here for fast lookup by the UI.
        """

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO conversation_traces(conversation_id, turn_id, orchestrator, hive_memory_id, type_memory_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    turn_id,
                    orchestrator,
                    None if hive_memory_id is None else int(hive_memory_id),
                    None if type_memory_id is None else int(type_memory_id),
                    created_at,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_conversation_trace(self, *, trace_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            row = cur.execute(
                "SELECT * FROM conversation_traces WHERE id = ?",
                (int(trace_id),),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "conversation_id": row["conversation_id"],
            "turn_id": row["turn_id"],
            "orchestrator": row["orchestrator"],
            "hive_memory_id": None if row["hive_memory_id"] is None else int(row["hive_memory_id"]),
            "type_memory_id": None if row["type_memory_id"] is None else int(row["type_memory_id"]),
            "created_at": row["created_at"],
        }

    def list_conversation_traces(
        self,
        *,
        conversation_id: str,
        turn_id: Optional[str] = None,
        orchestrator: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        clauses = ["conversation_id = ?"]
        params: List[Any] = [conversation_id]

        if turn_id:
            clauses.append("turn_id = ?")
            params.append(turn_id)
        if orchestrator:
            clauses.append("orchestrator = ?")
            params.append(orchestrator)

        where = " AND ".join(clauses)
        sql = f"SELECT * FROM conversation_traces WHERE {where} ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(sql, params).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "conversation_id": r["conversation_id"],
                    "turn_id": r["turn_id"],
                    "orchestrator": r["orchestrator"],
                    "hive_memory_id": None if r["hive_memory_id"] is None else int(r["hive_memory_id"]),
                    "type_memory_id": None if r["type_memory_id"] is None else int(r["type_memory_id"]),
                    "created_at": r["created_at"],
                }
            )
        return out


    # ---------------------------------------------------------------------
    # Run traces (code pipeline trace viewer support)
    # ---------------------------------------------------------------------

    def add_run_trace(
        self,
        *,
        run_id: str,
        orchestrator: str,
        created_at: str,
        hive_memory_id: Optional[int] = None,
        type_memory_id: Optional[int] = None,
    ) -> int:
        """Record a link from a code-pipeline run to a stored sequence trace.

        Like conversation traces, the *content* lives in the `memories` table
        (scope hive/type). This table just links run_id -> memory ids for fast
        UI lookup.
        """

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO run_traces(run_id, orchestrator, hive_memory_id, type_memory_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    orchestrator,
                    None if hive_memory_id is None else int(hive_memory_id),
                    None if type_memory_id is None else int(type_memory_id),
                    created_at,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_run_trace(self, *, trace_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            row = cur.execute(
                "SELECT * FROM run_traces WHERE id = ?",
                (int(trace_id),),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "run_id": row["run_id"],
            "orchestrator": row["orchestrator"],
            "hive_memory_id": None if row["hive_memory_id"] is None else int(row["hive_memory_id"]),
            "type_memory_id": None if row["type_memory_id"] is None else int(row["type_memory_id"]),
            "created_at": row["created_at"],
        }

    def list_run_traces(
        self,
        *,
        run_id: str,
        orchestrator: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        clauses = ["run_id = ?"]
        params: List[Any] = [run_id]

        if orchestrator:
            clauses.append("orchestrator = ?")
            params.append(orchestrator)

        where = " AND ".join(clauses)
        sql = f"SELECT * FROM run_traces WHERE {where} ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(sql, params).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "run_id": r["run_id"],
                    "orchestrator": r["orchestrator"],
                    "hive_memory_id": None if r["hive_memory_id"] is None else int(r["hive_memory_id"]),
                    "type_memory_id": None if r["type_memory_id"] is None else int(r["type_memory_id"]),
                    "created_at": r["created_at"],
                }
            )
        return out


    # ---------------------------------------------------------------------
    # Uploads API (documents/images)
    # ---------------------------------------------------------------------

    def add_upload_record(
        self,
        *,
        upload_id: str,
        filename: str,
        content_type: Optional[str],
        size_bytes: int,
        saved_path: str,
        extracted_text_path: Optional[str],
        text_chars: int,
        chunks_added: int,
        training_examples_added: int,
        created_at: str,
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO uploads(
                    upload_id, filename, content_type, size_bytes, saved_path, extracted_text_path,
                    text_chars, chunks_added, training_examples_added, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(upload_id),
                    str(filename),
                    None if content_type is None else str(content_type),
                    int(size_bytes),
                    str(saved_path),
                    None if extracted_text_path is None else str(extracted_text_path),
                    int(text_chars),
                    int(chunks_added),
                    int(training_examples_added),
                    str(created_at),
                ),
            )
            self._conn.commit()

    def get_upload(self, *, upload_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            row = cur.execute("SELECT * FROM uploads WHERE upload_id = ?", (str(upload_id),)).fetchone()
        if row is None:
            return None
        return {
            "upload_id": row["upload_id"],
            "filename": row["filename"],
            "content_type": row["content_type"],
            "size_bytes": None if row["size_bytes"] is None else int(row["size_bytes"]),
            "saved_path": row["saved_path"],
            "extracted_text_path": row["extracted_text_path"],
            "text_chars": None if row["text_chars"] is None else int(row["text_chars"]),
            "chunks_added": None if row["chunks_added"] is None else int(row["chunks_added"]),
            "training_examples_added": None if row["training_examples_added"] is None else int(row["training_examples_added"]),
            "created_at": row["created_at"],
        }

    def list_uploads(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute("SELECT * FROM uploads ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "upload_id": r["upload_id"],
                    "filename": r["filename"],
                    "content_type": r["content_type"],
                    "size_bytes": None if r["size_bytes"] is None else int(r["size_bytes"]),
                    "saved_path": r["saved_path"],
                    "extracted_text_path": r["extracted_text_path"],
                    "text_chars": None if r["text_chars"] is None else int(r["text_chars"]),
                    "chunks_added": None if r["chunks_added"] is None else int(r["chunks_added"]),
                    "training_examples_added": None if r["training_examples_added"] is None else int(r["training_examples_added"]),
                    "created_at": r["created_at"],
                }
            )
        return out

    # ---------------------------------------------------------------------
    # Training examples API
    # ---------------------------------------------------------------------

    def add_training_example(
        self,
        *,
        source: str,
        prompt: str,
        completion: str,
        created_at: str,
        upload_id: Optional[str] = None,
    ) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO training_examples(source, upload_id, prompt, completion, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(source),
                    None if upload_id is None else str(upload_id),
                    str(prompt),
                    str(completion),
                    str(created_at),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def count_training_examples_since(self, *, last_id: int) -> int:
        try:
            with self._lock:
                cur = self._conn.cursor()
                row = cur.execute("SELECT COUNT(*) AS n FROM training_examples WHERE id > ?", (int(last_id),)).fetchone()
            return int(row["n"] if row else 0)
        except Exception:
            return 0

    def recent_training_examples(self, *, limit: int = 200, since_id: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                "SELECT * FROM training_examples WHERE id > ? ORDER BY id DESC LIMIT ?",
                (int(since_id), int(limit)),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "source": r["source"],
                    "upload_id": r["upload_id"],
                    "prompt": r["prompt"],
                    "completion": r["completion"],
                    "created_at": r["created_at"],
                }
            )
        return out


_STORE: Optional[MemoryStore] = None




def upsert_speech_replacement(
    store: Optional[MemoryStore] = None,
    *,
    mode: str,
    src: str,
    dst: str,
    updated_at: str,
) -> None:
    """Module-level convenience wrapper."""
    (store or get_memory_store()).upsert_speech_replacement(
        mode=mode,
        src=src,
        dst=dst,
        updated_at=updated_at,
    )


def list_speech_replacements(
    store: Optional[MemoryStore] = None,
    *,
    mode: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Module-level convenience wrapper."""
    return (store or get_memory_store()).list_speech_replacements(mode=mode, limit=limit)
def get_memory_store() -> MemoryStore:
    global _STORE
    if _STORE is None:
        _STORE = MemoryStore(settings.memory_db_path)
    return _STORE
