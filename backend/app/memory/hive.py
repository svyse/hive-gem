from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from app.memory.store import MemoryStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class HiveMemory:
    store: MemoryStore

    def add(self, content: str, *, tags: Optional[List[str]] = None, success: Optional[bool] = None) -> int:
        return self.store.add(scope="hive", content=content, tags=tags, success=success, created_at=_now_iso())

    def search(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        return self.store.search(scope="hive", query=query, limit=limit)

    def recent(self, *, limit: int = 10) -> List[Dict[str, Any]]:
        return self.store.recent(scope="hive", limit=limit)
