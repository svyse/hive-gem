from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    """Tokenize text into coarse words for similarity matching.

    Notes:
    - Keep it intentionally lightweight (no external deps).
    - We drop very short tokens to reduce noise.
    """

    if not text:
        return []

    cleaned = []
    buf: List[str] = []

    def _flush():
        if buf:
            cleaned.append("".join(buf))
            buf.clear()

    for ch in text.lower():
        if ch.isalnum() or ch in ("_", "-"):
            buf.append(ch)
        else:
            _flush()
    _flush()

    # Remove noise tokens
    out = [t for t in cleaned if len(t) >= 3]
    return out


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    return float(inter) / float(uni) if uni else 0.0


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


_ERRORISH_TRACE_PHRASES = (
    "internal error in",
    "an internal error occurred",
    "internal error while answering",
    "please check the run log for details",
    "could not parse json from model output",
    "q&a agent output must be json object",
)


def _trace_is_usable(trace: Dict[str, Any]) -> bool:
    if trace.get("success") is False or trace.get("_memory_success") is False:
        return False
    payload = json.dumps(trace, ensure_ascii=False).lower()
    return not any(p in payload for p in _ERRORISH_TRACE_PHRASES)


@dataclass
class QueryTraceRecorder:
    """Records a per-query sequence of events (agent calls + memory accesses).

    This is used by orchestrators to:
    1) track what happened for a specific user query (ordered events)
    2) persist it into type memory + hive memory
    3) retrieve similar past traces and use them as "playbooks" for future queries
    """

    orchestrator: str
    run_id: str
    query: str
    domain: Optional[str] = None
    conversation_id: Optional[str] = None
    started_at: str = field(default_factory=_now_iso)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def event(self, kind: str, **data: Any) -> None:
        self.events.append({"ts": _now_iso(), "kind": kind, **(data or {})})

    def agent_sequence(self) -> List[str]:
        seq: List[str] = []
        for e in self.events:
            k = e.get("kind")
            if k in ("agent_spawn", "agent_call", "agent_terminate"):
                at = (e.get("agent_type") or "").strip()
                if at:
                    seq.append(at)
            elif k in ("orchestrator_consult", "orchestrator_spawn"):
                ot = (e.get("orchestrator_type") or "").strip()
                if ot:
                    seq.append(ot)
        return seq

    def memory_sequence(self) -> List[str]:
        seq: List[str] = []
        for e in self.events:
            if e.get("kind") != "memory_access":
                continue
            op = (e.get("op") or "").strip()
            if not op:
                continue
            scope = (e.get("scope") or "").strip()
            agent_type = (e.get("agent_type") or "").strip()
            bits = [op]
            if scope:
                bits.append(f"scope={scope}")
            if agent_type:
                bits.append(f"agent_type={agent_type}")
            seq.append("|".join(bits))
        return seq

    def signature(self) -> str:
        a = " > ".join(self.agent_sequence())
        m = " > ".join(self.memory_sequence())
        return f"agents:{a}; mem:{m}".strip()

    def to_dict(self, *, finished_at: Optional[str] = None, success: Optional[bool] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "kind": "sequence_trace",
            "orchestrator": self.orchestrator,
            "domain": self.domain,
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "query": self.query,
            "started_at": self.started_at,
            "finished_at": finished_at or _now_iso(),
            "success": success,
            "agent_sequence": self.agent_sequence(),
            "memory_sequence": self.memory_sequence(),
            "signature": self.signature(),
            "events": self.events,
        }
        if extra:
            try:
                payload["extra"] = extra
            except Exception:
                pass
        return payload


def extract_sequence_traces_from_memories(memories: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse and return sequence traces from memory-store entries."""

    traces: List[Dict[str, Any]] = []
    for m in memories or []:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, str):
            continue
        obj = _safe_json_loads(content)
        if not obj or obj.get("kind") != "sequence_trace":
            continue
        # Keep some provenance
        obj.setdefault("_memory_id", m.get("id"))
        obj.setdefault("_memory_scope", m.get("scope"))
        obj.setdefault("_memory_agent_type", m.get("agent_type"))
        obj.setdefault("_memory_created_at", m.get("created_at"))
        obj.setdefault("_memory_success", m.get("success"))
        traces.append(obj)
    return traces


def rank_similar_traces(
    *,
    query: str,
    traces: List[Dict[str, Any]],
    max_traces: int = 8,
    min_similarity: float = 0.08,
    orchestrator: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Rank traces by lexical similarity to the query.

    We use Jaccard over token sets (lightweight, dependency-free).
    """

    q_tokens = _tokenize(query)
    scored: List[Tuple[float, str, Dict[str, Any]]] = []

    for t in traces or []:
        if not _trace_is_usable(t):
            continue
        if orchestrator and (t.get("orchestrator") != orchestrator):
            continue
        t_query = str(t.get("query") or "")
        sim = _jaccard(q_tokens, _tokenize(t_query))
        if sim < float(min_similarity or 0.0):
            continue
        ts = str(t.get("finished_at") or t.get("_memory_created_at") or t.get("started_at") or "")
        scored.append((sim, ts, t))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [t for _, _, t in scored[: max(1, int(max_traces or 1))]]


def recommend_most_common_sequence(seqs: List[List[str]]) -> List[str]:
    """Given a list of sequences (lists of strings), return the most common one.

    - ties are broken by longer sequences (more informative)
    - if still tied, the first occurrence wins (caller should pass sequences in recency order)
    """

    tuples = [tuple(s) for s in seqs if s]
    if not tuples:
        return []

    counts = Counter(tuples)
    best_count = max(counts.values())
    # Candidates with best count
    candidates = [t for t, c in counts.items() if c == best_count]
    candidates.sort(key=lambda t: len(t), reverse=True)

    # Preserve recency: choose the earliest appearance among the tied candidates
    for s in tuples:
        if s in candidates:
            return list(s)
    return list(candidates[0])


def merge_preferred_order(preferred: List[str], existing: List[str], *, max_len: Optional[int] = None) -> List[str]:
    """Merge two sequences while preserving order.

    preferred items come first, then any remaining existing items.
    """

    out: List[str] = []
    for x in preferred or []:
        if x and x not in out:
            out.append(x)
    for x in existing or []:
        if x and x not in out:
            out.append(x)
    if max_len is not None:
        return out[: int(max_len)]
    return out
