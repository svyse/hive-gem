from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


class PathTraversalError(ValueError):
    pass


def safe_resolve(root: Path, rel_path: str) -> Path:
    root = root.resolve()
    candidate = (root / rel_path).resolve()
    if root not in candidate.parents and candidate != root:
        raise PathTraversalError(f"Refusing path outside root: {rel_path}")
    return candidate


def read_text(root: Path, rel_path: str) -> str:
    p = safe_resolve(root, rel_path)
    return p.read_text(encoding="utf-8")


def write_text(root: Path, rel_path: str, content: str) -> None:
    p = safe_resolve(root, rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def mkdir(root: Path, rel_path: str) -> None:
    p = safe_resolve(root, rel_path)
    p.mkdir(parents=True, exist_ok=True)


def list_files(root: Path, *, exts: List[str] | None = None, max_files: int = 2000) -> List[str]:
    root = root.resolve()
    out: List[str] = []
    for p in root.rglob("*"):
        if len(out) >= max_files:
            break
        if p.is_dir():
            continue
        if ".git" in p.parts or "__pycache__" in p.parts or ".venv" in p.parts:
            continue
        if exts and p.suffix.lower() not in [e.lower() for e in exts]:
            continue
        out.append(str(p.relative_to(root)))
    return sorted(out)
