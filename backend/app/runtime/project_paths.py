from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional


def repo_root() -> Path:
    """Return the project repository root.

    This file lives at backend/app/runtime/project_paths.py, so parents[3]
    is the directory that contains backend/, frontend/, and sample_projects/.
    """

    return Path(__file__).resolve().parents[3]


def sample_projects_root() -> Path:
    return repo_root() / "sample_projects"


def _clean_path(raw: str | os.PathLike[str]) -> str:
    text = str(raw or "").strip().strip('"').strip("'")
    text = text.replace("\\", "/")

    # User-facing shorthand from Windows examples: ~.\sample_projects\hello
    if text.startswith("~./") or text.startswith("~."):
        text = text[2:].lstrip("/")
    if text.startswith("./"):
        text = text[2:]
    return text.strip()


def _safe_resolve_under(root: Path, parts: Iterable[str]) -> Path:
    root = root.resolve()
    cleaned_parts = []
    for part in parts:
        p = str(part or "").strip()
        if not p or p in {".", ".."}:
            continue
        cleaned_parts.append(p)
    target = (root.joinpath(*cleaned_parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Project path escapes allowed root: {target}") from exc
    return target


def _is_simple_project_name(text: str) -> bool:
    # A bare project name like "hello" should mean sample_projects/hello.
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", text or ""))


def is_sample_project_path(path: Path) -> bool:
    try:
        Path(path).resolve().relative_to(sample_projects_root().resolve())
        return True
    except Exception:
        return False


def resolve_project_path(
    project_path: str | os.PathLike[str],
    *,
    create_if_missing: bool = False,
    prefer_sample_projects: bool = True,
) -> Path:
    """Resolve a user-entered project path.

    Rules:
    - "hello" resolves to <repo>/sample_projects/hello.
    - "sample_projects/hello" and "~.\\sample_projects\\hello" resolve under the repo root.
    - Existing relative paths are still supported for backwards compatibility.
    - Missing sample_projects paths can be created by code-mode callers.
    - The .workspaces/.workspace mechanism is left untouched.
    """

    text = _clean_path(project_path)
    if not text:
        raise ValueError("Project path is empty")

    root = repo_root()
    samples = sample_projects_root()
    parts = [p for p in text.split("/") if p not in {"", "."}]

    # Explicit sample_projects path.
    lowered = [p.lower() for p in parts]
    if "sample_projects" in lowered:
        idx = lowered.index("sample_projects")
        rel_parts = parts[idx + 1 :]
        target = _safe_resolve_under(samples, rel_parts)
        if create_if_missing:
            target.mkdir(parents=True, exist_ok=True)
        return target

    # Bare project name -> sample_projects/<name>.
    if prefer_sample_projects and _is_simple_project_name(text):
        target = _safe_resolve_under(samples, [text])
        if create_if_missing:
            target.mkdir(parents=True, exist_ok=True)
        return target

    # Absolute POSIX path or expanded home path. Keep legacy support.
    expanded = Path(text).expanduser()
    if expanded.is_absolute():
        target = expanded.resolve()
        if create_if_missing and is_sample_project_path(target):
            target.mkdir(parents=True, exist_ok=True)
        return target

    # Existing path relative to cwd.
    cwd_candidate = (Path.cwd() / expanded).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    # Existing path relative to repo root.
    repo_candidate = (root / expanded).resolve()
    if repo_candidate.exists():
        return repo_candidate

    # For non-existing paths, only create if they clearly target sample_projects.
    if create_if_missing and prefer_sample_projects and _is_simple_project_name(text):
        target = _safe_resolve_under(samples, [text])
        target.mkdir(parents=True, exist_ok=True)
        return target

    return repo_candidate
