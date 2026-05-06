from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.runtime.project_paths import is_sample_project_path, repo_root, resolve_project_path


def _repo_root() -> Path:
    return repo_root()


@dataclass
class Workspace:
    run_id: str
    run_root: Path
    project_root: Path  # the directory agents should operate on
    copied_to_workspace: bool = False


def create_workspace(run_id: str, project_path: str, copy_project_to_workspace: bool) -> Workspace:
    """Create per-run logs while resolving sample_projects directly.

    sample_projects are the canonical latest project files. When a user enters
    "hello", "sample_projects/hello", or "~.\\sample_projects\\hello", the
    pipeline creates <repo>/sample_projects/hello when needed and edits it in
    place. This keeps .workspace/.workspaces only for run logs and temporary
    metadata, not as the source of truth for project code.
    """

    run_root = settings.workspace_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    src_project = resolve_project_path(project_path, create_if_missing=True, prefer_sample_projects=True)
    if not src_project.exists():
        raise FileNotFoundError(f"Project path not found: {src_project}")
    if not src_project.is_dir():
        raise NotADirectoryError(f"Project path is not a directory: {src_project}")

    # sample_projects must always be edited in place. Non-sample legacy projects
    # can still be copied to a run workspace if the user checks that option.
    should_copy = bool(copy_project_to_workspace) and not is_sample_project_path(src_project)

    if should_copy:
        dst = run_root / "project"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_project, dst)
        project_root = dst
    else:
        project_root = src_project

    return Workspace(run_id=run_id, run_root=run_root, project_root=project_root, copied_to_workspace=should_copy)
