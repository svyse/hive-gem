from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.utils.file_utils import safe_resolve
from app.utils.safe_exec import ExecResult, run_command


_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def modules_root() -> Path:
    """Return the workspace modules root directory.

    By default this is: <WORKSPACE_ROOT>/modules
    """

    root = Path(settings.workspace_root) / "modules"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        raise ValueError("module name must not be empty")
    if not _NAME_RE.match(n):
        raise ValueError(
            "Invalid module name. Use letters/numbers plus '_' or '-' (max 64 chars). "
            f"Got: {name!r}"
        )
    return n


@dataclass
class WorkspaceModule:
    name: str
    description: str
    entrypoint: str
    rel_path: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    requirements: List[str] = None  # type: ignore[assignment]
    files: List[str] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "entrypoint": self.entrypoint,
            "rel_path": self.rel_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "requirements": list(self.requirements or []),
            "files": list(self.files or []),
        }


class WorkspaceModuleManager:
    """Create, list, and run small modules under WORKSPACE_ROOT/modules."""

    def __init__(self, *, root: Optional[Path] = None) -> None:
        self.root = (root or modules_root()).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Discovery
    # -------------------------

    def _module_dir(self, name: str) -> Path:
        name = _validate_name(name)
        # safe_resolve prevents path traversal outside self.root
        return safe_resolve(self.root, name)

    def list_modules(self) -> List[WorkspaceModule]:
        out: List[WorkspaceModule] = []
        if not self.root.exists():
            return []

        for p in sorted([x for x in self.root.iterdir() if x.is_dir()]):
            manifest = p / "module.json"
            if not manifest.exists():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
            except Exception:
                continue

            name = str(data.get("name") or p.name)
            entrypoint = str(data.get("entrypoint") or "run.py")
            desc = str(data.get("description") or "")
            rel_path = f"modules/{p.name}"

            reqs = data.get("requirements") or []
            if not isinstance(reqs, list):
                reqs = []
            reqs = [str(x) for x in reqs if str(x).strip()]

            files: List[str] = []
            try:
                for f in sorted([x for x in p.rglob("*") if x.is_file()]):
                    # Keep listing small/safe files only
                    if f.name == "__pycache__":
                        continue
                    files.append(str(f.relative_to(p)))
            except Exception:
                files = []

            out.append(
                WorkspaceModule(
                    name=name,
                    description=desc,
                    entrypoint=entrypoint,
                    rel_path=rel_path,
                    created_at=data.get("created_at"),
                    updated_at=data.get("updated_at"),
                    requirements=reqs,
                    files=files,
                )
            )

        return out

    def get_module(self, name: str) -> Optional[WorkspaceModule]:
        name = _validate_name(name)
        d = self._module_dir(name)
        manifest = d / "module.json"
        if not manifest.exists():
            return None
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
        except Exception:
            return None

        entrypoint = str(data.get("entrypoint") or "run.py")
        desc = str(data.get("description") or "")
        rel_path = f"modules/{d.name}"

        reqs = data.get("requirements") or []
        if not isinstance(reqs, list):
            reqs = []
        reqs = [str(x) for x in reqs if str(x).strip()]

        files: List[str] = []
        try:
            for f in sorted([x for x in d.rglob("*") if x.is_file()]):
                files.append(str(f.relative_to(d)))
        except Exception:
            files = []

        return WorkspaceModule(
            name=name,
            description=desc,
            entrypoint=entrypoint,
            rel_path=rel_path,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            requirements=reqs,
            files=files,
        )

    # -------------------------
    # Mutation
    # -------------------------

    def create_module(
        self,
        *,
        name: str,
        description: str = "",
        entrypoint: str = "run.py",
        files: Optional[Dict[str, str]] = None,
        requirements: Optional[List[str]] = None,
        overwrite: bool = True,
    ) -> WorkspaceModule:
        name = _validate_name(name)
        entrypoint = (entrypoint or "run.py").strip() or "run.py"
        if "/" in entrypoint or "\\" in entrypoint:
            raise ValueError("entrypoint must be a simple filename (no directories)")

        d = self._module_dir(name)
        d.mkdir(parents=True, exist_ok=True)

        # Refuse overwrite unless allowed
        manifest_path = d / "module.json"
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"Module already exists: {name}")

        # Requirements: if provided, we record them into module.json and requirements.txt.
        # If not provided, we preserve existing module requirements (if any).
        existing_reqs: List[str] = []
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and isinstance(existing.get("requirements"), list):
                    existing_reqs = [str(x) for x in existing.get("requirements") if str(x).strip()]
            except Exception:
                existing_reqs = []

        reqs: List[str]
        if requirements is None:
            reqs = list(existing_reqs)
        else:
            reqs = [str(x) for x in (requirements or []) if str(x).strip()]
            # Merge with existing to avoid losing previously recorded deps
            for x in existing_reqs:
                if x not in reqs:
                    reqs.append(x)

        # Filter to safe/simple requirement lines
        try:
            from app.utils.dependency_utils import is_simple_requirement, append_requirements, workspace_requirements_path

            reqs = [r for r in reqs if is_simple_requirement(r)]
            if reqs:
                append_requirements(d / "requirements.txt", reqs)
                append_requirements(workspace_requirements_path(), reqs)
        except Exception:
            pass

        # Write files
        files = dict(files or {})
        for rel, content in files.items():
            rel = str(rel or "").strip()
            if not rel:
                continue
            # Restrict file writes to within module dir.
            target = safe_resolve(d, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content or ""), encoding="utf-8")

        now = _now_iso()
        created_at = now
        if manifest_path.exists():
            # preserve original created_at if possible
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and existing.get("created_at"):
                    created_at = str(existing.get("created_at"))
            except Exception:
                pass

        manifest = {
            "name": name,
            "description": description or "",
            "entrypoint": entrypoint,
            "created_at": created_at,
            "updated_at": now,
            "requirements": reqs,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        return self.get_module(name) or WorkspaceModule(
            name=name,
            description=description or "",
            entrypoint=entrypoint,
            rel_path=f"modules/{name}",
            created_at=created_at,
            updated_at=now,
            requirements=reqs,
            files=list(files.keys()),
        )

    # -------------------------
    # Execution
    # -------------------------

    def run_module(
        self,
        *,
        name: str,
        args: Optional[List[str]] = None,
        timeout_s: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        mod = self.get_module(name)
        if mod is None:
            raise FileNotFoundError(f"Module not found: {name}")

        d = self._module_dir(mod.name)
        entry = d / (mod.entrypoint or "run.py")
        if not entry.exists():
            raise FileNotFoundError(f"Module entrypoint not found: {entry}")

        # Safety: always execute via allowlisted base command.
        cmd = ["python", str(entry.name)]
        for a in (args or []):
            cmd.append(str(a))

        # Default timeout is conservative because modules may be user-authored.
        if timeout_s is None:
            timeout_s = float(getattr(settings, "workspace_module_run_timeout_s", 60.0) or 60.0)

        merged_env: Dict[str, str] = {"WORKSPACE_ROOT": str(Path(settings.workspace_root).resolve())}
        if env:
            for k, v in env.items():
                if k and v is not None:
                    merged_env[str(k)] = str(v)

        # Optional: ensure declared deps are installed before execution.
        if getattr(settings, "auto_install_deps", True):
            try:
                from app.utils.dependency_utils import append_requirements, pip_install_requirements, workspace_requirements_path

                declared = list(getattr(mod, "requirements", []) or [])
                if declared:
                    # Keep module + workspace requirements.txt up to date
                    append_requirements(d / "requirements.txt", declared)
                    append_requirements(workspace_requirements_path(), declared)
                    # Install them into the environment
                    pip_install_requirements(d / "requirements.txt", cwd=d)
            except Exception:
                # Never block module execution just because auto-install failed
                pass

        res = run_command(cmd, cwd=d, timeout_s=int(timeout_s), env=merged_env)

        # If the module failed due to missing imports, attempt an auto-install + one retry.
        if res.returncode != 0 and getattr(settings, "auto_install_deps", True):
            try:
                from app.utils.dependency_utils import (
                    append_requirements,
                    extract_missing_modules,
                    module_to_pip_package,
                    pip_install,
                    workspace_requirements_path,
                )

                missing = extract_missing_modules((res.stderr or "") + "\n" + (res.stdout or ""))
                pkgs: List[str] = []
                for m in missing:
                    p = module_to_pip_package(m)
                    if p and p not in pkgs:
                        pkgs.append(p)

                if pkgs:
                    # Record into requirements files
                    append_requirements(d / "requirements.txt", pkgs)
                    append_requirements(workspace_requirements_path(), pkgs)

                    # Also update module.json's requirements list
                    try:
                        manifest_path = d / "module.json"
                        data = json.loads(manifest_path.read_text(encoding="utf-8"))
                        if isinstance(data, dict):
                            r0 = data.get("requirements")
                            if not isinstance(r0, list):
                                r0 = []
                            for p in pkgs:
                                if p not in r0:
                                    r0.append(p)
                            data["requirements"] = r0
                            manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass

                    # Install and retry once
                    pip_install(pkgs, cwd=d)
                    res = run_command(cmd, cwd=d, timeout_s=int(timeout_s), env=merged_env)
            except Exception:
                pass

        return res
