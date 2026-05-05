from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from app.core.config import settings
from app.utils.safe_exec import ExecResult, run_command


# Common patterns emitted by Python when imports fail.
_RE_MISSING_1 = re.compile(r"ModuleNotFoundError:\s*No module named ['\"]([^'\"]+)['\"]")
_RE_MISSING_2 = re.compile(r"ImportError:\s*No module named ['\"]([^'\"]+)['\"]")
_RE_MISSING_3 = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
_RE_MISSING_4 = re.compile(r"No module named\s+([A-Za-z0-9_\.]+)")


# Heuristic mapping from import-name -> pip package name.
# Note: many packages share the same name as their import, so we only map the common mismatches.
_MODULE_TO_PIP: Dict[str, str] = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "Crypto": "pycryptodome",
    "cryptodome": "pycryptodome",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "jwt": "pyjwt",
}


def _root_module(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    # Drop quotes and whitespace
    n = n.strip("'\" ")
    # Only keep root module for dotted imports (e.g., matplotlib.pyplot -> matplotlib)
    if "." in n:
        n = n.split(".", 1)[0]
    return n


def extract_missing_modules(text: str) -> List[str]:
    """Extract missing import root module names from stderr/stdout text."""

    blob = text or ""
    found: List[str] = []
    for rx in (_RE_MISSING_1, _RE_MISSING_2, _RE_MISSING_3):
        for m in rx.findall(blob):
            r = _root_module(m)
            if r:
                found.append(r)

    for m in _RE_MISSING_4.findall(blob):
        r = _root_module(m)
        if r:
            found.append(r)

    # De-dup while preserving order
    out: List[str] = []
    seen: Set[str] = set()
    for x in found:
        key = x.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def module_to_pip_package(module_name: str) -> str:
    """Convert a Python import name to an installable pip package name."""

    root = _root_module(module_name)
    if not root:
        return ""
    return _MODULE_TO_PIP.get(root, root)


def is_module_available(module_name: str) -> bool:
    """Return True if importlib can find the module."""

    root = _root_module(module_name)
    if not root:
        return False
    try:
        return importlib.util.find_spec(root) is not None
    except Exception:
        return False


def _req_name(req: str) -> str:
    """Extract the package name portion from a requirements spec."""

    s = (req or "").strip()
    if not s:
        return ""
    # Drop environment markers
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    # Drop direct URL refs and extras
    # We keep only leading name-ish token.
    m = re.match(r"^([A-Za-z0-9_.-]+)", s)
    if not m:
        return ""
    name = m.group(1)
    # Drop extras if present
    if "[" in name:
        name = name.split("[", 1)[0]
    return name.strip().lower()


def is_simple_requirement(req: str) -> bool:
    """Reject dangerous/complex requirement lines (URLs, editable installs, options)."""

    s = (req or "").strip()
    if not s:
        return False
    if s.startswith("-"):
        return False
    if "\n" in s or "\r" in s:
        return False
    if "http://" in s or "https://" in s:
        return False
    if "git+" in s:
        return False
    if "@" in s:
        # PEP 508 direct references use '@'. We disallow by default for safety.
        return False
    # Basic allowlist of characters
    if not re.match(r"^[A-Za-z0-9_\-\[\]\.\+<>=!~,;: ]+$", s):
        return False
    return True


def append_requirements(req_file: Path, requirements: Iterable[str]) -> List[str]:
    """Append requirements to a requirements.txt, avoiding duplicates by name.

    Returns the list of newly-added requirement lines.
    """

    req_file = Path(req_file)
    req_file.parent.mkdir(parents=True, exist_ok=True)
    if not req_file.exists():
        req_file.write_text("", encoding="utf-8")

    existing_lines = []
    try:
        existing_lines = req_file.read_text(encoding="utf-8").splitlines()
    except Exception:
        existing_lines = []

    existing_names: Set[str] = set()
    for line in existing_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        nm = _req_name(line)
        if nm:
            existing_names.add(nm)

    added: List[str] = []
    for r in requirements:
        r = (str(r or "").strip())
        if not r:
            continue
        if not is_simple_requirement(r):
            # Skip unsafe lines
            continue
        nm = _req_name(r)
        if not nm:
            continue
        if nm in existing_names:
            continue
        existing_names.add(nm)
        added.append(r)

    if not added:
        return []

    # Append with a newline boundary
    out_lines = list(existing_lines)
    if out_lines and out_lines[-1].strip() != "":
        out_lines.append("")
    out_lines.append("# Added by Agentic Hive (auto dependency install)")
    out_lines.extend(added)
    out_lines.append("")

    req_file.write_text("\n".join(out_lines), encoding="utf-8")
    return added


def workspace_requirements_path() -> Path:
    return Path(settings.workspace_root).resolve() / "requirements.txt"


def pip_install(packages: List[str], *, cwd: Path, timeout_s: Optional[int] = None) -> ExecResult:
    """Install packages into the current Python environment."""

    pkgs = [str(p).strip() for p in (packages or []) if str(p).strip()]
    if not pkgs:
        return ExecResult(returncode=0, stdout="", stderr="")

    # Safety: filter to simple requirements
    safe_pkgs = [p for p in pkgs if is_simple_requirement(p)]
    if not safe_pkgs:
        return ExecResult(returncode=1, stdout="", stderr="No valid/safe packages to install")

    if timeout_s is None:
        timeout_s = int(getattr(settings, "pip_install_timeout_s", 600) or 600)

    cmd = [
        "python",
        "-m",
        "pip",
        "install",
        "--no-input",
        "--disable-pip-version-check",
        *safe_pkgs,
    ]
    return run_command(cmd, cwd=Path(cwd), timeout_s=int(timeout_s))


def pip_install_requirements(req_file: Path, *, cwd: Path, timeout_s: Optional[int] = None) -> ExecResult:
    req_file = Path(req_file)
    if not req_file.exists():
        return ExecResult(returncode=0, stdout="", stderr="")

    if timeout_s is None:
        timeout_s = int(getattr(settings, "pip_install_timeout_s", 600) or 600)

    cmd = [
        "python",
        "-m",
        "pip",
        "install",
        "--no-input",
        "--disable-pip-version-check",
        "-r",
        str(req_file),
    ]
    return run_command(cmd, cwd=Path(cwd), timeout_s=int(timeout_s))
