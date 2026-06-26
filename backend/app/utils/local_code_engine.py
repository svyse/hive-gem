from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from app.core.config import settings
from app.utils.file_utils import list_files, read_text, safe_resolve
from app.utils.nlp_code_planner import (
    extract_files_from_nlp_output,
    infer_test_commands,
    prompt_mentions_interactive_input,
)
from app.utils.local_code_quality import (
    build_quality_lesson_payload,
    file_generation_rules_for_path,
    short_quality_issue_summary,
    validate_generated_file_content,
    validate_generated_path_for_prompt,
)
from app.utils.repetition_guard import (
    is_repetitive_text,
    sanitize_operation_content,
    sanitize_operations,
    truncate_repetitive_tail,
)


PROJECT_ENGINE_MANIFEST_SYSTEM = """You are a local code engine that creates and edits real project files from natural-language prompts.

You are running inside a safe code pipeline. Do NOT output JSON.

You are not a calculator template or a hello-world template. Read the actual user prompt, infer the requested language/framework/features, and design the project structure needed for that prompt.

Your current job is to design the file manifest only. Return only this shape:

PROJECT_FILES:
- relative/path.ext: one short purpose sentence

TEST_COMMANDS:
- command to validate the project, if safe and non-interactive

Rules:
- Paths must be relative to the project root.
- For a new project, include the full runnable project structure: entrypoint, reusable modules, tests when practical, README, config/requirements, and ignore files.
- For an edit/debug request, include only the files that need to be created or changed.
- If the prompt asks for terminal input, include a runnable entrypoint and tests for reusable logic.
- Use WEB_RESEARCH_CONTEXT when present for current APIs, file layout conventions, and examples, but do not copy irrelevant website text.
- Keep the manifest compact and practical for a local model.
- Do not include explanations outside PROJECT_FILES and TEST_COMMANDS.
"""


PROJECT_ENGINE_FILE_SYSTEM = """You are a local code engine that writes complete project files.

Return the complete content for exactly ONE requested file.
Do not output JSON.
Do not wrap the answer in markdown unless the file itself is markdown.
Do not include explanations before or after the file content.
If editing an existing file, return the full updated file content, not a diff.

Important quality rules:
- Source files must contain source code for their extension, not setup notes, transcripts, or README prose.
- Python files must be valid Python source; never put commands like python --version, pip install, git clone, cd, or pyenv into a .py file.
- Test files must contain executable tests, not usage instructions.
- requirements.txt contains package specifiers only, or can be empty for stdlib-only projects.
- .gitignore/.dockerignore contain ignore patterns only.
- If the user asks for user input, the runnable entrypoint must read input and print the requested result.
- Use WEB_RESEARCH_CONTEXT as reference material only; still write clean, minimal, runnable project files.
"""


PROJECT_ENGINE_REPAIR_SYSTEM = """You are a local code repair engine.

The project has compile/test/runtime errors. Decide which files must be edited and then write complete replacement contents for those files.
Do NOT output JSON. Use relative project paths only.
"""


_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".workspace",
    ".workspaces",
    ".memory",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    "dist",
    "build",
    "coverage",
    "target",
}

_BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".pyc",
    ".pyd",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
}

_CODE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".css",
    ".json",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".txt",
    ".sh",
    ".bat",
    ".ps1",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".rb",
    ".sql",
    ".env",
    ".gitignore",
    ".dockerignore",
    "Dockerfile",
}

_COMMAND_PATH_HEADS = {
    "python", "python3", "py", "pip", "pip3", "git", "cd", "npm", "yarn", "pnpm",
    "npx", "pyenv", "curl", "wget", "mkdir", "touch", "copy", "move", "del", "rm",
    "rmdir", "powershell", "cmd",
}
_PLACEHOLDER_PATH_RE = re.compile(r"^(?:file|module|script|data)_\d+(?:\.[A-Za-z0-9_+\-]{1,16})?$", re.IGNORECASE)

_PATH_LINE_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:FILE\s*:\s*)?`?([A-Za-z0-9_./@+\- ]+(?:\.[A-Za-z0-9_+\-]{1,16}|Dockerfile))`?\s*(?::|\-|--|\u2013|\u2014|$)"
)
_TEST_COMMANDS_RE = re.compile(r"(?ims)^\s*TEST_COMMANDS\s*:\s*(.*?)(?=^\s*PROJECT_FILES\s*:|^\s*FILE\s*:|\Z)")
_FILE_LABEL_FENCE_RE = re.compile(
    r"(?ims)^\s*`?([A-Za-z0-9_./@+\- ]+\.[A-Za-z0-9_+\-]{1,16})`?\s*:?\s*\n\s*```\s*([A-Za-z0-9_+.#\-]*)?\s*\n([\s\S]*?)\n\s*```"
)
_FENCE_RE = re.compile(r"(?ims)```\s*([A-Za-z0-9_+.#\-]*)?\s*\n([\s\S]*?)\n\s*```")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on", "y"}:
        return True
    if value in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _setting_int(attr: str, env_name: str, default: int) -> int:
    raw = os.getenv(env_name)
    if raw is None:
        raw = getattr(settings, attr, default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _setting_bool(attr: str, env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        raw = getattr(settings, attr, default)
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on", "y"}:
        return True
    if value in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _quality_validate_enabled() -> bool:
    return _setting_bool("local_code_engine_validate_files", "LOCAL_CODE_ENGINE_VALIDATE_FILES", True)


def env_code_engine_enabled() -> bool:
    return _env_bool("LOCAL_CODE_ENGINE_ENABLED", bool(getattr(settings, "local_code_engine_enabled", True)))


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\\", "/").strip().lower())


def _path_has_skip_part(path: str) -> bool:
    parts = [p for p in str(path or "").replace("\\", "/").split("/") if p]
    return any(part in _SKIP_DIRS for part in parts)


def _normalize_path(raw_path: str) -> Optional[str]:
    p = str(raw_path or "").strip().strip("`'\"")
    p = p.replace("\\", "/")
    p = re.sub(r"\s+", " ", p).strip()
    p = re.sub(r"^(?:path|file(?:name)?)\s*=\s*", "", p, flags=re.IGNORECASE).strip()
    while p.startswith("./"):
        p = p[2:]
    if not p:
        return None
    drive = p.split("/", 1)[0]
    if ":" in drive or p.startswith("/") or p.startswith("~"):
        return None
    parts = [part.strip() for part in p.split("/") if part.strip()]
    if not parts:
        return None
    if any(part in {".", ".."} for part in parts):
        return None
    if any(part in _SKIP_DIRS for part in parts):
        return None
    if any(part.startswith(".") and part not in {".gitignore", ".dockerignore", ".env.example"} for part in parts):
        return None
    normalized = str(PurePosixPath(*parts))
    if normalized in {".", ""}:
        return None
    # Keep filenames normal enough for Windows + POSIX local editing.
    if not re.match(r"^[A-Za-z0-9._/@+\- ]+$", normalized):
        return None
    # Do not let shell commands or code-fence inference placeholders become
    # project paths. Examples seen from small local models: `python main.py`,
    # `pip install -r`, `file_1.java`, `module_6.py`, `data_1.json`.
    if any(" " in part for part in parts):
        return None
    first_token = re.split(r"\s+", parts[0].lower(), maxsplit=1)[0]
    if first_token in _COMMAND_PATH_HEADS:
        return None
    name = parts[-1]
    if _PLACEHOLDER_PATH_RE.match(name):
        return None
    suffix = Path(name).suffix.lower()
    if suffix in _BINARY_SUFFIXES:
        return None
    if not suffix and name not in {"Dockerfile", "Makefile", "Procfile", ".gitignore", ".dockerignore", ".env.example", "README.md", "requirements.txt", "pyproject.toml", "package.json"}:
        return None
    return normalized


def _is_probably_text_file(path: str) -> bool:
    name = Path(path).name
    suffix = Path(path).suffix.lower()
    if name in {"Dockerfile", "Makefile", "Procfile"}:
        return True
    if suffix in _BINARY_SUFFIXES:
        return False
    return suffix in _CODE_SUFFIXES or not suffix


def _language_hint(prompt: str, project_root: Path | None = None) -> str:
    q = _compact(prompt)
    if any(x in q for x in ("fastapi", "flask", "django", "python", "pytest", "unittest", "pip")):
        return "python"
    if any(x in q for x in ("react", "vite", "tsx", "typescript")):
        return "typescript"
    if any(x in q for x in ("javascript", "node", "express", "npm", "browser", "html", "css", "website", "web page")):
        return "javascript"
    if any(x in q for x in ("java", "spring")):
        return "java"
    if any(x in q for x in ("golang", "go ", " go")):
        return "go"
    if any(x in q for x in ("rust", "cargo")):
        return "rust"
    try:
        if project_root is not None:
            project_name = project_root.name.lower()
            if re.search(r"(?:^|[_\-/])py(?:[_\-/]|$|v\d+)", project_name):
                return "python"
            if project_name.endswith(("_js", "-js")):
                return "javascript"
            if project_name.endswith(("_ts", "-ts")):
                return "typescript"
            if (project_root / "package.json").exists():
                return "javascript"
            if (project_root / "pyproject.toml").exists() or (project_root / "requirements.txt").exists():
                return "python"
            if list(project_root.glob("*.py")):
                return "python"
    except Exception:
        pass
    return "python"


def _request_mode(prompt: str, project_root: Path, error_text: str = "") -> str:
    q = _compact(prompt + "\n" + error_text)
    existing = False
    try:
        existing = any(project_root.iterdir())
    except Exception:
        existing = False
    if any(k in q for k in ("fix", "debug", "bug", "error", "traceback", "exception", "failing", "edit", "modify", "update", "change", "refactor")):
        return "edit_debug"
    return "edit_debug" if existing and q.startswith(("add ", "update ", "change ", "modify ")) else "create_project"


def _score_file(path: str, prompt: str, error_text: str) -> int:
    p = path.replace("\\", "/")
    lower = p.lower()
    q = _compact(prompt)
    err = _compact(error_text)
    score = 0
    base = Path(p).name.lower()
    if base and base in q:
        score += 80
    if lower in err or base in err:
        score += 100
    if lower in q:
        score += 60
    if lower.startswith(("src/", "app/", "backend/", "frontend/")):
        score += 15
    if base in {"main.py", "app.py", "index.js", "index.ts", "index.html", "package.json", "pyproject.toml", "requirements.txt", "readme.md"}:
        score += 35
    if "/test" in lower or lower.startswith("test"):
        score += 10
    suffix = Path(p).suffix.lower()
    lang = _language_hint(prompt)
    if lang == "python" and suffix == ".py":
        score += 20
    if lang == "javascript" and suffix in {".js", ".jsx", ".html", ".css", ".json"}:
        score += 20
    if lang == "typescript" and suffix in {".ts", ".tsx", ".json", ".html", ".css"}:
        score += 20
    # Prefer smaller, top-level files when otherwise tied.
    score -= min(20, lower.count("/") * 2)
    return score


def collect_project_snapshot(
    project_root: Path,
    *,
    prompt: str,
    error_text: str = "",
    max_files: Optional[int] = None,
    max_file_chars: Optional[int] = None,
) -> Dict[str, Any]:
    max_files = max_files or _setting_int("local_code_engine_max_context_files", "LOCAL_CODE_ENGINE_MAX_CONTEXT_FILES", 24)
    max_file_chars = max_file_chars or _setting_int("local_code_engine_max_file_chars", "LOCAL_CODE_ENGINE_MAX_FILE_CHARS", 4500)

    try:
        all_files = [p for p in list_files(project_root, max_files=3000) if not _path_has_skip_part(p) and _is_probably_text_file(p)]
    except Exception:
        all_files = []

    tree = "\n".join(all_files[:250])
    scored = sorted(all_files, key=lambda p: _score_file(p, prompt, error_text), reverse=True)
    chosen = scored[: max(1, int(max_files))]

    snapshots: List[Dict[str, str]] = []
    for rel in chosen:
        try:
            p = safe_resolve(project_root, rel)
            if p.exists() and p.stat().st_size > max(1, max_file_chars) * 10:
                continue
            content = read_text(project_root, rel)
            if len(content) > max_file_chars:
                content = content[:max_file_chars] + "\n...<truncated>...\n"
            snapshots.append({"path": rel, "content": content})
        except Exception:
            continue

    return {
        "tree": tree,
        "files": snapshots,
        "all_files": all_files,
        "file_count": len(all_files),
        "selected_count": len(snapshots),
    }


def _paths_from_error_text(project_root: Path, error_text: str, existing_files: Iterable[str]) -> List[str]:
    err = str(error_text or "")
    existing = {p.replace("\\", "/"): p for p in existing_files}
    out: List[str] = []

    # Python traceback shape: File ".../path.py", line N
    for m in re.finditer(r"File\s+\"([^\"]+)\"", err):
        raw = m.group(1).replace("\\", "/")
        rel = None
        try:
            abs_path = Path(raw)
            if abs_path.is_absolute():
                rel = str(abs_path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
        except Exception:
            rel = None
        if rel is None:
            for known in existing:
                if raw.endswith(known):
                    rel = existing[known]
                    break
        norm = _normalize_path(rel or raw)
        if norm and norm in existing and norm not in out:
            out.append(norm)

    for known in existing:
        if known.lower() in err.lower() and known not in out:
            out.append(known)
    return out[:8]


def parse_test_commands(text: str) -> List[str]:
    commands: List[str] = []
    for m in _TEST_COMMANDS_RE.finditer(str(text or "")):
        section = m.group(1)
        for line in section.splitlines():
            s = line.strip()
            if not s:
                continue
            s = re.sub(r"^[-*]\s*", "", s).strip().strip("` ")
            if not s or s.lower().startswith(("project_files", "file:")):
                continue
            lowered = s.lower()
            safe_prefixes = (
                "python ",
                "python3 ",
                "py ",
                "pytest",
                "npm test",
                "node ",
                "go test",
                "cargo test",
            )
            unsafe = (" rm ", " del ", " rmdir ", "format ", "shutdown", "curl ", "wget ", "powershell")
            if lowered.startswith(safe_prefixes) and not any(x in f" {lowered} " for x in unsafe):
                commands.append(s)
    return list(dict.fromkeys(commands))[:5]


def parse_manifest_paths(text: str, *, prompt: str = "", project_name: str = "") -> List[str]:
    raw = truncate_repetitive_tail(str(text or ""))
    if not raw.strip() or is_repetitive_text(raw):
        return []

    # Parse only the file-list part so commands like "python main.py" do not
    # become fake paths.
    file_section = re.split(r"(?im)^\s*TEST_COMMANDS\s*:", raw, maxsplit=1)[0]
    m_section = re.search(r"(?ims)^\s*PROJECT_FILES\s*:\s*(.*)", file_section)
    if m_section:
        file_section = m_section.group(1)

    paths: List[str] = []
    for m in _PATH_LINE_RE.finditer(file_section):
        norm = _normalize_path(m.group(1))
        if norm and _is_probably_text_file(norm) and norm not in paths:
            ok_path, _path_issues = validate_generated_path_for_prompt(norm, prompt=prompt, project_name=project_name)
            if ok_path:
                paths.append(norm)
    # Also accept FILE blocks as a manifest.
    for path, _content in extract_files_from_nlp_output(file_section, user_prompt=""):
        norm = _normalize_path(path)
        if norm and norm not in paths:
            ok_path, _path_issues = validate_generated_path_for_prompt(norm, prompt=prompt, project_name=project_name)
            if ok_path:
                paths.append(norm)
    return paths[:50]


def _ensure_support_files(paths: List[str], *, prompt: str, project_root: Path, mode: str) -> List[str]:
    out = []
    for p in paths:
        norm = _normalize_path(p)
        if not norm:
            continue
        ok_path, _path_issues = validate_generated_path_for_prompt(norm, prompt=prompt, project_name=project_root.name)
        if ok_path and norm not in out:
            out.append(norm)
    if mode == "edit_debug":
        return out

    lang = _language_hint(prompt, project_root)
    lower = {p.lower() for p in out}

    def add(path: str) -> None:
        if path.lower() not in lower:
            out.append(path)
            lower.add(path.lower())

    # For new projects, make the folder feel complete rather than just a lone file.
    add("README.md")
    add(".gitignore")

    if lang == "python":
        if not any(p.endswith(".py") and not p.startswith("tests/") for p in out):
            add("main.py")
        if not any(p.startswith("tests/") and p.endswith(".py") for p in out):
            add("tests/test_basic.py")
        if not any(p in {"requirements.txt", "pyproject.toml"} for p in out):
            add("requirements.txt")
    elif lang in {"javascript", "typescript"}:
        if not any(p.endswith((".js", ".ts", ".jsx", ".tsx")) for p in out):
            add("src/index.ts" if lang == "typescript" else "src/index.js")
        if "package.json" not in lower:
            add("package.json")
        if any(k in _compact(prompt) for k in ("web", "website", "html", "browser")):
            add("index.html")
            add("style.css")
    return out


def _default_manifest_paths(prompt: str, project_root: Path, *, mode: str, error_text: str, existing_files: Iterable[str]) -> List[str]:
    q = _compact(prompt)
    if mode == "edit_debug":
        err_paths = _paths_from_error_text(project_root, error_text, existing_files)
        if err_paths:
            return err_paths
        # Fall back to likely entrypoints in the existing project.
        candidates = [p for p in existing_files if Path(p).name.lower() in {"main.py", "app.py", "index.js", "index.ts", "index.html"}]
        return list(candidates[:5])

    if any(k in q for k in ("fastapi", "api server", "rest api")):
        return ["app/main.py", "requirements.txt", "tests/test_basic.py", "README.md", ".gitignore"]
    if "flask" in q:
        return ["app.py", "templates/index.html", "static/style.css", "requirements.txt", "tests/test_basic.py", "README.md", ".gitignore"]
    if any(k in q for k in ("web page", "website", "html", "css", "browser game", "frontend")) and not any(k in q for k in ("react", "vite")):
        return ["index.html", "style.css", "app.js", "README.md", ".gitignore"]
    if any(k in q for k in ("react", "vite")):
        return ["package.json", "index.html", "src/App.jsx", "src/main.jsx", "src/App.css", "README.md", ".gitignore"]
    if any(k in q for k in ("node", "express", "javascript")):
        return ["package.json", "src/index.js", "README.md", ".gitignore"]
    if "typescript" in q or " ts " in q:
        return ["package.json", "tsconfig.json", "src/index.ts", "README.md", ".gitignore"]
    return ["main.py", "tests/test_basic.py", "README.md", "requirements.txt", ".gitignore"]


def _default_content_for_path(path: str, *, prompt: str, project_name: str) -> Optional[str]:
    name = Path(path).name
    suffix = Path(path).suffix.lower()
    if name == ".gitignore":
        return "__pycache__/\n*.py[cod]\n.venv/\nvenv/\n.env\nnode_modules/\ndist/\nbuild/\n.coverage\n.pytest_cache/\n"
    if name == "requirements.txt":
        return ""
    if name == "README.md":
        return (
            f"# {project_name}\n\n"
            "Generated by the local code pipeline from this prompt:\n\n"
            f"> {str(prompt or '').strip()}\n\n"
            "## Run\n\n"
            "Review the entrypoint generated for this project and run the relevant command from the code pipeline logs.\n"
        )
    if path == "tests/test_basic.py":
        return (
            "import unittest\n"
            "from pathlib import Path\n\n\n"
            "class ProjectStructureTests(unittest.TestCase):\n"
            "    def test_project_has_entrypoint(self) -> None:\n"
            "        self.assertTrue(Path('main.py').exists() or Path('app.py').exists() or Path('app/main.py').exists())\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
    if suffix == ".py" and name in {"main.py", "app.py"}:
        if prompt_mentions_interactive_input(prompt):
            return (
                "def main() -> None:\n"
                "    print('Project generated safely, but the local model did not produce a complete valid implementation.')\n"
                "    value = input('Enter input: ')\n"
                "    print(value)\n\n\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            )
        return (
            "def main() -> None:\n"
            "    print('Project generated successfully.')\n\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
    if name == "package.json":
        return (
            "{\n"
            f"  \"name\": \"{re.sub(r'[^a-z0-9-]+', '-', project_name.lower()).strip('-') or 'local-project'}\",\n"
            "  \"version\": \"1.0.0\",\n"
            "  \"type\": \"module\",\n"
            "  \"scripts\": {\n"
            "    \"start\": \"node src/index.js\",\n"
            "    \"test\": \"node --test\"\n"
            "  }\n"
            "}\n"
        )
    if suffix == ".html":
        return "<!doctype html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\"><title>Local Project</title></head>\n<body><main id=\"app\"></main><script src=\"app.js\"></script></body>\n</html>\n"
    if suffix == ".css":
        return "body { font-family: system-ui, sans-serif; margin: 2rem; }\n"
    if suffix in {".js", ".ts"}:
        return "console.log('Project generated successfully.');\n"
    return None


def _matching_file_block(raw: str, path: str, prompt: str) -> Optional[str]:
    files = extract_files_from_nlp_output(raw, user_prompt=prompt)
    if not files:
        return None
    norm_target = _normalize_path(path)
    for file_path, content in files:
        if _normalize_path(file_path) == norm_target:
            return content
    if len(files) == 1:
        return files[0][1]
    return None


def clean_single_file_output(raw_text: str, *, path: str, prompt: str) -> Optional[str]:
    raw = truncate_repetitive_tail(str(raw_text or "")).strip()
    if not raw or is_repetitive_text(raw):
        return None

    from_block = _matching_file_block(raw, path, prompt)
    if from_block is not None:
        raw = from_block.strip("\n")
    else:
        label_match = None
        for m in _FILE_LABEL_FENCE_RE.finditer(raw):
            if _normalize_path(m.group(1)) == _normalize_path(path):
                label_match = m
                break
        if label_match is not None:
            raw = label_match.group(3).strip("\n")
        else:
            fences = list(_FENCE_RE.finditer(raw))
            if len(fences) == 1:
                raw = fences[0].group(2).strip("\n")
            elif len(fences) > 1:
                raw = fences[0].group(2).strip("\n")
            else:
                # Remove common chatter while keeping markdown files intact.
                if Path(path).suffix.lower() not in {".md", ".txt"}:
                    lines = raw.splitlines()
                    while lines and re.match(r"(?i)^\s*(here is|sure[,!]?|below is|this file|the complete)", lines[0].strip()):
                        lines.pop(0)
                    if lines and re.match(r"(?i)^\s*FILE\s*:", lines[0].strip()):
                        lines.pop(0)
                    raw = "\n".join(lines).strip("\n")

    try:
        cleaned = sanitize_operation_content(raw.rstrip() + "\n", max_chars=180_000)
    except ValueError:
        return None
    if not cleaned.strip() or is_repetitive_text(cleaned):
        return None
    return cleaned


def _format_memory_items(items: Any, *, max_items: int = 6, max_chars: int = 6000) -> str:
    if not isinstance(items, list):
        return ""
    chunks: List[str] = []
    for item in items[:max_items]:
        if not isinstance(item, dict):
            continue
        tags = item.get("tags") or []
        success = item.get("success")
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        chunks.append(f"tags={tags} success={success} created_at={item.get('created_at')}\n{content[:1200]}")
        if len("\n\n".join(chunks)) >= max_chars:
            break
    return "\n\n".join(chunks)[:max_chars]



def _format_web_research_context(web: Any) -> str:
    if not isinstance(web, dict) or not web:
        return ""
    if not web.get("enabled") and not web.get("items"):
        skipped = web.get("skipped") or web.get("error")
        return f"WEB_RESEARCH: disabled/skipped ({skipped})" if skipped else ""

    chunks: List[str] = []
    queries = web.get("queries") or []
    if isinstance(queries, list) and queries:
        chunks.append("QUERIES: " + "; ".join(str(q) for q in queries[:6]))

    for item in (web.get("items") or [])[:4]:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if query:
            chunks.append(f"QUERY: {query}")

        summary = item.get("summary")
        if isinstance(summary, dict):
            key_points = summary.get("key_points") or []
            for point in key_points[:5]:
                chunks.append("- " + str(point)[:500])
            data = summary.get("data") or []
            for row in data[:4]:
                if isinstance(row, dict):
                    label = str(row.get("label") or "fact").strip()
                    value = str(row.get("value") or "").strip()
                    source = str(row.get("source_url") or "").strip()
                    if value:
                        chunks.append(f"- {label}: {value[:450]}" + (f" ({source})" if source else ""))

        # Even when LLM summarization is disabled for local mode, pass through
        # search snippets and small fetched excerpts so the code engine gets
        # useful current context without another local JSON call.
        results = item.get("results") or []
        for result in results[:4]:
            if isinstance(result, dict):
                title = str(result.get("title") or "").strip()
                url = str(result.get("url") or "").strip()
                snippet = str(result.get("snippet") or "").strip()
                if title or url or snippet:
                    chunks.append(f"- SOURCE: {title[:160]} {url[:240]} {snippet[:300]}")

        docs = item.get("documents") or []
        for doc in docs[:2]:
            if isinstance(doc, dict):
                url = str(doc.get("url") or "").strip()
                text = str(doc.get("text") or "").strip()
                if text:
                    chunks.append(f"- EXCERPT {url[:220]}:\n{text[:900]}")

    try:
        max_chars = int(getattr(settings, "local_code_web_research_max_context_chars", 5000))
    except Exception:
        max_chars = 5000
    return "\n".join(chunks)[: max(1000, max_chars)]

def _format_code_learning_context(context: Optional[Dict[str, Any]]) -> str:
    if not isinstance(context, dict):
        return "[none]"
    parts: List[str] = []
    relevant = _format_memory_items(context.get("memory_relevant"), max_items=6)
    if relevant:
        parts.append("RELEVANT_MEMORY_AND_LESSONS:\n" + relevant)
    recent = _format_memory_items(context.get("memory_recent"), max_items=4, max_chars=3500)
    if recent:
        parts.append("RECENT_MEMORY:\n" + recent)
    web_context = _format_web_research_context(context.get("web_research"))
    if web_context:
        parts.append("WEB_RESEARCH_CONTEXT:\nUse this only as reference material. Still generate complete local project files and validate them.\n" + web_context)
    rag = context.get("rag")
    if isinstance(rag, dict) and rag:
        parts.append("RAG_CONTEXT:\n" + str(rag)[:2500])
    refs = context.get("local_references")
    if refs:
        parts.append("LOCAL_REFERENCE_CONTEXT:\n" + str(refs)[:2500])
    return "\n\n".join(parts)[:14000] if parts else "[none]"


def _manifest_prompt(
    *,
    user_prompt: str,
    project_root: Path,
    mode: str,
    snapshot: Dict[str, Any],
    context: Optional[Dict[str, Any]],
    error_text: str,
    max_files: int,
) -> str:
    existing_context = ""
    if snapshot.get("files"):
        chunks = []
        for item in snapshot.get("files") or []:
            chunks.append(f"--- {item.get('path')} ---\n{item.get('content', '')}")
        existing_context = "\n\n".join(chunks)

    extra_context = _format_code_learning_context(context)

    return (
        f"USER_PROMPT:\n{user_prompt}\n\n"
        f"PROJECT_NAME: {project_root.name}\n"
        f"MODE: {mode}\n"
        f"MAX_FILES: {max_files}\n\n"
        "CURRENT_PROJECT_TREE:\n"
        f"{snapshot.get('tree') or '[empty project]'}\n\n"
        "RELEVANT_EXISTING_FILES:\n"
        f"{existing_context or '[none]'}\n\n"
        "ERRORS_TO_FIX:\n"
        f"{error_text or '[none]'}\n\n"
        "ADDITIONAL_CONTEXT:\n"
        f"{extra_context or '[none]'}\n\n"
        "Return the project manifest now. For a new project, include the complete runnable project structure. "
        "For editing/debugging, include only files that should be changed."
    )


def _file_prompt(
    *,
    user_prompt: str,
    project_root: Path,
    path: str,
    manifest_text: str,
    snapshot: Dict[str, Any],
    mode: str,
    context: Optional[Dict[str, Any]],
    error_text: str,
) -> str:
    current = ""
    try:
        p = safe_resolve(project_root, path)
        if p.exists() and p.is_file():
            current = p.read_text(encoding="utf-8")
            limit = _setting_int("local_code_engine_max_file_chars", "LOCAL_CODE_ENGINE_MAX_FILE_CHARS", 4500)
            if len(current) > limit:
                current = current[:limit] + "\n...<truncated>...\n"
    except Exception:
        current = ""

    relevant = ""
    chunks = []
    for item in snapshot.get("files") or []:
        item_path = str(item.get("path") or "")
        if item_path == path:
            continue
        chunks.append(f"--- {item_path} ---\n{item.get('content', '')}")
        if len("\n\n".join(chunks)) > 7000:
            break
    relevant = "\n\n".join(chunks)

    return (
        f"USER_PROMPT:\n{user_prompt}\n\n"
        f"PROJECT_NAME: {project_root.name}\n"
        f"MODE: {mode}\n"
        f"REQUESTED_FILE: {path}\n\n"
        "FILE_SPECIFIC_RULES:\n"
        f"{file_generation_rules_for_path(path, prompt=user_prompt)}\n\n"
        "PROJECT_MANIFEST:\n"
        f"{manifest_text[:6000]}\n\n"
        "CURRENT_CONTENT_OF_REQUESTED_FILE:\n"
        f"{current or '[new file]'}\n\n"
        "RELEVANT_EXISTING_FILES:\n"
        f"{relevant or '[none]'}\n\n"
        "CODE_LEARNING_CONTEXT:\n"
        f"{_format_code_learning_context(context)}\n\n"
        "ERRORS_TO_FIX:\n"
        f"{error_text or '[none]'}\n\n"
        "Write only the complete content for REQUESTED_FILE. No markdown wrapper unless the file itself is markdown."
    )


async def build_local_code_engine_plan(
    *,
    llm: Any,
    project_root: Path,
    user_prompt: str,
    context: Optional[Dict[str, Any]] = None,
    log: Optional[Callable[[str], None]] = None,
    mode: Optional[str] = None,
    error_text: str = "",
    phase: str = "plan",
    prior_plan: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Create a write_file plan using local text generation instead of strict JSON.

    This is the general local code-engine path. It uses normal NLP prompts that
    small local chat/code models handle better than JSON schemas: first a compact
    manifest, then one file-content call per file. It can create new projects or
    edit/repair existing files.
    """

    if not env_code_engine_enabled():
        return None
    if llm is None or not hasattr(llm, "chat_text_async"):
        return None

    prompt = str(user_prompt or "").strip()
    if not prompt:
        return None

    max_files = _setting_int("local_code_engine_max_files", "LOCAL_CODE_ENGINE_MAX_FILES", 12)
    max_files = max(1, min(max_files, 30))
    max_context_files = _setting_int("local_code_engine_max_context_files", "LOCAL_CODE_ENGINE_MAX_CONTEXT_FILES", 24)
    max_file_chars = _setting_int("local_code_engine_max_file_chars", "LOCAL_CODE_ENGINE_MAX_FILE_CHARS", 4500)

    actual_mode = mode or _request_mode(prompt, project_root, error_text)
    if phase == "repair" and actual_mode != "edit_debug":
        actual_mode = "edit_debug"

    if actual_mode == "create_project" and not error_text:
        # A rerun into the same sample project may contain broken files from a
        # previous failed generation. For a create-project request, do not feed
        # those corrupted files back into the local model as examples to copy.
        snapshot = {
            "tree": "[existing files ignored for create_project generation]",
            "files": [],
            "all_files": [],
            "file_count": 0,
            "selected_count": 0,
        }
    else:
        snapshot = collect_project_snapshot(
            project_root,
            prompt=prompt,
            error_text=error_text,
            max_files=max_context_files,
            max_file_chars=max_file_chars,
        )

    manifest_raw = ""
    paths: List[str] = []
    test_commands: List[str] = []

    try:
        if log:
            log("local code engine: generating project manifest from NLP prompt")
        manifest_raw = await llm.chat_text_async(
            system=PROJECT_ENGINE_REPAIR_SYSTEM if phase == "repair" else PROJECT_ENGINE_MANIFEST_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": _manifest_prompt(
                        user_prompt=prompt,
                        project_root=project_root,
                        mode=actual_mode,
                        snapshot=snapshot,
                        context=context,
                        error_text=error_text,
                        max_files=max_files,
                    ),
                }
            ],
            temperature=0.15,
            purpose="code-project-manifest" if phase != "repair" else "code-project-repair-manifest",
        )
        paths = parse_manifest_paths(manifest_raw, prompt=prompt, project_name=project_root.name)
        test_commands = parse_test_commands(manifest_raw)
    except Exception as e:
        if log:
            log(f"local code engine manifest failed; using heuristic manifest: {type(e).__name__}: {e}")
        manifest_raw = ""

    if phase == "repair":
        err_paths = _paths_from_error_text(project_root, error_text, snapshot.get("all_files") or [])
        if err_paths:
            paths = list(dict.fromkeys(err_paths + paths))
        if prior_plan and not paths:
            for op in prior_plan.get("operations") or []:
                if isinstance(op, dict) and op.get("op") == "write_file":
                    norm = _normalize_path(str(op.get("path") or ""))
                    if norm and norm not in paths:
                        paths.append(norm)
        paths = paths[: min(max_files, 8)]
    else:
        if not paths:
            paths = _default_manifest_paths(
                prompt,
                project_root,
                mode=actual_mode,
                error_text=error_text,
                existing_files=snapshot.get("all_files") or [],
            )
        paths = _ensure_support_files(paths, prompt=prompt, project_root=project_root, mode=actual_mode)
        paths = paths[:max_files]

    if not paths:
        return None

    manifest_text = manifest_raw.strip() or "PROJECT_FILES:\n" + "\n".join(f"- {p}: generated file" for p in paths)
    operations: List[Dict[str, Any]] = []
    generated_files: List[Tuple[str, str]] = []
    skipped: List[str] = []
    quality_issues: List[Dict[str, Any]] = []
    validate_files = _quality_validate_enabled()
    max_file_retries = max(0, min(_setting_int("local_code_engine_file_retries", "LOCAL_CODE_ENGINE_FILE_RETRIES", 1), 3))

    for path in paths:
        norm = _normalize_path(path)
        if not norm:
            skipped.append(path)
            continue

        content: Optional[str] = None
        base_prompt = _file_prompt(
            user_prompt=prompt,
            project_root=project_root,
            path=norm,
            manifest_text=manifest_text,
            snapshot=snapshot,
            mode=actual_mode,
            context=context,
            error_text=error_text,
        )
        try:
            if log:
                log(f"local code engine: writing {norm}")
            raw_file = await llm.chat_text_async(
                system=PROJECT_ENGINE_FILE_SYSTEM,
                messages=[{"role": "user", "content": base_prompt}],
                temperature=0.18,
                purpose="code-file-write" if phase != "repair" else "code-file-repair",
            )
            content = clean_single_file_output(raw_file, path=norm, prompt=prompt)
        except Exception as e:
            if log:
                log(f"local code engine skipped {norm} after generation failure: {type(e).__name__}: {e}")

        # Validate and give the local model one or more self-repair attempts.
        if content is not None and validate_files:
            ok, issues = validate_generated_file_content(norm, content, prompt=prompt, project_name=project_root.name)
            attempt = 0
            while not ok and attempt < max_file_retries:
                attempt += 1
                quality_issues.append({"path": norm, "attempt": attempt, "issues": issues})
                if log:
                    log(f"local code engine quality gate rejected {norm}: {'; '.join(issues[:3])}; retrying")
                retry_prompt = (
                    base_prompt
                    + "\n\nPREVIOUS_OUTPUT_WAS_REJECTED_BY_QUALITY_GATE:\n"
                    + str(content or "")[:6000]
                    + "\n\nQUALITY_ERRORS_TO_FIX:\n"
                    + "\n".join(f"- {issue}" for issue in issues)
                    + "\n\nReturn a corrected complete version of REQUESTED_FILE only. No markdown wrapper, no explanation."
                )
                try:
                    raw_retry = await llm.chat_text_async(
                        system=PROJECT_ENGINE_FILE_SYSTEM,
                        messages=[{"role": "user", "content": retry_prompt}],
                        temperature=0.12,
                        purpose="code-file-rewrite" if phase != "repair" else "code-file-repair-rewrite",
                    )
                    candidate = clean_single_file_output(raw_retry, path=norm, prompt=prompt)
                except Exception as e:
                    if log:
                        log(f"local code engine retry failed for {norm}: {type(e).__name__}: {e}")
                    candidate = None
                if candidate is None:
                    break
                content = candidate
                ok, issues = validate_generated_file_content(norm, content, prompt=prompt, project_name=project_root.name)

            if not ok:
                quality_issues.append({"path": norm, "attempt": attempt + 1, "issues": issues})
                if log:
                    log(f"local code engine skipped {norm}; quality gate still failed: {'; '.join(issues[:3])}")
                content = None

        if content is None:
            fallback_content = _default_content_for_path(norm, prompt=prompt, project_name=project_root.name)
            if fallback_content is not None and validate_files:
                ok, issues = validate_generated_file_content(norm, fallback_content, prompt=prompt, project_name=project_root.name)
                if not ok:
                    quality_issues.append({"path": norm, "attempt": "default", "issues": issues})
                    fallback_content = None
            content = fallback_content
        if content is None:
            skipped.append(norm)
            continue

        operations.append({"op": "write_file", "path": norm, "content": content})
        generated_files.append((norm, content))

    operations, dropped = sanitize_operations(operations)
    if not operations:
        return None

    if not test_commands:
        test_commands = infer_test_commands(
            [(op["path"], op.get("content", "")) for op in operations],
            user_prompt=prompt,
        )
    if prompt_mentions_interactive_input(prompt):
        test_commands = [cmd for cmd in test_commands if not re.match(r"(?i)^\s*(?:python|python3|py)\s+main\.py\b", cmd.strip())]
        if not test_commands:
            py_paths = [op["path"] for op in operations if str(op.get("path") or "").endswith(".py")]
            if py_paths:
                test_commands = [f"python -m py_compile {py_paths[0]}"]

    summary_action = "Repaired project files" if phase == "repair" else "Created/edited project structure"
    note = (
        "Used the local code engine NLP planner: manifest plus one file-content generation per project file. "
        "This avoids strict JSON output for local code-pipeline planning."
    )
    if skipped:
        note += f" Skipped {len(skipped)} file(s) that were unsafe or not generated: {', '.join(skipped[:8])}."
    if dropped:
        note += f" Dropped {dropped} unsafe repeated-token operation(s)."
    if quality_issues:
        note += " Quality gate rejected/rewrote local-model file output: " + short_quality_issue_summary(quality_issues, max_items=5) + "."
    if error_text:
        note += " Included compile/test error text for repair."

    return {
        "summary": f"{summary_action}: {', '.join(op['path'] for op in operations[:10])}.",
        "operations": operations,
        "test_commands": test_commands,
        "notes": note,
        "metadata": {
            "local_code_engine_plan": True,
            "local_nlp_project_plan": True,
            "planner_kind": "local_code_engine_manifest_files",
            "mode": actual_mode,
            "phase": phase,
            "skip_local_json_plan": True,
            "skip_local_json_review": True,
            "generated_file_count": len(operations),
            "quality_issues_count": len(quality_issues),
            "quality_issues": quality_issues[:20],
            "quality_lesson": build_quality_lesson_payload(
                prompt=prompt,
                project_name=project_root.name,
                quality_issues=quality_issues,
            ) if quality_issues else {},
        },
    }
