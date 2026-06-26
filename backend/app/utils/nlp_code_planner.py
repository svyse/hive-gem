from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.utils.repetition_guard import is_repetitive_text, sanitize_operation_content, sanitize_operations, truncate_repetitive_tail
from app.utils.local_code_quality import validate_generated_file_content, validate_generated_path_for_prompt


NLP_CODE_PLAN_SYSTEM = """You are a local software project generator and editor.

Create or modify the requested project by returning FILE blocks, not JSON.
Use this exact format for every file:

FILE: relative/path.ext
```language
file contents here
```

Rules:
- Return only project files and optional TEST_COMMANDS.
- Paths must be relative paths inside the project.
- Do not include markdown explanation outside file blocks.
- For new projects, include the full runnable project structure: entrypoint, modules, README, config/requirements, and tests when practical.
- For edit/debug requests, return the complete updated contents of changed files, not a diff.
- For Python CLI programs that ask for user input, put reusable logic in functions so tests can import it.
- If you include tests, prefer Python stdlib unittest unless the user explicitly asks for pytest.

Optional test command format:
TEST_COMMANDS:
- python -m py_compile main.py
"""


_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._/@+\- ]+$")
_COMMAND_PATH_HEADS = {"python", "python3", "py", "pip", "pip3", "git", "cd", "npm", "yarn", "pnpm", "pyenv", "curl", "wget", "mkdir", "touch", "rm", "del", "powershell", "cmd"}
_PLACEHOLDER_PATH_RE = re.compile(r"^(?:file|module|script|data)_\d+(?:\.[A-Za-z0-9_+\-]{1,16})?$", re.IGNORECASE)
_FILE_BLOCK_RE = re.compile(
    r"(?ims)^\s*(?:FILE|File|file)\s*:\s*([^\n\r]+?)\s*\n\s*```\s*([A-Za-z0-9_+.#\-]*)?\s*\n([\s\S]*?)\n\s*```"
)
_HEADING_BLOCK_RE = re.compile(
    r"(?ims)^\s*#{1,6}\s+`?([^`\n\r]+?\.[A-Za-z0-9_+\-]{1,12})`?\s*\n\s*```\s*([A-Za-z0-9_+.#\-]*)?\s*\n([\s\S]*?)\n\s*```"
)
_FENCED_BLOCK_RE = re.compile(r"(?ims)```\s*([A-Za-z0-9_+.#\-]*)?\s*\n([\s\S]*?)\n\s*```")
_TEST_COMMANDS_RE = re.compile(r"(?ims)^\s*TEST_COMMANDS\s*:\s*(.*?)(?=^\s*(?:FILE\s*:|#{1,6}\s+|```)|\Z)")


_LANG_EXTENSION = {
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "html": ".html",
    "css": ".css",
    "json": ".json",
    "yaml": ".yaml",
    "yml": ".yml",
    "bash": ".sh",
    "shell": ".sh",
    "sh": ".sh",
    "java": ".java",
    "go": ".go",
    "rust": ".rs",
    "rs": ".rs",
    "cpp": ".cpp",
    "c++": ".cpp",
    "c": ".c",
}


def env_nlp_planner_enabled() -> bool:
    raw = os.getenv("LOCAL_CODE_NLP_PLANNER_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def is_local_backend(llm: Any) -> bool:
    backend = str(getattr(llm, "backend", "") or "").strip().lower()
    return backend == "local" or llm.__class__.__name__.lower().startswith("local")


def compact_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\\", "/").strip().lower())


def prompt_mentions_python(prompt: str) -> bool:
    q = compact_prompt(prompt)
    return any(term in q for term in ("python", ".py", " py ", "python script", "python project", "python code"))


def prompt_mentions_interactive_input(prompt: str) -> bool:
    q = compact_prompt(prompt)
    return any(
        term in q
        for term in (
            "input",
            "user input",
            "takes input",
            "take input",
            "selected operation",
            "choose operation",
            "select operation",
            "console",
            "terminal",
            "cli",
            "command line",
            "prompt the user",
        )
    )


def prompt_mentions_calculator(prompt: str) -> bool:
    q = compact_prompt(prompt)
    return "calculator" in q or "basic calc" in q or "arithmetic" in q


def _normalize_path(raw_path: str) -> Optional[str]:
    p = str(raw_path or "").strip().strip("`'\"")
    p = p.replace("\\", "/")
    p = re.sub(r"\s+", " ", p).strip()
    # Remove common prefixes local models add.
    p = re.sub(r"^(?:path|file(?:name)?)\s*=\s*", "", p, flags=re.IGNORECASE).strip()
    while p.startswith("./"):
        p = p[2:]
    if not p:
        return None
    if ":" in p.split("/", 1)[0]:
        return None
    if p.startswith("/") or p.startswith("~"):
        return None
    if not _SAFE_NAME_RE.match(p):
        return None
    parts = [part.strip() for part in p.split("/") if part.strip()]
    if not parts:
        return None
    blocked = {"..", ".memory", ".workspace", "__pycache__"}
    if any(part in blocked for part in parts):
        return None
    if any(part.startswith(".") and part not in {".gitignore", ".dockerignore", ".env.example"} for part in parts):
        return None
    normalized = str(PurePosixPath(*parts))
    if normalized in {".", ""}:
        return None
    if any(" " in part for part in parts):
        return None
    first_token = re.split(r"\s+", parts[0].lower(), maxsplit=1)[0]
    if first_token in _COMMAND_PATH_HEADS:
        return None
    name = parts[-1]
    if _PLACEHOLDER_PATH_RE.match(name):
        return None
    suffix = Path(name).suffix.lower()
    if suffix in {".pyc", ".pyo", ".pyd", ".dll", ".exe", ".zip", ".png", ".jpg", ".jpeg", ".pdf"}:
        return None
    if not suffix and name not in {"Dockerfile", "Makefile", "Procfile", ".gitignore", ".dockerignore", ".env.example", "README.md", "requirements.txt", "pyproject.toml", "package.json"}:
        return None
    return normalized


def _path_from_code_comment(content: str) -> Optional[str]:
    lines = str(content or "").splitlines()[:5]
    patterns = [
        r"^\s*#\s*(?:file|filename|path)?\s*:?\s*([A-Za-z0-9_./@+\- ]+\.[A-Za-z0-9_+\-]{1,12})\s*$",
        r"^\s*//\s*(?:file|filename|path)?\s*:?\s*([A-Za-z0-9_./@+\- ]+\.[A-Za-z0-9_+\-]{1,12})\s*$",
        r"^\s*<!--\s*(?:file|filename|path)?\s*:?\s*([A-Za-z0-9_./@+\- ]+\.[A-Za-z0-9_+\-]{1,12})\s*-->\s*$",
    ]
    for line in lines:
        for pat in patterns:
            m = re.match(pat, line, flags=re.IGNORECASE)
            if m:
                return _normalize_path(m.group(1))
    return None


def _infer_path_from_language(lang: str, content: str, prompt: str, index: int) -> str:
    language = str(lang or "").strip().lower()
    q = compact_prompt(prompt)
    comment_path = _path_from_code_comment(content)
    if comment_path:
        return comment_path

    if "html" in language or " html" in q or "web page" in q:
        return "index.html" if index == 0 else "app.html"
    if language in {"css"}:
        return "style.css"
    if language in {"javascript", "js", "node"}:
        if "test" in content.lower() and index > 0:
            return "test.js"
        return "app.js" if index == 0 else "helpers.js"
    if language in {"typescript", "ts"}:
        return "app.ts" if index == 0 else "helpers.ts"
    if language in {"bash", "shell", "sh"}:
        return "run.sh" if index == 0 else "scripts/helper.sh"
    if language == "json":
        if "package" in content[:100].lower():
            return "package.json"
        return "config.json"

    # Default to Python because the reported local code-pipeline issue is with
    # Python sample projects and most local coding prompts in this app are Python.
    if language in {"python", "py"} or prompt_mentions_python(prompt) or not language:
        if "unittest" in content.lower() or "pytest" in content.lower() or re.search(r"def\s+test_", content):
            return "tests/test_basic.py"
        return "main.py" if index == 0 else "app_logic.py"

    ext = _LANG_EXTENSION.get(language, ".txt")
    return f"generated{ext}"


def _strip_path_comment(content: str, path: str) -> str:
    lines = str(content or "").splitlines()
    if not lines:
        return content
    first = lines[0].strip()
    normalized_path = path.replace("\\", "/").lower()
    if normalized_path in first.lower() and re.match(r"^(#|//|<!--)", first):
        return "\n".join(lines[1:]).lstrip("\n")
    return content


def _extract_test_commands(text: str) -> List[str]:
    commands: List[str] = []
    for m in _TEST_COMMANDS_RE.finditer(str(text or "")):
        section = m.group(1)
        for line in section.splitlines():
            s = line.strip()
            if not s:
                continue
            s = re.sub(r"^[-*]\s*", "", s).strip()
            s = s.strip("` ")
            if not s or s.lower().startswith(("file:", "notes:")):
                continue
            if _is_safe_test_command(s):
                commands.append(s)
    return list(dict.fromkeys(commands))[:5]


def _is_safe_test_command(command: str) -> bool:
    s = str(command or "").strip()
    if not s or len(s) > 220:
        return False
    lowered = s.lower()
    blocked = (" rm ", " del ", " rmdir ", " format ", "shutdown", "powershell", "curl ", "wget ", "Invoke-WebRequest".lower())
    if any(token in f" {lowered} " for token in blocked):
        return False
    return lowered.startswith((
        "python ",
        "python3 ",
        "py ",
        "pytest",
        "npm test",
        "node ",
        "go test",
        "cargo test",
    ))


def _add_file(files: List[Tuple[str, str]], seen: set[str], path: str, content: str, *, user_prompt: str = "") -> None:
    normalized = _normalize_path(path)
    if not normalized or normalized in seen:
        return
    ok_path, _path_issues = validate_generated_path_for_prompt(normalized, prompt=user_prompt)
    if not ok_path:
        return
    body = str(content or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n") + "\n"
    body = _strip_path_comment(body, normalized)
    try:
        body = sanitize_operation_content(body, max_chars=120_000)
    except ValueError:
        return
    if not body.strip() or is_repetitive_text(body):
        return
    ok, _issues = validate_generated_file_content(normalized, body, prompt=user_prompt)
    if not ok:
        return
    files.append((normalized, body))
    seen.add(normalized)


def extract_files_from_nlp_output(text: str, *, user_prompt: str = "") -> List[Tuple[str, str]]:
    raw = truncate_repetitive_tail(str(text or ""))
    if not raw.strip() or is_repetitive_text(raw):
        return []

    files: List[Tuple[str, str]] = []
    seen: set[str] = set()

    for match in _FILE_BLOCK_RE.finditer(raw):
        _add_file(files, seen, match.group(1), match.group(3), user_prompt=user_prompt)

    for match in _HEADING_BLOCK_RE.finditer(raw):
        _add_file(files, seen, match.group(1), match.group(3), user_prompt=user_prompt)

    # Code fences without FILE headers are common Q&A-style local answers.
    # Infer reasonable filenames from the language, path comments, and prompt.
    for idx, match in enumerate(_FENCED_BLOCK_RE.finditer(raw)):
        content = match.group(2)
        # Skip fences already consumed as FILE/heading blocks by detecting exact body.
        if any(existing_content.strip() == str(content or "").strip() for _, existing_content in files):
            continue
        path = _infer_path_from_language(match.group(1) or "", content, user_prompt, idx)
        _add_file(files, seen, path, content, user_prompt=user_prompt)

    # Last-resort: local model may emit bare Python code without fences.
    if not files:
        stripped = raw.strip()
        if _looks_like_bare_code(stripped, user_prompt):
            path = _infer_path_from_language("python" if prompt_mentions_python(user_prompt) else "", stripped, user_prompt, 0)
            _add_file(files, seen, path, stripped, user_prompt=user_prompt)

    return files[:30]


def _looks_like_bare_code(text: str, prompt: str) -> bool:
    s = str(text or "").strip()
    if not s or len(s) > 120_000:
        return False
    code_markers = (
        "def ",
        "class ",
        "if __name__",
        "print(",
        "import ",
        "from ",
        "function ",
        "const ",
        "let ",
        "var ",
    )
    if any(marker in s for marker in code_markers):
        return True
    return prompt_mentions_python(prompt) and "=" in s and "\n" in s


def infer_test_commands(files: Iterable[Tuple[str, str]], *, user_prompt: str = "", explicit_commands: Optional[List[str]] = None) -> List[str]:
    file_items = list(files or [])
    commands = [cmd for cmd in (explicit_commands or []) if _is_safe_test_command(cmd)]
    if prompt_mentions_interactive_input(user_prompt):
        # Do not let generated test commands hang waiting for stdin.
        commands = [
            cmd
            for cmd in commands
            if not re.match(r"(?i)^\s*(?:python|python3|py)\s+main\.py\b", cmd.strip())
        ]
    paths = [path for path, _ in file_items]
    if commands:
        return list(dict.fromkeys(commands))[:5]

    test_py_files = [(path, content) for path, content in file_items if path.startswith("tests/") and path.endswith(".py")]
    if test_py_files:
        test_blob = "\n".join(str(content or "") for _path, content in test_py_files).lower()
        if "import pytest" in test_blob or ("def test_" in test_blob and "unittest" not in test_blob):
            return ["pytest -q"]
        return ["python -m unittest discover -s tests -p test_*.py"]

    py_paths = [p for p in paths if p.endswith(".py")]
    if py_paths:
        # Avoid hanging on interactive scripts. Compile is enough here because the
        # separate compile agent already checks all Python files too; this gives
        # the testing phase a deterministic, non-interactive command.
        main = "main.py" if "main.py" in py_paths else py_paths[0]
        return [f"python -m py_compile {main}"]

    if "package.json" in paths:
        return ["npm test"]
    if "index.html" in paths:
        # Static browser projects often need a browser, not `node app.js`.
        # The testing agent will mark the no-command case as a safe validation skip.
        return []
    if any(p.endswith(".js") for p in paths):
        first = next(p for p in paths if p.endswith(".js"))
        return [f"node {first}"]
    return []


def build_plan_from_nlp_output(
    raw_text: str,
    *,
    user_prompt: str,
    reason: str | None = None,
) -> Optional[Dict[str, Any]]:
    files = extract_files_from_nlp_output(raw_text, user_prompt=user_prompt)
    if not files:
        return None

    explicit_commands = _extract_test_commands(raw_text)
    operations = [{"op": "write_file", "path": path, "content": content} for path, content in files]
    operations, dropped = sanitize_operations(operations)
    if not operations:
        return None

    file_list = ", ".join(op["path"] for op in operations[:8])
    note = "Parsed file operations from local NLP/code-block output instead of requiring strict JSON."
    if dropped:
        note += f" Dropped {dropped} unsafe or repeated generated file(s)."
    if reason:
        note += f" Reason: {reason}"

    plan = {
        "summary": f"Created project files from the local NLP prompt: {file_list}.",
        "operations": operations,
        "test_commands": infer_test_commands([(op["path"], op.get("content", "")) for op in operations], user_prompt=user_prompt, explicit_commands=explicit_commands),
        "notes": note,
        "metadata": {
            "local_nlp_project_plan": True,
            "planner_kind": "nlp_file_blocks",
            "skip_local_json_plan": True,
            "skip_local_json_review": False,
        },
    }
    return plan


def build_generic_python_scaffold_plan(user_prompt: str, *, reason: str | None = None) -> Dict[str, Any]:
    """Final local fallback when both JSON and NLP parsing fail.

    This intentionally writes a runnable scaffold plus a README carrying the full
    prompt, so the code pipeline still creates a project instead of applying zero
    operations.  Specific deterministic templates should be preferred whenever
    possible.
    """

    prompt = str(user_prompt or "").strip() or "Create a Python project."
    safe_prompt = prompt.replace('"""', "'''")
    if prompt_mentions_interactive_input(prompt):
        main = (
            '"""Safe interactive scaffold generated from the local NLP prompt.\n\n'
            f"Prompt:\n{safe_prompt}\n"
            '"""\n\n'
            "def main() -> None:\n"
            "    print('The local model did not produce safe project files for the requested task.')\n"
            "    print('This scaffold is intentionally minimal so invalid generated files are not written.')\n"
            "    user_value = input('Enter input: ')\n"
            "    print(user_value)\n\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
    else:
        main = (
            '"""Project scaffold generated from the local NLP prompt.\n\n'
            f"Prompt:\n{safe_prompt}\n"
            '"""\n\n'
            "def main() -> None:\n"
            "    print('Project scaffold created successfully.')\n"
            "    print('Review README.md for the original prompt and extend main.py as needed.')\n\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
    readme = (
        "# Local Code Pipeline Project\n\n"
        "This scaffold was created because the local model did not return parseable file blocks or JSON.\n\n"
        "## Original prompt\n\n"
        f"{prompt}\n\n"
        "## Run\n\n"
        "```bash\npython main.py\n```\n"
    )
    notes = "Created a safe Python scaffold after local NLP/JSON planning failed."
    if reason:
        notes += f" Reason: {reason}"
    return {
        "summary": "Created a safe Python project scaffold from the local NLP prompt.",
        "operations": [
            {"op": "write_file", "path": "main.py", "content": main},
            {"op": "write_file", "path": "README.md", "content": readme},
        ],
        "test_commands": ["python -m py_compile main.py"],
        "notes": notes,
        "metadata": {
            "local_nlp_project_plan": True,
            "planner_kind": "generic_python_scaffold",
            "skip_local_json_plan": True,
            "skip_local_json_review": True,
        },
    }
