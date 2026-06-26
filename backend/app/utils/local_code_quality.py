from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_CODE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
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
}

_DOC_SUFFIXES = {".md", ".txt", ".rst"}
_SPECIAL_TEXT_NAMES = {
    "Dockerfile",
    "Makefile",
    "Procfile",
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "tsconfig.json",
    ".gitignore",
    ".dockerignore",
    ".env.example",
}
_BINARY_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".dll",
    ".exe",
    ".so",
    ".dylib",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".sqlite",
    ".sqlite3",
    ".db",
}
_SKIP_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".memory",
    ".workspace",
    ".workspaces",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "coverage",
}
_COMMAND_PATH_HEADS = {
    "python",
    "python3",
    "py",
    "pip",
    "pip3",
    "git",
    "cd",
    "npm",
    "yarn",
    "pnpm",
    "npx",
    "pyenv",
    "curl",
    "wget",
    "mkdir",
    "touch",
    "copy",
    "xcopy",
    "move",
    "del",
    "rm",
    "rmdir",
    "powershell",
    "cmd",
}

_SHELL_COMMAND_RE = re.compile(
    r"^\s*(?:"
    r"python(?:3)?\s+--?version\b|"
    r"python(?:3)?\s+-m\s+pip\b|"
    r"py\s+-m\s+pip\b|"
    r"pip(?:3)?\s+install\b|"
    r"git\s+clone\b|"
    r"cd\s+[^=].*|"
    r"pyenv\s+\w+\b|"
    r"npm\s+(?:install|run|start|test)\b|"
    r"yarn\s+(?:add|install|start|test)\b|"
    r"pnpm\s+(?:add|install|start|test)\b|"
    r"mkdir\s+\S+|"
    r"touch\s+\S+|"
    r"curl\s+\S+|"
    r"wget\s+\S+"
    r")",
    re.IGNORECASE,
)

_MARKDOWN_HINT_RE = re.compile(
    r"^\s*(?:#{1,6}\s+|[-*]\s+\*\*|\d+\.\s+\*\*|To use\b|First,\b|Next,\b|Then,\b|Clone the\b|Run the\b|Expected output\b)",
    re.IGNORECASE,
)

_PY_TEST_NAME_RE = re.compile(r"(?:^|/)(?:test_[^/]+|[^/]+_test)\.py$", re.IGNORECASE)
_PLACEHOLDER_NAME_RE = re.compile(r"^(?:file|module|script|data)_\d+(?:\.[A-Za-z0-9_+\-]{1,16})?$", re.IGNORECASE)
_SOURCE_START_RE = re.compile(
    r"^\s*(?:import\s+|from\s+\S+\s+import\s+|def\s+|class\s+|async\s+def\s+|if\s+__name__\s*==|try\s*:|except\b|return\b|print\s*\(|const\s+|let\s+|var\s+|function\s+|package\s+|public\s+class\s+|#include\b)",
    re.IGNORECASE,
)


def compact_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\\", "/").strip().lower())


def infer_requested_language(prompt: str = "", project_name: str = "") -> str:
    """Best-effort language inference used only for safety filtering.

    This is intentionally generic: it reads the prompt and project name.  A
    project named ``*_py`` should not receive Java/C++ files unless the prompt
    explicitly asks for them.
    """

    q = compact_prompt(f"{prompt} {project_name}")
    project = compact_prompt(project_name)
    if any(k in q for k in ("react", "vite", "tsx", "typescript")) or project.endswith(("_ts", "-ts")):
        return "typescript"
    if any(k in q for k in ("javascript", "node", "express", "npm")) or project.endswith(("_js", "-js")):
        return "javascript"
    if any(k in q for k in ("html", "css", "website", "web page", "browser")):
        return "web"
    if any(k in q for k in ("java", "spring")) or project.endswith(("_java", "-java")):
        return "java"
    if any(k in q for k in ("c++", "cpp", " c plus plus")) or project.endswith(("_cpp", "-cpp")):
        return "cpp"
    if any(k in q for k in ("golang", "go ", " go")) or project.endswith(("_go", "-go")):
        return "go"
    if any(k in q for k in ("rust", "cargo")) or project.endswith(("_rs", "-rs", "_rust", "-rust")):
        return "rust"
    if any(k in q for k in ("python", "pytest", "unittest", ".py", " py ")) or re.search(r"(?:^|[_\-/])py(?:[_\-/]|$|v\d+)", project):
        return "python"
    # Most current local-code usage in this app is Python when no language is
    # specified.  This keeps unknown prompts from receiving random mixed-language
    # files such as file_1.java and main.cpp.
    return "python"


def is_probably_entrypoint(path: str) -> bool:
    p = str(path or "").replace("\\", "/").lower().strip("/")
    name = Path(p).name.lower()
    return p in {
        "main.py",
        "app.py",
        "run.py",
        "app/main.py",
        "src/main.py",
        "index.js",
        "src/index.js",
        "index.ts",
        "src/index.ts",
        "index.html",
    } or name in {"main.py", "app.py", "index.js", "index.ts", "index.html"}


def prompt_requests_terminal_input(prompt: str) -> bool:
    q = compact_prompt(prompt)
    if not q:
        return False
    # Do not force stdin for web/API/GUI projects, where input is usually a form,
    # request body, or UI event rather than terminal input().
    if any(k in q for k in ("web", "website", "frontend", "html", "browser", "api", "rest", "server", "gui", "tkinter", "streamlit")):
        return False
    return any(
        k in q
        for k in (
            "input",
            "user input",
            "takes input",
            "take input",
            "ask the user",
            "asks the user",
            "query the user",
            "queries the user",
            "prompt the user",
            "enter ",
            "entered by the user",
            "from the user",
            "selected by the user",
            "choose operation",
            "select operation",
            "console",
            "terminal",
            "cli",
            "command line",
        )
    )


def _looks_like_test_file(path: str) -> bool:
    p = str(path or "").replace("\\", "/").lower()
    return p.startswith("tests/") or bool(_PY_TEST_NAME_RE.search(p))


def _non_comment_lines(content: str, *, comment_prefixes: Tuple[str, ...]) -> List[str]:
    out: List[str] = []
    for line in str(content or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if any(s.startswith(prefix) for prefix in comment_prefixes):
            continue
        out.append(s)
    return out


def _has_shell_command_line(content: str, *, comment_prefixes: Tuple[str, ...]) -> bool:
    for line in _non_comment_lines(content, comment_prefixes=comment_prefixes):
        if _SHELL_COMMAND_RE.search(line):
            return True
    return False


def _has_markdown_prose_line(content: str, *, comment_prefixes: Tuple[str, ...]) -> bool:
    for line in _non_comment_lines(content, comment_prefixes=comment_prefixes)[:40]:
        if "```" in line:
            return True
        if _MARKDOWN_HINT_RE.search(line):
            return True
    return False


def _python_has_terminal_input(content: str) -> bool:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return "input(" in content

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "input":
                return True
            if isinstance(func, ast.Attribute) and func.attr in {"prompt", "confirm"}:
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"argparse", "click", "typer"}:
                    return True
        if isinstance(node, ast.ImportFrom) and node.module in {"argparse", "click", "typer"}:
            return True
        if isinstance(node, ast.Attribute) and node.attr == "argv":
            value = node.value
            if isinstance(value, ast.Name) and value.id == "sys":
                return True
    return False


def _python_has_shell_expression(content: str) -> bool:
    """Catch shell snippets that still parse as Python, e.g. `python --version`."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False

    for node in tree.body:
        if isinstance(node, ast.Expr):
            value = node.value
            # Module docstring is fine.
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                continue
            # A real script can call print/main at top level. A bare arithmetic
            # expression made only from unknown names is usually a copied shell command.
            if isinstance(value, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)):
                names = [n.id for n in ast.walk(value) if isinstance(n, ast.Name)]
                calls = [n for n in ast.walk(value) if isinstance(n, ast.Call)]
                constants = [n for n in ast.walk(value) if isinstance(n, ast.Constant)]
                if names and not calls and not constants:
                    return True
    return False


def _python_local_function_arity_issues(content: str) -> List[str]:
    """Tiny static check for obvious local function call arity errors.

    This catches failures such as ``add(num1 * num2)`` when ``add`` is defined
    with two required positional arguments.  It is not a full linter; it only
    blocks clearly broken generated files before they are written.
    """

    issues: List[str] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return issues

    funcs: Dict[str, Tuple[int, Optional[int]]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            if args.vararg or args.kwarg:
                continue
            positional = list(args.posonlyargs) + list(args.args)
            # Top-level functions do not have self unless the model incorrectly
            # wrote it there, so count what Python actually requires.
            required = max(0, len(positional) - len(args.defaults))
            max_args = len(positional)
            funcs[node.name] = (required, max_args)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in funcs:
            continue
        min_args, max_args = funcs[name]
        positional_count = len(node.args)
        if positional_count < min_args or positional_count > max_args:
            issues.append(f"function `{name}` is called with {positional_count} positional argument(s), but expects {min_args}-{max_args}")
    return issues[:6]


def _validate_python(path: str, content: str, prompt: str) -> List[str]:
    issues: List[str] = []
    text = str(content or "")
    try:
        ast.parse(text)
    except SyntaxError as exc:
        issues.append(f"not valid Python syntax: {exc.msg} at line {exc.lineno}")

    if ">>>" in text:
        issues.append("contains interactive Python REPL prompt text (`>>>`) instead of source code")
    if _has_shell_command_line(text, comment_prefixes=("#",)):
        issues.append("contains shell/installation commands in a .py file; Python files must contain Python source code")
    if _python_has_shell_expression(text):
        issues.append("looks like a shell command that accidentally parsed as Python")
    if "```" in text:
        issues.append("contains markdown code fences inside a Python file")
    issues.extend(_python_local_function_arity_issues(text))

    if _looks_like_test_file(path):
        lower = text.lower()
        has_test_signal = (
            "unittest" in lower
            or "pytest" in lower
            or re.search(r"(?m)^\s*def\s+test_", text) is not None
            or re.search(r"(?m)^\s*class\s+Test", text) is not None
        )
        if not has_test_signal:
            issues.append("test file does not contain recognizable unittest/pytest tests")
    elif is_probably_entrypoint(path) and prompt_requests_terminal_input(prompt) and not _python_has_terminal_input(text):
        issues.append("prompt asks for user/terminal input, but the Python entrypoint does not read input")

    return issues


def _validate_javascript(path: str, content: str, prompt: str) -> List[str]:
    issues: List[str] = []
    text = str(content or "")
    if "```" in text:
        issues.append("contains markdown code fences inside a JavaScript/TypeScript file")
    if _has_shell_command_line(text, comment_prefixes=("//", "/*", "*")):
        issues.append("contains shell/installation commands in a JavaScript/TypeScript file")
    if _has_markdown_prose_line(text, comment_prefixes=("//", "/*", "*")):
        issues.append("looks like README/prose instead of JavaScript/TypeScript source")
    if is_probably_entrypoint(path) and prompt_requests_terminal_input(prompt):
        lower = text.lower()
        if not any(k in lower for k in ("readline", "process.argv", "prompt(", "inquirer")):
            issues.append("prompt asks for user/terminal input, but the JS/TS entrypoint does not read input")
    return issues


def _validate_requirements(content: str) -> List[str]:
    issues: List[str] = []
    for idx, line in enumerate(str(content or "").splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lower = s.lower()
        if lower.startswith(("pip ", "python ", "python3 ", "py ", "git clone", "cd ", "import ", "from ", "def ", "class ")):
            issues.append(f"requirements.txt line {idx} contains code or a command; use only package specifiers")
            continue
        # Allow normal package specs, editable/local paths, and git+ URLs. Reject
        # whitespace-heavy instructional prose.
        if " " in s and not s.startswith(("-e ", "--index-url ", "--extra-index-url ")):
            issues.append(f"requirements.txt line {idx} looks like prose/command text, not a package specifier")
    return issues


def _validate_ignore_file(content: str) -> List[str]:
    issues: List[str] = []
    for idx, line in enumerate(str(content or "").splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lower = s.lower()
        if lower.startswith(("pip ", "python ", "python3 ", "git clone", "cd ", "npm ", "yarn ", "pnpm ")):
            issues.append(f"ignore file line {idx} contains a command; ignore files must contain file patterns")
            continue
        if _SOURCE_START_RE.search(s) or re.search(r"\b(?:argparse|logging|traceback|sys\.argv|input\(|print\()", s):
            issues.append(f"ignore file line {idx} looks like source code; ignore files must contain file patterns")
            continue
        if any(ch in s for ch in (";", "{", "}")) or ("=" in s and not s.startswith("*.")):
            issues.append(f"ignore file line {idx} looks like code/config instead of an ignore pattern")
            continue
        if " " in s and "\\ " not in s:
            issues.append(f"ignore file line {idx} contains unescaped spaces and looks unlike a normal ignore pattern")
    return issues


def _validate_json(content: str) -> List[str]:
    try:
        json.loads(str(content or ""))
        return []
    except Exception as exc:
        return [f"not valid JSON: {exc}"]


def _source_line_ratio(text: str) -> float:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return 0.0
    code_like = 0
    for line in lines:
        if _SOURCE_START_RE.search(line) or line.endswith(":") or line.endswith(";") or re.search(r"[{}]", line):
            code_like += 1
    return code_like / max(1, len(lines))


def _validate_markdown(path: str, content: str) -> List[str]:
    issues: List[str] = []
    text = str(content or "")
    stripped = text.lstrip()
    if Path(path).name.lower() == "readme.md":
        if _source_line_ratio(text) > 0.55 and "```" not in text:
            issues.append("README.md looks like raw source code instead of project documentation")
        if stripped.startswith(("import ", "from ", "def ", "class ", "package ", "public class", "#include")) and not stripped.startswith("# "):
            issues.append("README.md starts like a source file; README should be documentation")
    return issues


def _validate_txt(path: str, content: str) -> List[str]:
    issues: List[str] = []
    name = Path(path).name.lower()
    text = str(content or "")
    if _PLACEHOLDER_NAME_RE.match(name) and _source_line_ratio(text) > 0.25:
        issues.append("placeholder .txt file contains source-code-like content; use a real source filename or documentation name")
    return issues


def _validate_shell(path: str, content: str) -> List[str]:
    issues: List[str] = []
    text = str(content or "")
    stripped = text.strip()
    if not stripped:
        issues.append("empty shell script")
        return issues
    if re.match(r"^(?:git@|https?://|ssh://)", stripped):
        issues.append("shell script is only a repository URL, not executable shell commands")
    if stripped.lower() in {"new file", "todo", "placeholder"}:
        issues.append("shell script is placeholder text, not a script")
    if "```" in text:
        issues.append("contains markdown code fences inside a shell script")
    return issues


def _validate_java(content: str) -> List[str]:
    text = str(content or "")
    issues: List[str] = []
    if re.search(r"(?m)^\s*#\s*", text) or "def " in text or "input(" in text or "print(" in text:
        issues.append("Java file appears to contain Python code")
    if "class " not in text and "interface " not in text and "enum " not in text:
        issues.append("Java file does not contain a class/interface/enum declaration")
    return issues


def _validate_cpp(path: str, content: str) -> List[str]:
    text = str(content or "")
    issues: List[str] = []
    if "package " in text and "public class" in text:
        issues.append("C/C++ file appears to contain Java code")
    if Path(path).suffix.lower() in {".c", ".cpp"} and not any(k in text for k in ("#include", "int main", "void main", "std::", "namespace ", "class ", "struct ")):
        issues.append("C/C++ file does not contain recognizable C/C++ source")
    return issues


def _allowed_suffixes_for_language(language: str) -> set[str]:
    common = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
    if language == "python":
        return common | {".py"}
    if language == "javascript":
        return common | {".js", ".jsx", ".html", ".css"}
    if language == "typescript":
        return common | {".ts", ".tsx", ".js", ".jsx", ".html", ".css"}
    if language == "web":
        return common | {".html", ".css", ".js", ".jsx", ".ts", ".tsx"}
    if language == "java":
        return common | {".java"}
    if language == "cpp":
        return common | {".c", ".cpp", ".h", ".hpp"}
    if language == "go":
        return common | {".go"}
    if language == "rust":
        return common | {".rs", ".toml"}
    return common | _CODE_SUFFIXES


def validate_generated_path_for_prompt(
    path: str,
    *,
    prompt: str = "",
    project_name: str = "",
    allow_placeholder_names: bool = False,
) -> Tuple[bool, List[str]]:
    """Validate generated relative paths before file writes.

    This catches the failure mode from local models where commands or code-fence
    language guesses become filenames, e.g. ``python main.py``, ``pip install -r``,
    ``file_1.java`` or ``data_1.json`` in a Python project.
    """

    rel = str(path or "").replace("\\", "/").strip().strip("`'\"")
    issues: List[str] = []
    if not rel:
        return False, ["missing file path"]
    if rel.startswith(("/", "~")) or ":" in rel.split("/", 1)[0]:
        issues.append("path must be relative to the project root")
    parts = [part for part in rel.split("/") if part]
    if not parts:
        return False, ["missing file path"]
    if any(part in {".", ".."} for part in parts):
        issues.append("path must not contain . or .. segments")
    if any(part in _SKIP_PARTS for part in parts):
        issues.append("path targets generated/cache/private directory that should not be written by the code engine")
    if any(part.startswith(".") and part not in {".gitignore", ".dockerignore", ".env.example"} for part in parts):
        issues.append("hidden paths are not allowed except .gitignore, .dockerignore, and .env.example")

    # Spaces make command hallucinations look like paths. Real project files
    # almost never need spaces, so block them for generated code.
    if any(" " in part for part in parts):
        issues.append("path contains spaces; this often means a shell command was mistaken for a filename")

    first_token = re.split(r"\s+", parts[0].lower(), maxsplit=1)[0]
    if first_token in _COMMAND_PATH_HEADS:
        issues.append("path starts with a shell command, not a project file path")

    name = parts[-1]
    suffix = Path(name).suffix.lower()
    if suffix in _BINARY_SUFFIXES:
        issues.append("binary/cache artifacts must not be generated or written")
    if not suffix and name not in _SPECIAL_TEXT_NAMES and name not in {"Dockerfile", "Makefile", "Procfile"}:
        issues.append("path has no file extension and is not a recognized special project file")

    if _PLACEHOLDER_NAME_RE.match(name) and not allow_placeholder_names:
        issues.append("placeholder filename produced by code-block inference is not allowed; use meaningful project filenames")

    language = infer_requested_language(prompt, project_name)
    allowed = _allowed_suffixes_for_language(language)
    special = name in _SPECIAL_TEXT_NAMES or name in {"Dockerfile", "Makefile", "Procfile"}
    # Shell scripts are allowed only when the prompt asks for scripts or shell.
    prompt_q = compact_prompt(prompt)
    shell_requested = any(k in prompt_q for k in ("shell", "bash", ".sh", "script file", "powershell", ".ps1", "batch file"))
    if suffix in {".sh", ".bat", ".ps1"} and not shell_requested:
        issues.append("shell/batch script path was generated even though the prompt did not ask for shell scripts")
    elif suffix and suffix not in allowed and not special:
        issues.append(f"{suffix} file does not match inferred requested language `{language}`")

    return len(issues) == 0, issues


def validate_generated_file_content(
    path: str,
    content: str,
    *,
    prompt: str = "",
    project_name: str = "",
) -> Tuple[bool, List[str]]:
    """Validate local-model file output before it is written to disk.

    The validator is intentionally generic. It does not know how to build a
    calculator or a hello-world app. It checks that the path is sane, the content
    matches the requested file type and prompt class, and common hallucinations
    such as setup transcripts or mixed-language source are rejected.
    """

    rel = str(path or "").replace("\\", "/").strip()
    name = Path(rel).name
    suffix = Path(name).suffix.lower()
    text = str(content or "")
    issues: List[str] = []

    ok_path, path_issues = validate_generated_path_for_prompt(rel, prompt=prompt, project_name=project_name)
    issues.extend(path_issues)

    if not rel:
        return False, ["missing file path"]
    if not text.strip() and name not in {"requirements.txt", ".env.example"}:
        return False, issues + ["empty file content"]
    if "\x00" in text:
        issues.append("contains NUL/binary data")

    if name == "requirements.txt":
        issues.extend(_validate_requirements(text))
    elif name in {".gitignore", ".dockerignore"}:
        issues.extend(_validate_ignore_file(text))
    elif name == "Dockerfile":
        if _has_markdown_prose_line(text, comment_prefixes=("#",)):
            issues.append("Dockerfile looks like prose/markdown instead of Docker instructions")
    elif suffix == ".py":
        issues.extend(_validate_python(rel, text, prompt))
    elif suffix in {".js", ".jsx", ".ts", ".tsx"}:
        issues.extend(_validate_javascript(rel, text, prompt))
    elif suffix == ".json":
        issues.extend(_validate_json(text))
    elif suffix == ".md":
        issues.extend(_validate_markdown(rel, text))
    elif suffix == ".txt":
        issues.extend(_validate_txt(rel, text))
    elif suffix in {".sh", ".bat", ".ps1"}:
        issues.extend(_validate_shell(rel, text))
    elif suffix == ".java":
        if "```" in text:
            issues.append("contains markdown code fences inside a Java file")
        issues.extend(_validate_java(text))
    elif suffix in {".c", ".cpp", ".h", ".hpp"}:
        if "```" in text:
            issues.append("contains markdown code fences inside a C/C++ file")
        issues.extend(_validate_cpp(rel, text))
    elif suffix in {".html", ".css", ".toml", ".yaml", ".yml", ".go", ".rs", ".cs", ".php", ".rb", ".sql"}:
        if "```" in text:
            issues.append("contains markdown code fences inside a source/config file")
        if suffix not in {".html", ".css", ".sql"} and _has_markdown_prose_line(text, comment_prefixes=("#", "//", "/*", "*", "--")):
            issues.append("looks like README/prose instead of source/config content")
    elif suffix in _DOC_SUFFIXES:
        pass
    elif suffix in _CODE_SUFFIXES:
        if _has_markdown_prose_line(text, comment_prefixes=("#", "//", "/*", "*")):
            issues.append("looks like documentation instead of source code")

    return len(issues) == 0, issues


def filter_operations_for_quality(
    operations: Iterable[Dict[str, Any]],
    *,
    prompt: str = "",
    project_name: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    for op in operations or []:
        if not isinstance(op, dict):
            continue
        path = str(op.get("path") or "")
        kind = op.get("op")
        if kind in {"mkdir", "delete_file"}:
            ok_path, path_issues = validate_generated_path_for_prompt(path, prompt=prompt, project_name=project_name)
            if ok_path:
                kept.append(op)
            else:
                issues.append({"path": path, "issues": path_issues})
            continue
        if kind != "write_file":
            kept.append(op)
            continue
        content = op.get("content")
        if not isinstance(content, str):
            issues.append({"path": path, "issues": ["write_file content is not text"]})
            continue
        ok, file_issues = validate_generated_file_content(path, content, prompt=prompt, project_name=project_name)
        if ok:
            kept.append(op)
        else:
            issues.append({"path": path, "issues": file_issues})
    return kept, issues


def file_generation_rules_for_path(path: str, *, prompt: str = "") -> str:
    rel = str(path or "").replace("\\", "/")
    name = Path(rel).name
    suffix = Path(name).suffix.lower()
    rules: List[str] = [
        f"You are writing `{rel}` only.",
        "Return raw file content only: no markdown wrapper, no explanations, no install/run transcript.",
        "Do not invent placeholder filenames such as file_1.py, module_2.py, data_1.json, or command-like names such as python main.py.",
    ]
    if name == "requirements.txt":
        rules.append("requirements.txt may be empty; otherwise use one Python package specifier per line. Do not write pip/git/cd commands or Python imports.")
    elif name in {".gitignore", ".dockerignore"}:
        rules.append("Ignore files must contain ignore patterns only. Do not write source code, shell commands, or setup instructions.")
    elif name == "README.md":
        rules.append("README.md should explain what the generated project does and how to run/test it locally. It must not be a raw source-code file.")
    elif suffix == ".py":
        rules.append("The file must be valid Python source code. Do not include shell commands like python --version, pip install, git clone, cd, or pyenv.")
        if _looks_like_test_file(rel):
            rules.append("This is a Python test file. Write real unittest or pytest tests, not usage instructions or duplicated app code.")
        if is_probably_entrypoint(rel) and prompt_requests_terminal_input(prompt):
            rules.append("The user prompt asks for terminal/user input. This entrypoint must read input from the user, usually with input(...), then print the requested result.")
    elif suffix in {".js", ".jsx", ".ts", ".tsx"}:
        rules.append("The file must be valid JavaScript/TypeScript source. Do not include shell commands or markdown instructions.")
        if is_probably_entrypoint(rel) and prompt_requests_terminal_input(prompt):
            rules.append("The user prompt asks for terminal/user input. Use readline/process.argv/prompt-style input as appropriate.")
    elif suffix == ".json":
        rules.append("The file must be valid JSON.")
    elif suffix in _CODE_SUFFIXES:
        rules.append("The file must match its extension and contain source/config content, not README prose or code for another language.")
    return "\n".join(f"- {rule}" for rule in rules)


def short_quality_issue_summary(quality_issues: Iterable[Dict[str, Any]], *, max_items: int = 8) -> str:
    parts: List[str] = []
    for item in list(quality_issues or [])[:max_items]:
        path = str(item.get("path") or "[unknown]")
        issue_values = item.get("issues") or []
        if isinstance(issue_values, list):
            issue_text = "; ".join(str(x) for x in issue_values[:3])
        else:
            issue_text = str(issue_values)
        if issue_text:
            parts.append(f"{path}: {issue_text}")
    return " | ".join(parts)


def lesson_from_quality_issues(quality_issues: Iterable[Dict[str, Any]]) -> List[str]:
    lessons: List[str] = []
    blob = short_quality_issue_summary(quality_issues, max_items=20).lower()
    if not blob:
        return lessons
    if "shell" in blob or "command" in blob or "pip" in blob or "git clone" in blob:
        lessons.append("Do not write shell/setup commands into source files, tests, requirements.txt, ignore files, or file paths unless the target file is explicitly a shell script requested by the user.")
    if "user/terminal input" in blob or "does not read input" in blob:
        lessons.append("When the prompt asks for user input, the runnable entrypoint must actually read input and print the requested output.")
    if "test file" in blob or "unittest" in blob or "pytest" in blob:
        lessons.append("Files under tests/ must contain executable tests, not README instructions or duplicated application code.")
    if "syntax" in blob:
        lessons.append("Every generated source file must parse/compile for its language before it is accepted.")
    if "requirements.txt" in blob:
        lessons.append("requirements.txt should contain only package specifiers, or be empty for stdlib-only projects.")
    if "ignore file" in blob:
        lessons.append(".gitignore/.dockerignore should contain file patterns only.")
    if "placeholder filename" in blob or "path contains spaces" in blob or "does not match inferred requested language" in blob:
        lessons.append("Generate meaningful project-relative file paths that match the requested language; never turn commands or code-fence guesses into filenames.")
    return lessons[:8]


def build_quality_lesson_payload(
    *,
    prompt: str,
    project_name: str,
    quality_issues: Iterable[Dict[str, Any]],
    compile_summary: str = "",
    test_summary: str = "",
) -> Dict[str, Any]:
    quality = list(quality_issues or [])[:20]
    return {
        "type": "code_generation_lesson",
        "project_name": project_name,
        "prompt": str(prompt or "")[:4000],
        "quality_issues": quality,
        "compile_summary": str(compile_summary or "")[:4000],
        "test_summary": str(test_summary or "")[:4000],
        "lessons": lesson_from_quality_issues(quality),
    }
