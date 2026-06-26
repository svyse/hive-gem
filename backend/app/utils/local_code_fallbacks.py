from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\\", "/").strip().lower())


def _deterministic_examples_enabled() -> bool:
    """Opt-in only for legacy demo templates.

    The general local code engine should build projects from the user's NLP
    prompt, not from hardcoded hello-world/calculator templates. Keep these
    old templates available only as an explicit emergency switch.
    """

    raw = os.getenv("LOCAL_CODE_ENGINE_DETERMINISTIC_EXAMPLES_ENABLED")
    if raw is None:
        try:
            from app.core.config import settings

            raw = getattr(settings, "local_code_engine_deterministic_examples_enabled", False)
        except Exception:
            raw = False
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _project_signal(project_root: Any | None) -> str:
    if project_root is None:
        return ""
    raw = str(project_root or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("\\", "/")
    name = normalized.rstrip("/").split("/")[-1]
    return f"{raw} {name}"


def simple_python_signal_from_json_blob(blob: str) -> str:
    """Extract the useful task/path signal from a structured prompt blob.

    Local JSON prompts contain a lot of schema text.  For deterministic fallbacks
    we only want the user task and project path, because the schema itself can be
    misleading.  This helper is intentionally forgiving and works even when the
    surrounding model prompt is not a single JSON document.
    """

    text = str(blob or "")
    chunks: list[str] = []
    for key in ("user_prompt", "prompt", "project_root", "project_path", "path"):
        pattern = rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"'
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw = match.group(1)
            try:
                value = json.loads('"' + raw + '"')
            except Exception:
                value = raw.replace('\\n', ' ').replace('\\\\', '\\')
            value = str(value or "").strip()
            if value and value not in chunks:
                chunks.append(value)

    # Also include obvious Windows/POSIX project path snippets, because some
    # frontend flows send the project name more reliably than the natural-language prompt.
    for match in re.finditer(r'(?i)(?:[A-Za-z]:)?[\/][^\n\r"{}]*hello[_\- ]?(?:py|python|world)[^\n\r"{} ]*', text):
        value = match.group(0).strip()
        if value and value not in chunks:
            chunks.append(value)

    for match in re.finditer(r'(?i)(?:[A-Za-z]:)?[\/][^\n\r"{}]*(?:calculator|calc)[_\- ]?(?:py|python|cli|app)?[^\n\r"{} ]*', text):
        value = match.group(0).strip()
        if value and value not in chunks:
            chunks.append(value)

    return "\n".join(chunks)


def looks_like_simple_python_project_request(
    prompt: str,
    *,
    project_root: Any | None = None,
    extra_signal_text: str | None = None,
) -> bool:
    if not _deterministic_examples_enabled():
        return False
    """Detect tiny deterministic Python starter requests.

    This deliberately targets Hello World / starter Python prompts and hello_py
    sample project runs.  The code pipeline's local JSON prompts are much harder
    for small local models than plain Q&A, so these requests should not depend on
    a JSON planner, reviewer, Docker planner, or test-scaffold generator.
    """

    signal = "\n".join(
        part
        for part in (
            str(prompt or ""),
            _project_signal(project_root),
            str(extra_signal_text or ""),
        )
        if str(part or "").strip()
    )
    q = _compact(signal)
    if not q:
        return False

    words = q.split()
    if len(words) > 160:
        # Long prompts may contain hello-world examples inside unrelated context.
        # Let the normal planner handle them unless the active project name itself
        # is a hello_py starter.
        project_q = _compact(_project_signal(project_root) + " " + str(prompt or ""))
        if "hello_py" not in project_q and "hello-python" not in project_q and "hello_python" not in project_q:
            return False

    complex_markers = (
        "api",
        "backend",
        "frontend",
        "database",
        "sqlite",
        "postgres",
        "mysql",
        "mongodb",
        "flask",
        "django",
        "fastapi",
        "streamlit",
        "tkinter",
        "gui",
        "web app",
        "rest",
        "login",
        "auth",
        "docker compose",
        "microservice",
        "package",
        "library",
        "cli with",
        "argument parser",
    )
    if any(marker in q for marker in complex_markers):
        return False

    hello_markers = (
        "hello world",
        "hello-world",
        "hello_world",
        "helloworld",
        "hello, world",
        "hello py",
        "hello_py",
        "hello-python",
        "hello_python",
        "print hello",
        "prints hello",
        "say hello",
        "says hello",
        "hello program",
        "hello project",
    )
    python_markers = (
        "python",
        ".py",
        " py ",
        "py script",
        "py code",
        "python script",
        "python code",
        "hello_py",
        "hello-python",
        "hello_python",
    )
    code_intent_markers = (
        "code",
        "script",
        "program",
        "project",
        "app",
        "file",
        "sample_projects",
        "sample project",
    )
    create_intent_markers = (
        "make",
        "create",
        "build",
        "write",
        "generate",
        "new",
        "simple",
        "starter",
        "sample",
    )

    has_hello = any(term in q for term in hello_markers)
    has_python = any(term in q for term in python_markers)
    has_code_intent = any(term in q for term in code_intent_markers)
    has_create_intent = any(term in q for term in create_intent_markers)

    # The frontend run path frequently includes sample_projects/hello_py_vN.
    hello_project_path = bool(re.search(r"(?:^|[/ _-])hello[_ -]?(?:py|python)(?:[_ -]?v?\d+)?(?:$|[/ _-])", q))

    return bool((has_hello or hello_project_path) and (has_python or has_code_intent) and (has_create_intent or has_code_intent))


def build_simple_python_project_plan(
    user_prompt: str,
    *,
    reason: str | None = None,
    project_root: Any | None = None,
    extra_signal_text: str | None = None,
) -> Optional[Dict[str, Any]]:
    if not _deterministic_examples_enabled():
        return None
    if not looks_like_simple_python_project_request(
        user_prompt,
        project_root=project_root,
        extra_signal_text=extra_signal_text,
    ):
        return None

    note = "Deterministic local code-pipeline fallback used for a simple Python Hello World request."
    if reason:
        note += f" Reason: {reason}"

    return {
        "summary": "Created a minimal Python Hello World project.",
        "operations": [
            {
                "op": "write_file",
                "path": "main.py",
                "content": (
                    "def main() -> None:\n"
                    "    print(\"Hello, World!\")\n"
                    "\n"
                    "\n"
                    "if __name__ == \"__main__\":\n"
                    "    main()\n"
                ),
            },
            {
                "op": "write_file",
                "path": "README.md",
                "content": (
                    "# Hello World Python Project\n"
                    "\n"
                    "Run the project with:\n"
                    "\n"
                    "```bash\n"
                    "python main.py\n"
                    "```\n"
                    "\n"
                    "Expected output:\n"
                    "\n"
                    "```text\n"
                    "Hello, World!\n"
                    "```\n"
                ),
            },
        ],
        "test_commands": ["python main.py"],
        "notes": note,
        "metadata": {
            "deterministic_fallback": True,
            "fallback_kind": "simple_python_hello_world",
            "skip_local_json_plan": True,
            "skip_local_json_review": True,
            "skip_local_json_tests": True,
            "skip_local_json_docker": True,
        },
    }

def looks_like_calculator_python_project_request(
    prompt: str,
    *,
    project_root: Any | None = None,
    extra_signal_text: str | None = None,
) -> bool:
    if not _deterministic_examples_enabled():
        return False
    signal = "\n".join(
        part
        for part in (
            str(prompt or ""),
            _project_signal(project_root),
            str(extra_signal_text or ""),
        )
        if str(part or "").strip()
    )
    q = _compact(signal)
    if not q:
        return False

    if len(q.split()) > 220:
        project_q = _compact(_project_signal(project_root) + " " + str(prompt or ""))
        if "calculator" not in project_q and "calc" not in project_q:
            return False

    complex_markers = (
        "api",
        "backend",
        "frontend",
        "database",
        "sqlite",
        "postgres",
        "mysql",
        "mongodb",
        "flask",
        "django",
        "fastapi",
        "streamlit",
        "tkinter",
        "gui",
        "web app",
        "rest",
        "login",
        "auth",
        "docker compose",
        "microservice",
    )
    if any(marker in q for marker in complex_markers):
        return False

    has_calculator = any(term in q for term in ("calculator", "basic calc", "arithmetic calculator", "simple calc"))
    has_python = any(term in q for term in ("python", ".py", " py ", "python script", "python code", "python project"))
    has_cli_or_input = any(
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
            "display result",
        )
    )
    has_arithmetic = any(
        term in q
        for term in (
            "add",
            "addition",
            "subtract",
            "subtraction",
            "multiply",
            "multiplication",
            "divide",
            "division",
            "operation",
            "operators",
            "+",
            "-",
            "*",
            "/",
        )
    )
    has_create_intent = any(term in q for term in ("make", "create", "build", "write", "generate", "new", "simple", "project", "program", "app", "script"))
    calculator_project_path = bool(re.search(r"(?:^|[/ _-])(?:calculator|calc)[_ -]?(?:py|python|cli)?(?:[_ -]?v?\d+)?(?:$|[/ _-])", q))

    return bool(
        (has_calculator or calculator_project_path)
        and (has_python or has_cli_or_input or calculator_project_path)
        and (has_arithmetic or has_cli_or_input or calculator_project_path)
        and (has_create_intent or calculator_project_path)
    )


def build_python_cli_calculator_project_plan(
    user_prompt: str,
    *,
    reason: str | None = None,
    project_root: Any | None = None,
    extra_signal_text: str | None = None,
) -> Optional[Dict[str, Any]]:
    if not _deterministic_examples_enabled():
        return None
    if not looks_like_calculator_python_project_request(
        user_prompt,
        project_root=project_root,
        extra_signal_text=extra_signal_text,
    ):
        return None

    note = "Deterministic local code-pipeline fallback used for a Python CLI calculator request."
    if reason:
        note += f" Reason: {reason}"

    main_py = '''SUPPORTED_OPERATIONS = {
    "1": "add",
    "+": "add",
    "add": "add",
    "2": "subtract",
    "-": "subtract",
    "subtract": "subtract",
    "3": "multiply",
    "*": "multiply",
    "multiply": "multiply",
    "4": "divide",
    "/": "divide",
    "divide": "divide",
}


def calculate(left: float, right: float, operation: str) -> float:
    """Return the result for the requested arithmetic operation."""
    normalized = SUPPORTED_OPERATIONS.get(str(operation).strip().lower())
    if normalized == "add":
        return left + right
    if normalized == "subtract":
        return left - right
    if normalized == "multiply":
        return left * right
    if normalized == "divide":
        if right == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return left / right
    raise ValueError(f"Unsupported operation: {operation}")


def read_number(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def read_operation() -> str:
    print("Select an operation:")
    print("1. Add (+)")
    print("2. Subtract (-)")
    print("3. Multiply (*)")
    print("4. Divide (/)")
    return input("Operation: ").strip()


def format_result(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


def main() -> None:
    print("Python Calculator")
    left = read_number("Enter the first value: ")
    right = read_number("Enter the second value: ")
    operation = read_operation()
    try:
        result = calculate(left, right, operation)
    except (ValueError, ZeroDivisionError) as exc:
        print(f"Error: {exc}")
        return
    print(f"Result: {format_result(result)}")


if __name__ == "__main__":
    main()
'''
    test_py = '''import unittest

from main import calculate


class CalculatorTests(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(calculate(2, 3, "+"), 5)

    def test_subtract(self) -> None:
        self.assertEqual(calculate(7, 4, "subtract"), 3)

    def test_multiply(self) -> None:
        self.assertEqual(calculate(6, 5, "*"), 30)

    def test_divide(self) -> None:
        self.assertEqual(calculate(8, 2, "/"), 4)

    def test_divide_by_zero(self) -> None:
        with self.assertRaises(ZeroDivisionError):
            calculate(8, 0, "/")

    def test_unknown_operation(self) -> None:
        with self.assertRaises(ValueError):
            calculate(1, 2, "power")


if __name__ == "__main__":
    unittest.main()
'''
    readme = '''# Python CLI Calculator

A small command-line calculator that asks the user for two values, lets the user choose an operation, and displays the result.

## Run

```bash
python main.py
```

## Test

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Supported operations:

- Add: `1`, `+`, or `add`
- Subtract: `2`, `-`, or `subtract`
- Multiply: `3`, `*`, or `multiply`
- Divide: `4`, `/`, or `divide`
'''
    return {
        "summary": "Created a Python CLI calculator project with user input, arithmetic operations, and tests.",
        "operations": [
            {"op": "write_file", "path": "main.py", "content": main_py},
            {"op": "mkdir", "path": "tests"},
            {"op": "write_file", "path": "tests/test_calculator.py", "content": test_py},
            {"op": "write_file", "path": "README.md", "content": readme},
        ],
        "test_commands": ["python -m unittest discover -s tests -p test_*.py"],
        "notes": note,
        "metadata": {
            "deterministic_fallback": True,
            "fallback_kind": "python_cli_calculator",
            "skip_local_json_plan": True,
            "skip_local_json_review": True,
            "skip_local_json_tests": True,
            "skip_local_json_docker": True,
        },
    }


def build_known_python_project_plan(
    user_prompt: str,
    *,
    reason: str | None = None,
    project_root: Any | None = None,
    extra_signal_text: str | None = None,
) -> Optional[Dict[str, Any]]:
    return (
        build_simple_python_project_plan(
            user_prompt,
            reason=reason,
            project_root=project_root,
            extra_signal_text=extra_signal_text,
        )
        or build_python_cli_calculator_project_plan(
            user_prompt,
            reason=reason,
            project_root=project_root,
            extra_signal_text=extra_signal_text,
        )
    )


def looks_like_known_python_project_request(
    prompt: str,
    *,
    project_root: Any | None = None,
    extra_signal_text: str | None = None,
) -> bool:
    return bool(
        looks_like_simple_python_project_request(prompt, project_root=project_root, extra_signal_text=extra_signal_text)
        or looks_like_calculator_python_project_request(prompt, project_root=project_root, extra_signal_text=extra_signal_text)
    )


def plan_is_deterministic_fallback(plan: Dict[str, Any] | None) -> bool:
    if not isinstance(plan, dict):
        return False
    meta = plan.get("metadata") or {}
    return isinstance(meta, dict) and bool(meta.get("deterministic_fallback"))


def plan_has_meaningful_operations(plan: Dict[str, Any] | None) -> bool:
    if not isinstance(plan, dict):
        return False
    ops = plan.get("operations") or []
    if not isinstance(ops, list):
        return False
    safe_ops = {"mkdir", "write_file", "delete_file"}
    for op in ops:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("op") or "")
        path = str(op.get("path") or "").strip()
        if kind not in safe_ops or not path:
            continue
        if kind == "write_file" and not isinstance(op.get("content"), str):
            continue
        return True
    return False


def plan_looks_like_repetition_noop(plan: Dict[str, Any] | None) -> bool:
    if not isinstance(plan, dict):
        return False
    if plan_has_meaningful_operations(plan):
        return False
    blob = " ".join(str(plan.get(k) or "") for k in ("summary", "notes")).lower()
    markers = (
        "repeated-token",
        "repeated token",
        "model repeated",
        "generation collapsed",
        "skipped generated operations",
        "unsafe repeated-token",
        "could not parse json from model output",
        "logicagent review must be a json object",
        "no generated file changes were applied",
        "local json review failed",
    )
    return any(marker in blob for marker in markers)


def maybe_simple_python_plan_fallback(
    *,
    user_prompt: str,
    plan: Dict[str, Any] | None,
    reason: str | None = None,
    project_root: Any | None = None,
    extra_signal_text: str | None = None,
) -> Optional[Dict[str, Any]]:
    if plan_has_meaningful_operations(plan) and not plan_looks_like_repetition_noop(plan):
        return None
    return build_known_python_project_plan(
        user_prompt,
        reason=reason,
        project_root=project_root,
        extra_signal_text=extra_signal_text,
    )
