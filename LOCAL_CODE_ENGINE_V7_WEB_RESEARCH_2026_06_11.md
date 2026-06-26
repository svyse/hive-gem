# Local Code Engine v7 - strict write gate + web research context

This patch is cumulative with the earlier local-code-engine patches. It addresses the attached `calculator.zip` output and wires web research into local code-pipeline mode as optional coding context.

## What went wrong in the attached calculator project

The local model did not only write bad Python. It produced a mixed transcript of commands, setup notes, README prose, placeholder filenames, copied snippets, and multiple languages. The pipeline then accepted too much of that text as files.

Examples from the attached project:

- `main.py` was not valid Python and did not reliably implement the requested interactive calculator.
- `.gitignore` contained Python source code instead of ignore patterns.
- `README.md` contained Python application code instead of documentation.
- `requirements.txt` contained command/code-like text instead of package names.
- `tests/test_basic.py` started with a markdown code fence and a huge hallucinated import list.
- `file_1.java`, `main.cpp`, `module_6.py`, `data_1.json`, and similar files were hallucinated placeholder files.
- `python main.py` was created as a file because a shell command was misread as a filename.
- `__pycache__` / `.pyc` artifacts leaked into the generated project ZIP/output.
- `src/calculator.py` and `test_calculator.py` contained broken function calls and duplicated app code instead of proper tests.

## Why this kept happening

The code pipeline was still trusting local-model text too early. The model could drift into markdown, shell commands, or random file names, and earlier validation was not placed strongly enough at the final write boundary. Also, short prompts such as `create calculator` skipped some context collection paths, and the existing `web_research_agent` was effectively bypassed for local coding.

## What v7 changes

### 1. Web research is now wired into local code-pipeline context

For code-pipeline runs, the orchestrator now calls `_collect_web_research()` even when the active LLM backend is local, provided this is enabled:

```env
LOCAL_CODE_WEB_RESEARCH_ENABLED=true
```

This is now the default in `config.py`.

### 2. Local web query generation does not use local JSON

The local model is not asked to generate strict JSON search queries. Instead, local code mode uses deterministic prompt-shaped web queries, such as:

```text
python official docs example create a calculator ...
python project structure tests best practices create a calculator ...
```

This avoids the repeated-token JSON loop while still giving the code engine helpful external context.

### 3. Web research summarization avoids local JSON by default

`WebResearchAgent` now uses deterministic extractive summaries for local backend mode. It passes useful search titles, snippets, URLs, and short fetched excerpts to the code engine without calling the local model for JSON summarization.

The local model can be allowed to summarize web research only if you explicitly enable:

```env
LOCAL_CODE_WEB_RESEARCH_LLM_SUMMARY_ENABLED=true
```

The recommended value is `false` until your local model is reliable at JSON.

### 4. The local code engine now receives `WEB_RESEARCH_CONTEXT`

The manifest and file-generation prompts now include web-research context when available. The local model is instructed to use it only as reference material and still write complete, clean, runnable files.

### 5. Strict generic quality gate is still active

The final write path rejects:

- command-like paths such as `python main.py`, `pip install -r`, `git clone ...`
- placeholder names such as `file_1.java`, `module_6.py`, `data_1.json`, `script_2.sh`
- Java/C++ files in a Python project unless the prompt asks for Java/C++
- Python code inside `.gitignore`, `.dockerignore`, or `README.md`
- prose/commands inside `requirements.txt`
- markdown fences inside source files
- non-test content inside test files
- obvious broken local function arity mistakes
- repeated-token output

### 6. Short prompts no longer skip web/context collection

The orchestrator no longer returns early for short prompts. Short prompts are common in code mode and now still receive memory lessons, RAG context, and optional web research.

## Recommended `.env` values

```env
LOCAL_CODE_ENGINE_ENABLED=true
LOCAL_CODE_ENGINE_VALIDATE_FILES=true
LOCAL_CODE_ENGINE_FILE_RETRIES=2
LOCAL_CODE_ENGINE_DETERMINISTIC_EXAMPLES_ENABLED=false
LOCAL_CODE_ENGINE_ALLOW_LEGACY_NLP_FALLBACK=false
LOCAL_CODE_ENGINE_CLEAN_NEW_PROJECT_ARTIFACTS=true
LOCAL_CODE_LEARNING_LESSONS_ENABLED=true

WEB_RESEARCH_ENABLED=true
LOCAL_CODE_WEB_RESEARCH_ENABLED=true
LOCAL_CODE_WEB_RESEARCH_MAX_QUERIES=3
LOCAL_CODE_WEB_RESEARCH_TIMEOUT_S=25
LOCAL_CODE_WEB_RESEARCH_MAX_CONTEXT_CHARS=5000
LOCAL_CODE_WEB_RESEARCH_LLM_SUMMARY_ENABLED=false
```

## Apply

Run from your project root:

```powershell
cd C:\Users\shera\Desktop\gen_v2
Expand-Archive -Path .\hive_050626_local_code_engine_v7_web_research.zip -DestinationPath . -Force
```

Then restart backend and frontend.

## Expected logs

For local code-pipeline mode you should now see logs like:

```text
orchestrator:... | gathering context
orchestrator:... | web research: 2 queries (no scaling)
web_research:... | web research: python official docs example ...
module:... | local code engine: generating project manifest from NLP prompt
module:... | local code engine: writing main.py
logic:... | local NLP/file plan detected; skipped local JSON logic review
```

If web search fails or times out, the code pipeline continues:

```text
web research skipped/failed: ...
```

Web research improves context, but it does not replace validation. The strict quality gate still decides whether files are safe enough to write.
