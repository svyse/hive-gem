# Agentic Hive – Functioning Report (Upgraded + Multi-turn Conversations)

This report explains how the **Agentic Hive Studio** works after the upgrades in this package, including:
- multi-orchestrator Q&A
- per-agent-type and hive memory
- logging + voice input
- **multi-turn conversational sessions** (persistent chat threads)

## What was added in this upgrade

### Scale and orchestration
- **Higher default agent spawn limits** (configurable) so you can run larger “hives” simultaneously.
- **A dedicated LoggingAgent** that writes run activity (and Q&A activity) into a text log.
- **A multi-orchestrator Q&A architecture**:
  - **Cybersecurity now includes dedicated Networking and Pentesting specialist agents** (defensive/authorized use).
  - **HiveMasterOrchestrator** (main/router orchestrator for Q&A)
  - Domain orchestrators: **Interactive**, **HR**, **Law (India & International)**, **Finance**, **Economics**, **Social Dynamics & Speech**, **Medicine**, **Cybersecurity**, **Web Design**, **Computer Vision**
- **Orchestrator-to-orchestrator communication**: any orchestrator can spawn and consult other orchestrators.
- **StrategistAgent**: spawnable by any orchestrator to create a strategy/plan.

### Memory + learning
- **Sequence traces (per query/run)**: orchestrators record the *ordered sequence* of agent calls and memory accesses.
- **Sequence learning**: orchestrators look up similar past traces and prefer successful call/routing patterns.

- **Separate type memories per agent type**, including every Q&A agent and every orchestrator.
- **Agents learn from both their type memory and hive memory** on every call.
- **Orchestrators have access to all memory scopes** (hive + type + agent) for routing and context building.

### Verbal requests
- **Voice input in the React UI** using the browser Web Speech API.
- The UI marks prompts/questions as `input_mode=voice` when speech is used.
- The backend commits prompts/questions into **hive memory** and into the **conversation thread**, tagged with the `input_mode`.

### Multi-turn conversational sessions
- Q&A supports **persistent chat threads** via `conversation_id`.
- Each turn appends the user message + hive answer into the conversation.
- Domain orchestrators also append their domain answers into the same conversation, creating **threads per orchestrator**.

Conversation artifacts:
- Q&A log file: `<WORKSPACE_ROOT>/qa_conversations/<conversation_id>/hive.log` (appends across turns)

---

## 1) High-level architecture

The system is a **FastAPI backend** + **React UI**:

- **Code pipeline mode**
  - `POST /api/runs` creates a run
  - `GET /api/runs/{run_id}` polls for progress
  - The **OrchestratorAgent** runs the software-factory pipeline on a project workspace

- **Interactive Q&A mode**
  - `POST /api/qa/ask` runs a Q&A request synchronously
  - A `conversation_id` (optional) allows the request to be appended to an existing chat thread
  - The **HiveMasterOrchestrator** routes the question to domain orchestrators
  - Domain orchestrators spawn specialist Q&A agents and/or other orchestrators

Both modes share:
- **MemoryStore** (hive/type/agent scopes + conversations)
- **MessageBus** (pub/sub for events)
- **AgentRegistry** (spawning/limits/termination)

---

## 2) Runtime components

### AgentRegistry (`backend/app/agents/registry.py`)
Responsible for:
- spawning agents + orchestrators
- enforcing limits
- terminating agents (flush memory + free slots)

**Important behavior:**
- Any agent type ending with `_orchestrator` is treated as an orchestrator for concurrency limits.

Config knobs:
- `AGENT_MAX_PER_TYPE`
- `MAX_ORCHESTRATORS`
- `AGENT_TYPE_LIMITS` (per-type overrides)

### MessageBus (`backend/app/runtime/bus.py`)
- In-process pub/sub.
- The **logging agent** subscribes to `logline`, `qa`, `trace`, and `broadcast` and writes them to disk.

### MemoryStore (`backend/app/memory/store.py`)
MemoryStore uses sqlite with:

#### A) Long-term memory table: `memories`
Fields:
- `scope`: `hive | type | agent`
- `agent_type`, `agent_id`
- `content`, `tags`, `success`, `created_at`

Scopes:
- **Hive memory**: global shared memory
- **Type memory**: per agent type (separate buckets for each Q&A agent type and each orchestrator type)
- **Agent memory**: per spawned instance

### Sequence trace memory (`sequence_trace`)
Orchestrators persist a JSON trace record into **type memory** and **hive memory** after each query/run, tagged with `sequence_trace`.
These traces include ordered events: memory accesses, agent spawns/calls, and orchestrator consultations.

**Learning behavior**
- Q&A agents always retrieve relevant *type* + *hive* memories.
- Orchestrators retrieve across *all scopes* by default.

#### B) Conversation tables: `conversations`, `conversation_messages`
These tables back persistent chat sessions:
- `conversations`: stores `conversation_id`, `title`, `orchestrator`, timestamps, metadata
- `conversation_messages`: ordered messages (`role=user|assistant`), content, orchestrator, input_mode, metadata

Orchestrator threads:
- User messages are included for all threads.
- Each orchestrator appends its own `assistant` message(s) into the conversation.

---

## 3) How a Q&A turn runs (multi-turn aware)

1. Frontend calls `POST /api/qa/ask`.
2. Backend `qa_runner.run_qa(...)`:
   - creates or reuses a `conversation_id`
   - appends the user message into `conversation_messages`
   - commits the user message into **hive memory** (tags include `conversation_id` + `input_mode`)
   - starts a `LoggingAgent` that appends into the conversation log file
3. The `HiveMasterOrchestrator`:
   - builds context (memories + conversation history + optional local references + optional web research)
   - chooses domain orchestrator(s)
   - consults each orchestrator concurrently
4. Each domain orchestrator:
   - builds context (including its orchestrator thread)
   - spawns specialist Q&A agents
   - synthesizes an answer
   - appends a **domain answer** into the conversation thread
5. Hive master returns a final answer.
6. Backend:
   - appends hive master answer into the conversation thread
   - commits the answer into **hive memory**

---

## 4) API summary

### Q&A
- `POST /api/qa/ask`
  - supports `conversation_id` to continue a thread
  - returns `conversation_id`, `turn_id`, and `messages`

### Conversation management
- `POST /api/qa/conversations` – create a conversation
- `GET /api/qa/conversations` – list conversations
- `GET /api/qa/conversations/{conversation_id}` – fetch conversation + messages

### Trace viewer (sequence traces)
The UI can open a **Trace Viewer** panel for any conversation turn.

- `GET /api/qa/conversations/{conversation_id}/traces?turn_id=...&orchestrator=...`
  - lists stored traces for the turn (one per orchestrator that handled the request)
  - includes a summary: query, success, agent_sequence, memory_sequence

- `GET /api/qa/traces/{trace_id}`
  - returns the full trace payload including ordered `events[]`
  - these events include memory accesses, agent calls, and sequence-learning decisions

### Memory
- `GET /api/memory/hive/search?q=...`
- `GET /api/memory/type/{agent_type}/search?q=...`
- `GET /api/memory/all/search?q=...`

---

## 5) Notes / recommended next steps

If you want richer long-running “assistant-like” behavior:
- Add **conversation summarization** after N turns (store summaries in conversation metadata)
- Add per-conversation **system prompts** (stored in conversation metadata)
- Add tools for **document upload** and context grounding for each conversation
