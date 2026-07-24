# Module 2 — Goal Conditioning & Orchestrator Agent (Implementation Plan)

> Status: **implemented** (all 7 phases). 88/88 tests passing.
> Branch (suggested): `feat/m2-goal-orchestrator`
>
> Delivered: `agents/goal_classifier.py`, `agents/planner.py`, rewritten
> `agents/orchestrator.py`; `TaskType`/`GoalClassification`/`ReActStep` schemas;
> directive-aware mining/visualization stubs; `task_type`/`classification`
> surfaced through the API. Tests: `test_goal_classifier.py`, `test_planner.py`,
> updated `test_orchestrator.py`.

Module 2 is the **reasoning and coordination core** of MAGE. It accepts the
user's natural-language goal, classifies the underlying analytical task type,
constructs a **conditional analysis pipeline tailored to that goal**, runs a
ReAct-style loop (reason → act → observe, with a hard step cap), routes
sub-tasks to specialist agents, maintains session state, and aggregates their
outputs into a coherent analysis context.

**Core novelty (FR-02):** the goal conditions *which computations are
prioritized and executed*, not merely which results are surfaced post-hoc.

**Constraint:** no paid APIs. Everything in this module runs locally and free.

---

## Scope note — what is NOT in Module 2

- **RAG / retrieval / grounded recommendations** belong to a **later module**
  (proposal M5 + M6) and are already partially built in `rag/`. We deliberately
  **leave RAG work for that module.** Module 2 only *routes to* the existing
  `RecommendationAgent` as one step in the pipeline; it does not modify RAG.
- Real statistical mining (M3) and visualization (M4) are separate modules.
  In M2 those agents remain stubs but are upgraded to **accept and echo the
  directives** M2 passes them, so goal-conditioning is visible and testable now.

---

## Current state (entering Module 2)

Completed / working:
- **M1 Ingestion** (`agents/ingestion_agent.py`, `data_pipeline/ingestion.py`) —
  real: CSV/TSV/JSON/Parquet/Excel, header validation, profiling → `IngestionResult`.
- Backend (FastAPI), auth, frontend, and the RAG layer (later module) exist.

Not yet built (this module's job):
- **Orchestrator** (`agents/orchestrator.py`) is a **skeleton**: fixed
  `_default_plan`, placeholder `_reason()`, no goal parsing, no task
  classification, no real ReAct loop.
- No goal classifier, no conditional planner.

Dependency to watch (not blocking M2): `IngestionAgent.run()` returns the
`IngestionResult` **profile**, not the raw DataFrame. M2 only needs the column
schema (already in the profile). M3 will need the raw DataFrame handoff.

---

## Task taxonomy → conditional pipeline (the heart of M2)

M2 **defines and passes** these directives. M3/M4 execute them later.

| task_type          | Mining directives (passed via context)                         | Viz directive              |
|--------------------|----------------------------------------------------------------|----------------------------|
| classification     | class balance, feature importance vs target, correlation       | grouped bar / box-by-class |
| regression         | correlation, feature importance, linearity/residual hints      | scatter + heatmap          |
| clustering         | standardize → K-Means + DBSCAN → silhouette                    | scatter / pairplot         |
| anomaly_detection  | IQR + Isolation Forest, distribution tails                     | box / highlighted scatter  |
| reporting          | full descriptive profile, missingness                          | distributions              |

**Acceptance signal:** two different goals on the *same dataset* must produce
*different directives* — this is exactly the same-dataset/different-goal
evaluation design and the project's central claim.

---

## Data flow

```
goal + expertise + IngestionResult(schema)
        │
        ▼
GoalClassifier ──► GoalClassification{task_type, target_column?, confidence, rationale}
        │
        ▼
PipelinePlanner ──► ordered list of PlannedStep(agent_name, directives)
        │
        ▼
Orchestrator ReAct loop:
   for each planned step → REASON → ACT(agent, directives) → OBSERVE(log ReActStep)
   (hard MAX_REACT_STEPS cap; early-stop on required-agent failure)
        │
        ▼
Aggregate ──► AnalysisResponse{task_type, classification, steps[ReActStep], ...}
```

---

## Components & files

### New
- `agents/goal_classifier.py`
  - `GoalClassifier` + a `TaskClassifierProvider` interface (swappable).
  - **Default provider (free, local):** embedding zero-shot — embed the goal
    with the existing `rag.embeddings` model, cosine-compare against short
    task-type descriptions; combine with keyword/schema rules as a
    tiebreaker / high-confidence override.
  - Optional `LLMProvider` behind the same interface — activates *only* if an
    API key is ever configured. **Never required.**
- `agents/planner.py`
  - `PipelinePlanner`: `(TaskType, IngestionResult) → list[PlannedStep]`.
  - Holds the task-taxonomy → directives table above.

### Modified
- `agents/orchestrator.py` — replace `_default_plan` + `_reason()` with:
  classify → plan → real ReAct loop (structured step logging, step cap,
  early-stop) → session-state accumulation → aggregate incl. `task_type`.
- `agents/mining_agent.py`, `agents/visualization_agent.py` — `run(context)`
  reads `directives`/`task_type` and echoes which computations it was told to
  run (still stubs; makes conditioning visible/testable now).
- `backend/schemas/analysis.py` — add `TaskType`, `GoalClassification`,
  `ReActStep{agent_name, action, reasoning, observation, status, latency_ms}`
  (matches the deck's `agent_steps` ER table); add `task_type` + `classification`
  to `AnalysisResponse`.
- `backend/services/orchestrator_service.py`, `backend/routers/analysis.py` —
  surface `task_type`/`classification` in the API response.

> RecommendationAgent / RAG query changes are **deferred to the RAG module.**

---

## ReAct loop design

- **State:** a mutable `context` dict carrying goal, expertise, classification,
  and per-agent outputs (this is the session state).
- **Per step:** `reason` (why this agent now) → `act` (call agent with its
  directives) → `observe` (capture output, status, `latency_ms`; append a
  `ReActStep`).
- **Termination:** plan exhausted OR `MAX_REACT_STEPS` reached OR a hard failure
  in a required agent (ingestion). Runaway-safe.
- **Explainability (FR-06):** every step is a structured, inspectable
  `ReActStep` — the trail the frontend "Agent Timeline" renders.

---

## Why the free classifier satisfies the proposal

The proposal's *"Claude or GPT-4o behind a swappable interface"* describes a
**design pattern**, not a hard dependency. We implement the interface; the
**default implementation is local and free** (embeddings + rules). A paid
provider is an optional plug-in nobody must enable — which makes the
"swappable provider" claim in the report literally true, at zero cost. This
mirrors the RAG layer, which already runs entirely on local `sentence-transformers`.

---

## Testing

- `tests/agents/test_goal_classifier.py` — each of the 5 task types classified
  correctly from representative goals; empty/ambiguous goal → `reporting`.
- `tests/agents/test_planner.py` — each `TaskType` yields the expected directives.
- **Update** `tests/agents/test_orchestrator.py` — current tests assert all four
  agents always run; with a conditional pipeline some steps may be skipped.
  Change to: Ingestion + Mining + Recommendation always present, Visualization
  conditional; assert `task_type` is set and each step is a well-formed `ReActStep`.

---

## Phased build order

1. Schemas (`TaskType`, `GoalClassification`, `ReActStep`, response fields).
2. `GoalClassifier` (embeddings + rules) + tests.
3. `PipelinePlanner` + directive table + tests.
4. Orchestrator ReAct rewrite + step logging.
5. Specialist `run()` signatures accept/echo directives.
6. API wire-through (`task_type`/`classification` in response).
7. Update orchestrator tests; run full suite.

---

## Acceptance criteria

- Two different goals on the same dataset → different pipelines/directives (FR-02).
- Every run returns a `task_type` + a full ordered `ReActStep` trail (FR-06).
- Runs offline with no API key and no cost.
- Hard step cap prevents runaway execution.
