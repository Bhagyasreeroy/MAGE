# MAGE — Continuation Plan & Full Context Handoff

> **Purpose of this file:** a complete, self-contained briefing so a fresh Claude session (or a new
> team member) can continue building MAGE with zero context loss. Read this file top to bottom
> before touching code.
>
> **Written:** 11 August 2026 · **Branch at time of writing:** `feat/m2-goal-orchestrator`
> **Last full code audit:** 11 Aug 2026 (this document) · previous audit: `docs/PROJECT_PROGRESS.md` (24 Jul 2026)

---

## ⏰ READ THIS FIRST — the timeline is the binding constraint

| Milestone | Planned dates | Status as of **11 Aug 2026** |
|---|---|---|
| M1–M5 | Jun 2 – Jul 28 | Windows closed. Built, with known gaps (below) |
| **M6** Recommendation + SHAP/LIME | Jul 14 – **Aug 4** | ✅ **SHAP + 3rd register done 11 Aug** |
| **M7** Frontend (live dashboard) | Jul 21 – **Aug 11** | ✅ **Live streaming dashboard done 11 Aug** |
| **M8** API (WebSocket, Celery) | Jul 28 – **Aug 11** | 🟡 **WebSocket done 11 Aug**; Celery still absent |
| **Integration & E2E testing** | **Aug 11 – Aug 18** | 🔵 In progress. **FR-05 benchmarked 11 Aug (max 2.03s)** |
| **Empirical evaluation** | **Aug 11 – Aug 18** | ✅ **Harness built & run 11 Aug — `evaluation/`** |
| **Report writing & submission** | **Aug 18 – Aug 25** | Final deadline **25 Aug 2026** |

**You have ~14 days to submission.** This is not a "build everything" plan — it is a triage plan.
Section 7 gives the prioritized order. Do not start optional work before the must-haves.

---

## 1. Project identity

**MAGE — Multimodal Agentic Goal-conditioned EDA with RAG-grounded Recommendations**

- **Type:** MCA IV trimester Specialization Project, CHRIST (Deemed to be University)
- **Submission:** August 2026
- **Team:** Bhagyasree Roy (2547118) · Nandini Singh (2547135) · Neha N (2547160)
- **Guide:** Dr. Tegil J John
- **Proposal PDF:** `Project Proposal (2).pdf` (14 pages) — the authoritative spec
- **Presentation:** `Project_Presentation.pptx (2).pdf`

### The aim (one sentence)

Build an EDA system where the user's **natural-language goal decides which statistical
computations actually run** — not just which results get shown — where **every recommendation
carries a citation** back to retrievable methodology, and where the **same analysis is explained
differently** depending on the reader's expertise; then **prove** the goal-conditioning works via a
same-dataset/different-goal experiment against a generic AutoEDA baseline.

### The problem being solved

- **Rule-based AutoEDA** (ydata-profiling, Sweetviz, AutoViz) runs a *fixed* computation set on every
  dataset. A churn investigation and a pricing analysis on identical data produce identical output.
- **LLM agents** (LIDA, DataInterpreter, DS-Agent) do use the goal — but to shape which *questions*
  or *charts* appear, not which *computations* run. Their advice is ungrounded and unverifiable.
- Both produce **one fixed output register** regardless of who is reading.

The proposal's key line, worth memorising for the viva:
> *"Goal-awareness without methodological grounding is not actually progress — it just moves the
> trust problem from generic to ungrounded."*

### The stated contribution (do NOT overclaim)

The proposal explicitly concedes MAGE is **not** "the first goal-aware EDA system." The contribution
is a *specific, testable combination*: computation-level goal-conditioning **+** RAG-grounded
citability **+** role-adaptive output, **empirically evaluated** against a generic baseline.

### The 7 objectives

| # | Objective | Delivered? |
|---|---|---|
| 1 | **Multimodal ingestion** — CSV/Excel/PDF/scanned/DB → one unified schema | 🟢 90% (OCR deferred) |
| 2 | **Goal-conditioned planning** — classify task type, build *conditional* pipeline | ✅ ~90% — **this is the thesis and it works** |
| 3 | **Multi-agent orchestration** — planner routes to specialists, aggregates | 🟡 ~75% (rule-based, not model-driven) |
| 4 | **RAG-grounded recommendations** — KB + run-memory, citable not improvised | 🟡 ~70% (**run-memory absent**) |
| 5 | **Explainability** — feature-level explanations + confidence scores | 🟢 ~85% (**SHAP done 11 Aug**; LIME not attempted) |
| 6 | **Role-adaptive output** — technical vs. plain-language registers | 🟢 ~85% (**3 of 3 done 11 Aug**, deterministic) |
| 7 | **Empirical validation** — same-dataset/different-goal vs. AutoEDA baseline | ✅ **~90% — harness built & run (11 Aug), see `evaluation/`** |

---

## 2. Repository map

```
MAGE/
├── agents/                        # ⭐ The multi-agent layer (~2,000 LOC)
│   ├── orchestrator.py       230 L  OrchestratorAgent — coordinator, owns the agent registry
│   ├── goal_classifier.py    295 L  NL goal → TaskType (keywords + embedding fallback)
│   ├── planner.py            142 L  TaskType → per-agent directives  ← THE CONDITIONING LIVES HERE
│   ├── ingestion_agent.py    256 L  Load + normalize any source
│   ├── mining_agent.py       571 L  Statistics/ML, gated by directives
│   ├── visualization_agent.py 316 L Chart *spec* selection (does not draw)
│   ├── recommendation_agent.py 275 L RAG retrieval + citation + dual register
│   └── qa_agent.py           242 L  Follow-up Q&A, grounded in the same KB
│
├── backend/                       # FastAPI app
│   ├── main.py                    App entry, router mounting, CORS, session middleware
│   ├── core/config.py         58 L Pydantic Settings singleton (`settings`)
│   ├── core/database.py       62 L Async SQLAlchemy engine + Base
│   ├── core/deps.py           62 L get_db, get_current_user
│   ├── core/security.py       88 L JWT create/verify, password hashing
│   ├── models/                    ORM: user.py, dataset.py, analysis_run.py  (3 tables only)
│   ├── schemas/analysis.py        Pydantic contracts (TaskType, ExpertiseLevel, ReActStep, …)
│   ├── routers/
│   │   ├── analysis.py       304 L run, ingest, history, datasets, 3× export
│   │   ├── auth.py           292 L register/login/refresh/me/password/delete
│   │   ├── oauth.py          131 L Google login + callback
│   │   └── health.py          27 L
│   └── services/
│       ├── orchestrator_service.py 129 L HTTP ↔ agent bridge, persistence
│       ├── dataset_service.py      100 L save/get/stats, bytes stored in Postgres
│       ├── analysis_run_service.py  64 L save/list/get runs
│       └── export_service.py       365 L PDF (ReportLab) + JSON + BibTeX
│
├── frontend/                      # Next.js 14 App Router + TS + Tailwind
│   └── app/
│       ├── page.tsx               Landing (goal input + expertise selector)
│       ├── signin/ signup/ auth-callback/
│       ├── dashboard/
│       │   ├── page.tsx           Dashboard home
│       │   ├── analysis/[id]/     Report view + 3 export cards  ← where a live view would go
│       │   ├── datasets/ knowledge/ settings/
│       │   └── layout.tsx
│       ├── components/charts.tsx  Renders VisualizationAgent specs
│       ├── components/markdown.tsx
│       └── lib/api.ts             All fetch wrappers + token handling
│       └── lib/auth-context.tsx   ⚠️ HAS UNCOMMITTED CHANGES (see §9)
│
├── evaluation/                    # ⭐ Objective 7 — the validation harness (11 Aug)
│   ├── datasets.py                5 pinned datasets (4 sklearn + 1 seeded synthetic)
│   ├── baseline.py                Goal-invariant AutoEDA baseline (spec + live validator)
│   ├── harness.py                 The same-dataset/different-goal sweep
│   ├── metrics.py                 Divergence, precision/recall/F1, citation coverage
│   ├── report.py                  → results.json + results.md
│   └── BASELINE_VALIDATION.md     Proof the pinned spec matches real ydata-profiling
│
├── rag/
│   ├── embeddings.py           80 L  all-MiniLM-L6-v2, local, 384-dim
│   ├── vector_store.py        235 L  FAISS + Chroma behind one interface
│   └── knowledge_loader.py    210 L  frontmatter parse + table-aware chunking
│
├── data_pipeline/
│   ├── ingestion.py           411 L  DataIngestionEngine: files, PDF, DB URI, REST
│   └── processing.py          101 L  Pandas only; Spark/Dask raise NotImplementedError
│
├── data/knowledge_base/           5 markdown methodology docs (~243 lines total)
├── tests/                         385 test functions across 25 files
├── docs/                          architecture.md, api_contracts.md, PROJECT_PROGRESS.md,
│                                  auth_documentation.md, module2_plan.md, this file
├── infra/k8s/README.md            Placeholder only
├── docker-compose.yml             backend + frontend + postgres (Chroma is embedded, no container)
└── scripts/generate_project_report.py   ⚠️ UNTRACKED — builds MAGE_Project_Report.pdf
```

---

## 3. How to run it

```bash
# Backend (from repo root — PYTHONPATH=. matters, the agents/ package is imported from root)
cd backend && pip install -e ".[dev]" && cd ..
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev        # http://localhost:3000

# Full stack
docker compose up --build                        # backend:8000, frontend:3000, postgres:5432

# Tests — MUST pass before any commit
PYTHONPATH=. pytest tests/ -v
```

**Environment:** copy `.env.example` → `.env`. Everything works **without any API key** — embeddings
are local. Only Google OAuth needs `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`.

---

## 4. Architecture — how it actually works

### The pipeline

```
User goal ──▶ GoalClassifier ──▶ PipelinePlanner ──▶ OrchestratorAgent
 "find fraud"    anomaly_detection   4 PlannedSteps      drives them in order
                                     with directives
      ┌──────────────────────────────────────┴─────────────────────────┐
      ▼                ▼                    ▼                          ▼
 IngestionAgent   MiningAgent      VisualizationAgent      RecommendationAgent
 df + schema      only the         only the charts         RAG retrieve → cite →
                  computations     the plan named          technical + plain text
                  the plan named
      └──────────────────────── every step logged Reason/Act/Observe ──┘
                                          │
                            Persisted to analysis_runs
                                          │
                          Export: PDF / JSON / BibTeX
```

### The 5 goal types and what each runs

`agents/planner.py:44-71` is the single source of truth:

| TaskType | Mining computations | Priority | Charts |
|---|---|---|---|
| `classification` | class_balance, feature_importance, correlation | feature_importance | grouped_bar, box_by_class |
| `regression` | correlation, feature_importance, linearity_check | correlation | scatter, correlation_heatmap |
| `clustering` | standardize, kmeans, dbscan, silhouette | kmeans | cluster_scatter, pairplot |
| `anomaly_detection` | iqr_outliers, isolation_forest, distribution_tails | isolation_forest | box, highlighted_scatter |
| `reporting` *(default)* | descriptive_profile, missingness, distribution | descriptive_profile | histograms, missingness_matrix |

### Design decisions you must not accidentally undo

1. **`MiningAgent.run()` gating logic** (`mining_agent.py:100-142`). A `wants(*tokens)` helper returns
   `True` when **either** the token was requested **or** no directives were given at all. This is what
   keeps a bare `MiningAgent().run()` returning the full default profile, so pre-existing tests stay
   green. Four computations are **strictly opt-in** and bypass `wants()` entirely:
   `isolation_forest`, `dbscan`, `class_balance`, `linearity_check`.
2. **VisualizationAgent emits *specs*, never images.** Plain dicts. The frontend renders them in
   React; `export_service.py` renders the *same* specs in ReportLab. One source of truth, two
   renderers. Do not make this agent produce PNGs.
3. **Export never recomputes.** `export_service` reads only the persisted `AnalysisRun`, guaranteeing
   the PDF matches what the user saw. Keep it that way.
4. **`computations_run` is the evidence field.** Every mining result includes it. It is what proves
   FR-02 and what the evaluation harness will measure. Never remove it.
5. **ID-column exclusion.** `_is_identifier_column()` keeps row-IDs and keys out of clustering and
   feature importance. Removing it silently degrades every model.
6. **Specialists are isolated.** Only the orchestrator knows the others exist. Agents must not import
   each other.
7. **Ownership checks on every read/export.** `_get_owned_run()` 404s another user's run.

### Data model — only 3 tables

- `users` — id, email, full_name, hashed_password (nullable for OAuth), auth_provider, is_active,
  **default_expertise_level**, timestamps
- `datasets` — id, user_id, filename, **content (LargeBinary — file bytes live in Postgres)**,
  row_count, column_count, created_at
- `analysis_runs` — id, user_id, dataset_id, goal, expertise_level, status, summary,
  **steps (JSON)**, **recommendations (JSON)**, **rag_sources (JSON)**, created_at

The ER diagram in the proposal also specifies a **`run_memory`** table. **It does not exist.** That is
why Objective 4's "improves with use" benefit is currently unrealized.

---

## 5. Current state — module by module

### M1 Ingestion — 🟢 ~90%
**Done:** CSV, TSV, JSON, Parquet, Excel (XLSX/XLS); PDF tables via `pdfplumber`; DB URI via
SQLAlchemy (table / raw query / `#fragment` selection); REST/HTTP via `httpx` with content-type
sniffing; `classify_source()` auto-routing; header normalization; canonical schema; persistence.
**Missing:** OCR for scanned docs (**deliberately deferred**); no dedicated API endpoint exposing
DB/REST sources (engine supports them, only file upload is wired to HTTP).

### M2 Goal Conditioning & Orchestrator — ✅ ~90% (**the thesis, and it works**)
**Done:** weighted keyword lexicon → local-embedding prototype fallback → safe `reporting` default;
5 task types; target-column extraction refined post-ingestion; conditional planner; orchestrator
with agent registry; full Reason/Act/Observe log with per-step latency; **directives honored
downstream by both Mining and Visualization**; proven by `tests/agents/test_goal_conditioning.py`.
**Known weakness (state it honestly):** the "ReAct loop" is a `for` loop over a statically-built
4-step plan, not a model-driven reason-act-observe cycle. Consequence: `MAX_REACT_STEPS = 10`
(`orchestrator.py:39`) is **dead code** — the plan is always exactly 4 steps, so the cap can never
fire. Do not describe it as an active safety guard.

### M3 Mining — 🟢 ~85%
**Done:** descriptive stats (numeric + categorical), data quality (completeness/uniqueness/missing),
Pearson correlation, IQR outliers, PCA feature importance, KMeans with silhouette-selected `k`,
Isolation Forest, DBSCAN with k-distance `eps` heuristic, class balance + imbalance ratio,
linearity check, pattern synthesis, ID-column exclusion.
**Missing:** time-series decomposition (named in the module spec); target-aware supervised feature
importance (linearity partly covers it; PCA is still the default ranker).

### M4 Visualization — 🟡 ~70%
**Done:** correlation heatmap, feature-importance bar, cluster scatter (2D PCA), histograms,
boxplots, categorical bars, scatter (two most-correlated numerics), missingness matrix;
goal-conditioned chart selection with graceful fallback to the default set when nothing is feasible.
**Missing:** `grouped_bar`, `box_by_class`, `pairplot`, `highlighted_scatter` are **routed to
approximations** (`visualization_agent.py:112-133`) not dedicated builders; violin plots; line charts
for temporal data; choropleths.

### M5 RAG — 🟡 ~70%
**Done:** local `all-MiniLM-L6-v2` (384-dim, no API key); FAISS **and** Chroma behind one interface;
FAISS index persisted to disk; frontmatter parsing; overlap chunking that **won't split a markdown
table mid-row**; top-k retrieval wired into recommendations.
**KB contents (5 docs, ~243 lines):**
| File | Title | Teaches |
|---|---|---|
| `clustering.md` | Clustering Method Selection | KMeans vs DBSCAN vs Hierarchical; elbow + silhouette; k-distance for `eps` |
| `correlation_analysis.md` | Correlation and Association Analysis | Pearson/Spearman/Kendall/Cramér's V decision table; \|r\| bands; multicollinearity warning |
| `outlier_detection.md` | Outlier Detection Methods | IQR 1.5× rule; why z-score is self-sabotaging; Isolation Forest for multivariate |
| `missing_values.md` | Handling Missing Values | MCAR/MAR/MNAR taxonomy; >60% drop / 5–40% impute; strategy per column type |
| `distribution_profiling.md` | Distribution Profiling & Viz Selection | Skew/kurtosis diagnostics; goal→chart table; **contains an explicit goal-conditioning note** |
**Missing:** corpus is small — goals outside these 5 topics retrieve weakly or get filtered by the
`MIN_CONFIDENCE = 0.15` threshold; **`run_memory` store absent**; incremental cross-session
indexing (NFR-02) unverified.

### M6 Recommendation & Explainability — 🟡 ~65% ⚠️ **OVERDUE**
**Done:** query built from goal + actual mining findings; top-k retrieval; `MIN_CONFIDENCE = 0.15`
gate; source de-duplication so you don't get 5 recs citing one doc; ranked output; dual register
(`text_technical` / `text_plain`); `QAAgent` for follow-ups.
**Missing:** **SHAP / LIME — zero code anywhere** (explicit FR + Objective 5, and named in the
module description); only 2 of 3 registers (spec says Beginner / Analyst / Data Scientist); text is
deterministic paraphrase, not LLM synthesis.

### M7 Frontend — 🟡 ~75% ⚠️ **DUE TODAY**
**Done:** landing, signin, signup, auth-callback, dashboard home, analysis detail (report view),
datasets, knowledge, settings; Google OAuth end-to-end; `middleware.ts` route protection; chart
rendering from specs; markdown rendering; **3 export download cards with per-format loading state**.
**Missing:** **live analysis dashboard with real-time WebSocket agent-step streaming** — the
headline UI feature in the module spec; the onboarding *wizard* is a single-page form rather than the
specified 3-view wizard; verify the 3-way expertise selector drives the register end-to-end.

### M8 API & Auth — 🟡 ~70% ⚠️ **DUE TODAY**
**Done:** `POST /analysis/run`, file ingest, history list + detail, datasets list + delete,
`/export/pdf|json|citations`, full auth suite (register/login/refresh/me/patch/password/delete),
Google OAuth, health; JWT + refresh tokens; Postgres async; Pydantic validation; expertise plumbed
through; **PDF (ReportLab, with charts) + JSON + BibTeX export**; per-user ownership enforcement.
**Missing:** **WebSocket endpoint**; **Celery + Redis job queue** (analysis runs *synchronously* —
the HTTP request is held open for the whole pipeline); rate limiting; session-scoped file sandboxing
+ delete-on-session-end (NFR-04 — current behaviour is "keep everything forever").

### Export / reporting — ✅ the most complete area
`export_service.py` (365 L): PDF with title block, goal, executive summary, data-quality table, up to
**6 charts actually rendered** (bar / histogram / scatter colour-grouped by cluster / heatmap table /
boxplot-as-five-number-table), recommendations, and a "Grounded In" citation list with real document
titles (module-scope `_TITLE_CACHE`). Per-chart try/except so one bad spec never kills the export.
JSON = full run payload including every agent step. BibTeX = `@misc` per grounded source.
`tests/backend/test_export.py` has 8 tests, **5 of them security** (401s, 404s, cross-user isolation).
**Left here:** step log absent from the PDF (only in JSON) — adding it would strengthen FR-06;
boxplots degrade to a table; no CSV export; `generate_pdf` runs synchronously.

### Requirements scoreboard

| Req | Description | Status |
|---|---|---|
| FR-01 | ≥5 input formats | ✅ 8 formats + DB + REST (OCR deferred) |
| FR-02 | Goal conditions **all** computations, not post-hoc | ✅ **conditioned in plan AND execution** |
| FR-03 | Every recommendation carries a RAG citation | ✅ |
| FR-04 | Expertise visibly alters output language | ✅ **3 of 3 registers (11 Aug)**, deterministic |
| FR-05 | End-to-end < 60s for <100k rows | ✅ **benchmarked 11 Aug: max 2.03s over 25 runs** (`evaluation/results.md`) |
| FR-06 | Every agent step logged/inspectable | ✅ (but not in the PDF) |
| NFR-01 | Horizontal scaling via containerization | 🟡 Docker ✅, K8s placeholder |
| NFR-02 | Incremental vector indexing | ❌ unverified |
| NFR-03 | LLM retry with exponential backoff | ❌ N/A until an LLM exists |
| NFR-04 | Files sandboxed + deleted post-session | ❌ |

---

## 6. Complete gap inventory

**Cross-cutting (highest leverage)**
1. **LLM backbone stubbed.** `openai_api_key` / `anthropic_api_key` empty in `core/config.py`;
   `langchain-openai` installed, `anthropic` not; **no `.messages.create` / `.chat.completions`
   anywhere**. Caps Objectives 3 and 6 and makes NFR-03 moot.
2. **No live/async layer.** No WebSocket → no live dashboard. No Celery/Redis → synchronous runs.
3. **No evaluation harness.** Objective 7. Zero code. Was blocked on FR-02 executing; now unblocked.

**Per module:** M1 OCR, DB/REST endpoint · M2 model-driven loop · M3 time-series decomposition,
supervised importance · M4 four real chart builders, violin, line, choropleth · M5 KB size,
`run_memory`, incremental indexing · M6 **SHAP/LIME**, 3rd register, LLM synthesis · M7 live
dashboard, wizard · M8 WebSocket, Celery/Redis, rate limiting, NFR-04 sandboxing ·
**Infra** K8s manifests, Prometheus/Grafana, LangSmith/Langfuse, MinIO/S3 — all proposed, all absent.

---

## 7. THE PLAN — prioritized for a 14-day runway

Ordered by *(marks at risk) ÷ (effort)*. **Do Phase 1 before anything else.** Everything in Phase 1
is a named deliverable in the proposal with zero or near-zero code today.

### PHASE 1 — Must-have for submission (Days 1–5, i.e. Aug 11–15)

#### 1.1 Empirical evaluation harness — Objective 7 · ✅ **DONE 11 Aug 2026**

**Built and run.** `evaluation/` (datasets · baseline · harness · metrics · report) + 91 tests in
`tests/evaluation/`. `PYTHONPATH=. python -m evaluation.harness` writes `evaluation/results.json`
and `evaluation/results.md`. Headline numbers from the 25-run sweep (5 datasets × 5 goals):

| Metric | MAGE | AutoEDA baseline |
|---|---|---|
| **Cross-goal computation divergence** | **0.940** | **0.000** |
| Cross-goal chart divergence | 0.942 | 0.000 |
| Task-relevant precision | 1.000 | 0.400 |
| Task-relevant recall | 0.601 | 0.539 |
| Task-relevant F1 | 0.749 | 0.458 |
| Citation coverage (FR-03) | 1.000 | — |
| Goal classification accuracy | 1.000 | — |
| Max runtime (FR-05, <60s) | **2.03s PASS** | — |

*(Figures re-run after §1.2 added SHAP — see that section for why divergence moved 0.950 → 0.940
and why recall now beats the baseline instead of tying it.)*

Notes for the write-up:
- The baseline is a **pinned declarative spec**, validated as *exact* against a live
  ydata-profiling 4.18.4 run on all 5 datasets — see `evaluation/BASELINE_VALIDATION.md`.
  ydata-profiling is deliberately **not** a project dependency: it forces `pandas<3`, which
  downgrades the pinned stack and breaks `test_ingestion_agent`. `--live-baseline` re-runs the
  validation when it is installed, and degrades gracefully when it isn't.
- **Read precision and recall together.** The baseline matches MAGE's recall by brute force
  (running everything); its low precision is what that costs. F1 is the fair single number.
- MAGE's precision of 1.000 is ceilinged — it shows the planner emits nothing irrelevant, not that
  the pipeline is optimal. Say so rather than presenting it as a quality score.
- Timings exclude a discarded warm-up run. The embedding model loads lazily (~15s, once per
  process); without the warm-up that one-off cost lands on whichever run goes first and gets
  reported as the FR-05 worst case, overstating it by ~50×.

*Original scoping notes, retained for reference:*


*Why first:* it is the project's entire validation claim, it is scheduled to start today, it has no
code, and it needs **no new infrastructure** — pure Python over machinery that already works.

Create `evaluation/`:
```
evaluation/
├── __init__.py
├── datasets.py      # 2–3 fixed public datasets (or generated); pinned, deterministic
├── baseline.py      # ydata-profiling run → extract the set of computations it performs
├── harness.py       # run N goals × M datasets through OrchestratorAgent, collect results
├── metrics.py       # the lift metrics (below)
└── report.py        # emit results.json + a markdown/LaTeX table for the report
```
**Design (this is the experiment):** hold the dataset constant, vary only the goal — the
same-dataset/different-goal design. For each (dataset, goal) pair record `computations_run`, the
chart set, the recommendations, and the cited sources.

Metrics to implement — keep them simple and defensible:
- **Task-relevant computation precision:** fraction of computations run that appear in a
  hand-labelled "relevant for this task type" set. Baseline scores the same way against its fixed set.
- **Divergence across goals:** Jaccard distance between `computations_run` for two goals on the same
  dataset. Expect ~0 for the baseline (identical every time) and clearly >0 for MAGE. **This single
  number is the headline result.**
- **Chart-set divergence:** same Jaccard measure over chart types.
- **Citation coverage:** % of recommendations carrying a source (should be 100% — proves FR-03).
- **Wall-clock runtime** per run → **this also closes FR-05**, so do it here rather than separately.

**Acceptance:** `python -m evaluation.harness` writes `evaluation/results.json` plus a markdown table
you can paste into the report; the table shows MAGE diverging across goals and the baseline not.

#### 1.2 SHAP feature attribution — Objective 5 / M6 · ✅ **DONE 11 Aug 2026**

**Real SHAP, not the permutation fallback.** `shap.TreeExplainer` over a small `GradientBoosting`
model fitted to the detected target. 34 tests in `tests/agents/test_feature_attribution.py`;
suite now **363 green**.

- Strictly opt-in on a new `shap_attribution` token, added to the classification and regression
  entries in `planner.py` — follows the existing pattern exactly, so it does **not** run for
  clustering / anomaly / reporting goals, and the unconditioned default profile is untouched.
  Tests assert both the presence and the absence.
- New `feature_attribution` key on the mining result. It reuses the `feature_importance` *render*
  type so the frontend and ReportLab exporter both draw it with no changes, but carries a distinct
  title naming the target and method so it is never confused with the unsupervised PCA ranking.
- Verified end-to-end through HTTP: run → SHAP → chart spec → PDF (`%PDF`, renders).

**Two honesty features built into the output — keep them, they are defensible in the viva:**
- `method` names what actually ran. If SHAP is unavailable or errors, it falls back to
  `sklearn.inspection.permutation_importance` and **says so** rather than presenting one method's
  numbers under the other's name. The report should quote this field, not assume SHAP.
- `model_score` is the fitted model's train score. Attributions from a model that cannot predict
  the target are not meaningful; this is what lets a reader judge that. The generated pattern text
  quotes both, e.g. *"'customer_age' contributes most to predicting 'churned' (53% of total
  attribution, via shap.TreeExplainer, model score 0.82)."*

**Sanity-checked against known ground truth, not just for well-formedness:** on the synthetic set
SHAP ranks `customer_age` top for churn (the generator ties churn to the age-defined segment), and
on sklearn's diabetes it ranks `s5`/`bmi` top (the known dominant features). Tests plant a driver
and a pure-noise feature and assert the driver wins by >3×.

**Dependency:** `shap>=0.45.0` added to `backend/pyproject.toml`. It pulls `numba`, which pins
`numpy<2.5`, so the stack sits at **numpy 2.4.6** — pandas 3.0.3 and scipy 1.18.0 are untouched and
all 363 tests pass. *Always `pip install --dry-run` first here* — that is how this was caught before
it broke anything, after ydata-profiling silently downgraded pandas earlier in the day.

⚠️ **Evaluation ground truth was edited as part of this change.** `shap_attribution` was added to
the classification/regression relevance sets in `evaluation/metrics.py`. That edit **raises** MAGE's
precision (0.900 → 1.000), so it deserves scrutiny: it is justified because supervised attribution
is standard practice in both workflows and was absent only because the computation did not exist
when the sets were written. There is a dated CHANGE LOG comment above the constant recording this.
The baseline is unaffected — ydata-profiling performs no attribution.

**Updated headline numbers** (`evaluation/results.md`, re-run after this change):
computation divergence **0.940** vs baseline 0.000 · precision **1.000** vs 0.400 ·
recall **0.601** vs 0.539 · F1 **0.749** vs 0.458 · citation coverage 1.000 · max runtime 2.03s.
Divergence dipped 0.950 → 0.940 because classification and regression now share one more
computation — expected, and honest. Recall now **exceeds** the baseline rather than tying it,
because SHAP adds relevant work the baseline structurally cannot do.

*Original scoping notes, retained for reference:*

*Scope it small.* Do **not** attempt SHAP for every model. Add one honest, working path:
- In `MiningAgent`, when a `target_column` exists and the task is classification/regression, fit a
  small `sklearn` tree model, run `shap.TreeExplainer`, and emit per-feature attributions as a new
  output key (e.g. `feature_attribution`).
- Gate it on a new directive token (`shap_attribution`) added to the classification and regression
  entries in `planner.py`, following the existing strictly-opt-in pattern.
- Surface it as a chart spec (reuse the bar builder) and reference it in recommendation text.
- Add `shap` to `backend/pyproject.toml`.
- Tests: attributions present for a classification goal, absent for a clustering goal.

**Fallback if SHAP proves fiddly:** ship permutation importance
(`sklearn.inspection.permutation_importance`) under the same directive and **say so plainly** in the
report — a working target-aware attribution honestly labelled beats a broken SHAP integration. This
also closes the M3 "supervised feature importance" gap.

#### 1.3 WebSocket step streaming — M7 + M8 · ✅ **DONE 11 Aug 2026**

**Built and verified end-to-end.** `WS /analysis/stream`, plus a live agent trail in the UI.
24 tests in `tests/backend/test_stream.py`; suite now **329 green**.

- `OrchestratorAgent.run()` takes `on_step=None`; default path byte-identical, all prior tests
  untouched. Callback exceptions are logged and swallowed — a dead client cannot abort a run.
- `OrchestratorService.run()` now calls the agent via `asyncio.to_thread`. This was **required**,
  not cosmetic: on the event loop the callback fires but nothing can be sent until the run ends,
  so it would batch rather than stream. It also stops one analysis blocking every other request.
- Endpoint authenticates from a `?token=` query param (browsers cannot set headers on a WebSocket)
  applying *exactly* the checks `get_current_user` does — signature, `access` type, live user.
  Tests cover garbage tokens, refresh-tokens-as-access, and valid-signature-unknown-user.
- Datasets go by `dataset_id`, not over the socket: client POSTs `/analysis/ingest` first.
- Step frames omit the heavy `output` payload (hundreds of KB); the full result rides on the
  terminal `complete` frame and is what gets persisted.

**Two bugs found and fixed while building — worth knowing about:**
1. **Connection leak.** `Depends(get_db)` teardown does not run reliably for WebSocket routes;
   every connection that touched the DB orphaned its connection until SQLAlchemy's GC reclaimed it
   (11 `SAWarning`s per test run, zero in the REST tests). Fixed by opening the session explicitly
   with `async with async_session()`. **If you add another WebSocket route, do the same — do not
   use `Depends(get_db)` there.**
2. **~13s cold-start stall.** The embedding model loads lazily on first use, so the first analysis
   after any restart hung on `RecommendationAgent`. Added `rag.embeddings.warm_up()`, kicked off in
   the background from the app lifespan. Measured against a real server: total run **16.4s → 3.8s**.

**Verified streaming for real** (live uvicorn + real websocket client, 4,000-row dataset):
`IngestionAgent t+13.7ms · MiningAgent t+2.0s · VisualizationAgent t+2.0s ·
RecommendationAgent t+3.7s · complete t+3.8s` — first frame arrives at 13.7 ms of a 3.8 s run,
so it is genuinely incremental, not batched.

*Frontend note:* this repo runs **Next.js 16.2.10** — read `frontend/node_modules/next/dist/docs/`
before touching frontend code, per `frontend/AGENTS.md`. Two live deprecations spotted:
`middleware.ts` → `proxy.ts` (still builds, renamed in v16), and React 19's `FormEvent` →
`SubmitEvent` (fixed in the file I touched; other files may still use it).

*Original scoping notes, retained for reference:*

The data already exists and is already correct; this is plumbing.
- `OrchestratorAgent.run()` gains an optional `on_step: Callable[[dict], None] | None = None`,
  invoked after each step is appended. Default `None` → **behaviour and all existing tests unchanged.**
- New `@router.websocket("/analysis/stream")` in `backend/routers/analysis.py`: accept, authenticate
  from a query-param token, run the pipeline with `on_step` pushing each step as JSON, send a final
  `{"type":"complete", ...}`, close.
- Frontend: a `useAnalysisStream` hook + a live step list on the dashboard — agent name, reasoning,
  observation, status, latency, appearing as they arrive.
- Test: connect, assert 4 step messages arrive in order followed by a completion message.

*Demo payoff:* run two different goals side by side and the audience **watches** different
computations appear. That is a far stronger demonstration of the thesis than diffing JSON.

#### 1.4 Third expertise register — FR-04 / M6 · ✅ **DONE 11 Aug 2026**

**FR-04 closed. Phase 1 complete.** 22 tests in `tests/agents/test_expertise_registers.py`;
suite now **385 green**.

Three registers on every recommendation, mapped in `_REGISTER_BY_EXPERTISE` (`orchestrator.py`):

| Expertise value | Spec audience | Field | What it gives you |
|---|---|---|---|
| `beginner` | Beginner | `text_plain` | Plain language, no jargon, 2 sentences |
| `intermediate` | **Analyst** *(new)* | `text_analyst` | The finding + 4 sentences of reasoning, **attributed inline** |
| `expert` | Data Scientist | `text_technical` | Full excerpt, markdown intact, finding bolded |

Verified over HTTP end-to-end — same dataset, same goal, three levels:
371 / 1062 / 1101 chars, all three renderings distinct. Frontend labels in the analysis form and
settings now read Beginner / **Analyst** / **Data Scientist** to match the spec; the stored enum
values are unchanged (they are the API contract and persist on the user record).

**Design note worth defending in the viva.** The first implementation separated the registers by
*truncation length alone* — and that quietly failed. Several knowledge-base chunks are a single
sentence, and against those a length rule collapses plain and analyst onto the same string, so
FR-04 would have broken for exactly the recommendations whose source is shortest. The analyst
register therefore differs **structurally**: it names the methodology inline ("Per Clustering
Method Selection: …"). That is not decoration — knowing *which* documented procedure is being
invoked is what lets an analyst check it — and it makes the distinction hold for every chunk
length. Registers now fall back down the chain (`analyst → technical → insight`), so runs
persisted before `text_analyst` existed still render prose.

⚠️ **Adjacent bug found, deliberately NOT fixed here — worth doing in §2.1.**
`_strip_markdown_structure` discards markdown **table rows wholesale**. Any KB document whose
substance *is* a table contributes only its heading to a recommendation — the goal→chart decision
table in `distribution_profiling.md` is the clearest casualty, and it is one of the most useful
artefacts in the corpus. This affects **all three registers**, not just the new one, and is also
why plain and technical can still coincide on those chunks. Preserving tables for the technical
register would fix both, but it changes what the ReportLab exporter has to render, so it belongs
with the knowledge-base work. `test_known_gap_markdown_tables_are_dropped_from_every_register`
pins the current behaviour so it stays visible — **that test is designed to fail once tables are
preserved; delete it then.**

*Original scoping notes, retained for reference:*

Currently `beginner → plain`, everything else `→ technical`. The spec says
**Beginner / Analyst / Data Scientist**. Add the middle register in `recommendation_agent.py`
(plain-language finding + the statistic that supports it, without full technical framing), wire the
3-way selector through, and test that all three produce **measurably different** text.
Small change, closes a functional requirement outright.

### PHASE 2 — Strong value, do if Phase 1 lands early (Days 6–9, Aug 16–19)

#### 2.1 Expand the knowledge base — M5
Cheapest quality win in the project: authoring markdown, **zero code changes**, loader picks files up
automatically. Target ~12–15 docs. Highest-value additions, matched to what M3 actually computes:
classification model selection, regression diagnostics, class imbalance handling, feature
engineering, time-series basics, PCA/dimensionality reduction, data-leakage pitfalls,
train/test methodology. Use the existing frontmatter format (`title`, `doc_type`, `section`).
*Measure the effect in the evaluation harness — "citation quality improved with corpus size" is a
result you can report.*

#### 2.2 Put the agent step log in the PDF — FR-06
FR-06 claims a complete inspectable explainability trail; the PDF (the artefact an examiner actually
reads) omits it. Add a "Agent Execution Trail" section to `generate_pdf` — a table of
agent / reasoning / observation / status / latency. Low effort, directly strengthens a core FR, and
lands in the deliverable that gets graded.

#### 2.3 `run_memory` table — Objective 4's missing half
The proposal's "improves with use" benefit. Minimum viable version: a `run_memory` table storing
(goal text, task_type, dataset fingerprint, embedding, key findings). On a new run, retrieve the
top-k most similar prior runs and include them as additional grounding context alongside the KB.
Even a modest implementation converts a ❌ into a 🟡 on a stated objective.

#### 2.4 Real builders for the four approximated charts — M4
`grouped_bar`, `box_by_class`, `pairplot`, `highlighted_scatter` currently map to stand-ins. Writing
genuine builders removes a caveat you would otherwise have to disclose, and improves the visual
quality of the demo.

### PHASE 3 — Only with time to spare (Days 10–12, Aug 20–22)

- **Wire the LLM** (`claude-opus-5` or GPT-4o behind a swappable interface, per the proposal). Big
  win for Objectives 3 and 6, and makes NFR-03 meaningful — but it introduces non-determinism,
  cost, and network dependence late in the schedule. **Do not start this if the evaluation harness
  isn't finished**, and re-run the harness afterwards if you do.
- **Celery + Redis** (NFR + M8). Real async. Note the WebSocket already delivers the *visible*
  benefit, so this is architectural completeness rather than user-facing gain.
- **NFR-04 session sandboxing** — TTL-based dataset cleanup. Small, closes an NFR.
- **Rate limiting** — `slowapi` middleware. ~20 lines, closes an M8 bullet.
- **Time-series decomposition** (M3) and **line charts** (M4) — pair naturally; only worth it if a
  temporal dataset is in the demo.
- **OCR** — explicitly deferred. Leave deferred; say so in the report.
- **K8s manifests / Prometheus / MinIO** — do not attempt. Document as future work.

### PHASE 4 — Integration, benchmarking, report (Days 12–14, Aug 22–25)

1. `PYTHONPATH=. pytest tests/ -v` — all green, no skips.
2. `docker compose up --build` from clean — verify the whole stack comes up.
3. Full manual E2E: signup → Google OAuth → upload → each of the 5 goal types → live streaming →
   report view → all 3 exports.
4. **FR-05 benchmark**: a <100k-row dataset, timed, recorded. (Free if §1.1 captured runtime.)
5. Freeze the code. Run the evaluation harness one final time and use *those* numbers in the report.
6. Write up: results tables from `evaluation/results.json`, screenshots, architecture diagram, and an
   explicit **limitations** section (see §8).

---

## 8. Honesty guide — what to claim and what to concede

Being straight about limitations is worth more marks than overclaiming, and every item below is
something an examiner reading the code could find.

**Claim confidently**
- Goal-conditioning is real and executes: different goals → different `computations_run`, proven by
  an automated test, not a demo anecdote.
- Every recommendation carries a retrievable citation; the citation chain is exportable as BibTeX.
- Full Reason/Act/Observe trail with per-step latency.
- Runs entirely locally with **no API key** — reproducible, deterministic, no data leaves the machine.
- 8 input formats + live DB + REST ingestion.
- 385 tests (all green, 11 Aug), including cross-user isolation on every export path and 91
  covering the evaluation harness.
- The goal-conditioning claim is **measured, not asserted**: 0.950 cross-goal divergence vs. the
  baseline's 0.000, against a baseline verified exact against the real ydata-profiling.

**Concede plainly, before you are asked**
- The "ReAct loop" is a fixed 4-step for-loop, not model-driven reason-act-observe. `MAX_REACT_STEPS`
  is therefore dead code, not an active guard.
- No LLM is wired. Classification is keyword + embedding rules; recommendation text is template
  paraphrase. This is why output is deterministic — a genuine strength for reproducibility, but not
  the LLM reasoning the proposal envisages.
- Four chart types are approximations, not dedicated builders.
- The knowledge base is small; retrieval degrades outside its 5 topics.
- `run_memory` does not exist, so the "improves with use" benefit is currently unrealized.
- OCR was deliberately deferred.
- Analysis is synchronous; no job queue.

**If asked "how is this better than ChatGPT?"** — do not claim better analysis; a frontier model
will out-reason MAGE on any single dataset. The defensible answer: *different problem.* MAGE is
verifiable (every claim cites a retrievable source), reproducible (deterministic — same input,
identical output), auditable (full step log), local (data never leaves the machine), free per run,
and emits structured output that feeds a dashboard/PDF/BibTeX pipeline. ChatGPT is a brilliant
generalist that improvises; MAGE follows an auditable procedure. Also redirect: the proposal's
baseline is **ydata-profiling**, not ChatGPT — and against that baseline the goal-conditioning claim
is measurable and clean.

---

## 9. Uncommitted state — deal with this first

```
M  frontend/app/lib/auth-context.tsx      ← real bug fix, NOT committed
?? MAGE_Project_Report.pdf                ← generated artefact
?? scripts/generate_project_report.py     ← untracked tooling
```

`auth-context.tsx` adds `clearSession()` calls before both `/signin` redirects. It fixes a genuine
**infinite redirect loop**: a stale `mage_token` cookie made `middleware.ts` bounce `/signin` →
`/dashboard` forever, showing an endless loading spinner. **Commit this.** Suggested message:

```
fix(auth): clear stale session cookie before redirecting to signin

A stale mage_token cookie made the middleware bounce /signin back to
/dashboard, producing an infinite redirect loop and a permanent loading
spinner. Clear both the localStorage token and the cookie before every
redirect to /signin.
```

Decide on `scripts/generate_project_report.py` — it builds the project write-up PDF (separate from
the per-analysis export path). Either commit it (useful, reproducible) or add it to `.gitignore`
along with `MAGE_Project_Report.pdf`. Don't leave it dangling through submission.

Branch note: work is on `feat/m2-goal-orchestrator`, well past M2 scope. Consider merging to `main`
before the final push so the submitted default branch is the real state of the project.

---

## 10. Conventions and gotchas

- **`PYTHONPATH=.` from repo root** for both uvicorn and pytest — `agents/`, `rag/`, and
  `data_pipeline/` are root-level packages. `orchestrator_service.py` even does a `sys.path.insert`
  to survive being launched from `backend/`.
- **All 385 tests must stay green.** Any change to `MiningAgent` or `VisualizationAgent` must keep the
  no-directives default profile intact — that backward-compatibility path is what several older
  tests rely on.
- **Docstring style:** module header with a `────` underline, then a short prose description of
  responsibilities. Match it.
- **pandas only** in `data_pipeline` — Spark and Dask paths raise `NotImplementedError` on purpose.
- **Import `settings`**, never re-instantiate `Settings()`.
- **Chroma is embedded** (`PersistentClient`) — there is no ChromaDB container despite what older
  README diagrams show. `docker-compose.yml` is the truth: backend, frontend, postgres.
- **Docker healthchecks use `python`/`node`, not `curl`** — curl isn't in the slim/alpine runtime
  images. Don't "fix" them to curl.
- **New computations** follow the strictly-opt-in pattern: add the token to `planner.py`, then gate on
  `"token" in computations` in `mining_agent.py` (bypassing `wants()`), then add a test asserting it
  runs for the right goal and **not** for the wrong one.
- **`MIN_CONFIDENCE = 0.15`** in `recommendation_agent.py` is the retrieval floor. Lowering it
  increases coverage but weakens grounding — if you change it, justify it in the report.
- **`_MAX_CHARTS = 6`** caps PDF length in `export_service.py`.

---

## 11. Key file quick-reference

| Need to change… | Go to |
|---|---|
| Which computations a goal runs | `agents/planner.py:44-71` |
| How goals are classified | `agents/goal_classifier.py:46-108` (keywords + prototypes) |
| Add a new statistical computation | `agents/mining_agent.py:100-142` (gating) + a `_compute_*` method |
| Add a chart type | `agents/visualization_agent.py:105-146` (routing) + a `_*_spec` builder |
| Recommendation text / registers | `agents/recommendation_agent.py:105-140` |
| Retrieval threshold | `agents/recommendation_agent.py:54` (`MIN_CONFIDENCE`) |
| Add an API endpoint | `backend/routers/analysis.py` |
| PDF report layout | `backend/services/export_service.py:272+` (`generate_pdf`) |
| PDF chart rendering | `backend/services/export_service.py:217-270` (`_spec_to_flowable`) |
| Settings / env vars | `backend/core/config.py` |
| DB schema | `backend/models/` (3 files) |
| API contracts / types | `backend/schemas/analysis.py` |
| Frontend fetch layer | `frontend/app/lib/api.ts` |
| Report view + exports UI | `frontend/app/dashboard/analysis/[id]/page.tsx` |
| Knowledge base content | `data/knowledge_base/*.md` |

---

## 12. Prompt to open the new chat with

> I'm continuing work on MAGE, a multi-agent goal-conditioned EDA system at `/Users/22sarthak/MAGE`
> (branch `feat/m2-goal-orchestrator`). Read `docs/CONTINUATION_PLAN.md` first — it's a full context
> handoff with the architecture, current state, known gaps, and a prioritized plan. Also read
> `docs/PROJECT_PROGRESS.md` for the earlier audit.
>
> Submission is 25 Aug 2026, so this is triage. Start with Phase 1 item 1.1, the empirical evaluation
> harness (Objective 7) — it's the project's validation claim and has no code yet. Before writing
> code, confirm the metric design with me.
>
> Constraints: `PYTHONPATH=. pytest tests/ -v` must stay green (385 tests); keep the no-directives
> default profile in MiningAgent/VisualizationAgent intact; match existing docstring style.

---

*End of handoff. Every status claim here was verified against the code on 11 Aug 2026 — not copied
forward from the older progress doc.*
