# MAGE — Project Progress & Integration Checklist

> **Purpose:** Track what's built vs. pending against the Project Proposal & Presentation.
> **Last audited:** 2026-07-24 (branch `feat/m2-goal-orchestrator`, post-merge of `integration/combined-features`)
> **Legend:** ✅ done · 🟡 partial · ❌ not started · ⚠️ built but not truly wired

---

## 0. TL;DR

The **happy path works end-to-end**: upload CSV → goal classified → mined → visualized → RAG-grounded recommendations → export (PDF/JSON/BibTeX). All 8 modules have a working vertical slice.

**The three gaps that matter most for matching the docs:**
1. **⚠️ Goal-conditioning is not actually executed.** The planner produces task-specific directives, but MiningAgent/VisualizationAgent ignore them and run a fixed pipeline — so two different goals on the same dataset currently produce (nearly) identical mining + charts. This is the project's *core evaluated hypothesis* (FR-02, same-dataset/different-goal).
2. **⚠️ LLM backbone is stubbed.** Classification & recommendations are deterministic; no provider is wired. RAG embeddings run locally (no key needed).
3. **❌ No live/async layer.** Analysis runs synchronously; no WebSocket streaming, no Celery/Redis job queue.

---

## 1. Module-by-Module Status

### M1 — Data Ingestion & Normalization
- [x] ✅ CSV, TSV, JSON, Parquet, Excel (XLSX/XLS) — `agents/ingestion_agent.py`
- [x] ✅ Header validation + normalization to canonical schema
- [x] ✅ File upload endpoint + dataset persistence — `backend/routers/analysis.py`, `backend/services/dataset_service.py`
- [x] ✅ **PDF table extraction** (pdfplumber) — `data_pipeline/ingestion.py:_load_pdf` *(done 2026-07-24)*
- [x] ✅ **Database URI ingestion** (SQLAlchemy, any dialect; table/query/`#fragment` selection) — `load_from_database` *(done 2026-07-24)*
- [x] ✅ **REST/HTTP ingestion** (httpx; JSON/CSV/TSV/Parquet/Excel by content-type or hint) — `load_from_url` *(done 2026-07-24)*
- [x] ✅ Auto source-type routing (`classify_source` → file/url/database) + agent routing — `IngestionAgent._load_by_type`
- [ ] ❌ **OCR for scanned documents** (pytesseract / OCR.space) — *deferred by decision; only remaining M1 item*
- [ ] 🟡 **API surface for DB/REST sources** — engine + agent support them; a dedicated upload-alternative endpoint is not yet exposed in `routers/analysis.py`

### M2 — Goal Conditioning & Orchestrator
- [x] ✅ Natural-language goal → task-type classification — `agents/goal_classifier.py`
- [x] ✅ Rule-based + local-embedding fallback classifier
- [x] ✅ Conditional plan builder (per-task directives) — `agents/planner.py`
- [x] ✅ ReAct-style step loop with hard step cap (`MAX_REACT_STEPS=10`) — `agents/orchestrator.py`
- [x] ✅ Inspectable Reason-Act-Observe step log (explainability trail, FR-06)
- [x] ✅ Target-column refinement from ingested schema
- [x] ✅ **Directives are now honored downstream** — MiningAgent & VisualizationAgent condition their computations/charts on the plan *(done 2026-07-24; see §3-A)*
- [ ] ⚠️ **"ReAct loop" is a fixed for-loop over the plan**, not a model-driven reason-act-observe cycle — *acceptable for now, but note for the report's honesty*

### M3 — Data Mining Agent
- [x] ✅ Descriptive statistics (numeric + categorical) — `agents/mining_agent.py`
- [x] ✅ Data-quality metrics (completeness / uniqueness / missing)
- [x] ✅ Pearson correlation matrix
- [x] ✅ IQR-based outlier detection
- [x] ✅ PCA-loading feature importance (unsupervised)
- [x] ✅ KMeans clustering (silhouette-selected k)
- [x] ✅ **Isolation Forest** outlier detection — conditioned on anomaly goals *(done 2026-07-24)*
- [x] ✅ **DBSCAN** clustering (k-distance eps heuristic) — conditioned on clustering goals *(done 2026-07-24)*
- [x] ✅ **Class balance** (target distribution / imbalance ratio) — classification goals *(done 2026-07-24)*
- [x] ✅ **Linearity check** (feature↔target Pearson) — regression goals *(done 2026-07-24)*
- [ ] ❌ **Time-series decomposition** (statsmodels seasonal_decompose) — *docs, absent*
- [ ] 🟡 **Supervised feature importance** (target-aware) — partial: linearity ranks features vs target; PCA still the unsupervised default

### M4 — Visualization Agent
- [x] ✅ Correlation heatmap — `agents/visualization_agent.py`
- [x] ✅ Feature-importance bar
- [x] ✅ Cluster scatter (2D PCA)
- [x] ✅ Histogram, boxplot, categorical bar
- [x] ✅ **Chart set now conditioned on planner `charts` directive** — task types yield different chart sets *(done 2026-07-24; see §3-A)*
- [x] ✅ **Scatter** (two most-correlated numerics) + **missingness matrix** added *(done 2026-07-24)*
- [ ] 🟡 **grouped_bar / box_by_class / pairplot / highlighted_scatter** — routed but approximated (mapped to bar/box/cluster_scatter/scatter); dedicated builders still TODO
- [ ] ❌ **Violin plots** — *docs, absent*
- [ ] ❌ **Line charts** (temporal trends) — *docs, absent*
- [ ] ❌ **Choropleths** (geospatial) — *docs, absent*

### M5 — RAG Pipeline
- [x] ✅ Local sentence-transformers embeddings (`all-MiniLM-L6-v2`, 384-dim) — `rag/embeddings.py`
- [x] ✅ FAISS + Chroma vector store — `rag/vector_store.py`
- [x] ✅ Knowledge-base loader + corpus (5 methodology docs) — `rag/knowledge_loader.py`, `data/knowledge_base/`
- [x] ✅ Top-k semantic retrieval wired into recommendations
- [ ] 🟡 **Knowledge base is small (5 docs)** — expand curated corpus for stronger grounding
- [ ] ❌ **Run-memory store** (prior-run retrieval — "improves with use") — *docs `run_memory` table, not implemented*
- [ ] ❌ **Incremental indexing across sessions** (NFR-02) — *verify/implement*

### M6 — Recommendation & Explainability
- [x] ✅ RAG-grounded recommendations with source citation — `agents/recommendation_agent.py`
- [x] ✅ Dual-register output (`text_technical` / `text_plain`)
- [x] ✅ Confidence threshold + ranking
- [x] ✅ Conversational Q&A agent — `agents/qa_agent.py`
- [ ] 🟡 **Only 2 registers** (beginner→plain, else→technical); docs specify **3** (Beginner / Analyst / Data Scientist)
- [ ] ⚠️ **Text is deterministic paraphrase**, not LLM synthesis (see §3, item B)
- [ ] ❌ **SHAP / LIME feature attribution** — *explicit FR, zero code anywhere*

### M7 — Frontend & UX
- [x] ✅ Landing, sign-in/up, dashboard, datasets, settings, knowledge pages — `frontend/app/`
- [x] ✅ Google OAuth callback + route protection
- [x] ✅ Markdown rendering, charts, analysis detail view
- [x] ✅ Export to PDF from report view
- [ ] ❌ **Live analysis dashboard with real-time WebSocket agent-step streaming** — *headline UI feature, absent (see §3, item C)*
- [ ] 🟡 **Expertise selector**: confirm 3-way (Beginner/Analyst/Data Scientist) drives output register end-to-end

### M8 — API Layer & Authentication
- [x] ✅ FastAPI REST endpoints (analysis, auth, oauth, health) — `backend/routers/`
- [x] ✅ JWT auth (python-jose) + Google OAuth (authlib)
- [x] ✅ Postgres persistence (users, datasets, analysis_runs)
- [x] ✅ Pydantic validation, expertise-level plumbed through
- [x] ✅ Export service: PDF report, JSON, BibTeX citation bundle — `backend/services/export_service.py`
- [ ] ❌ **WebSocket endpoint** for live agent updates — *absent*
- [ ] ❌ **Celery + Redis job queue** (async analysis) — *absent; runs synchronously*
- [ ] ❌ **Rate limiting** — *docs, verify/absent*
- [ ] ❌ **Session-scoped file sandboxing + delete-on-session-end** (NFR-04) — *verify current retention behavior*

---

## 2. Functional / Non-Functional Requirements

| Req | Description | Status |
|-----|-------------|--------|
| FR-01 | Accept ≥5 formats (CSV, Excel, JSON, PDF, DB URI) | ✅ CSV/TSV/JSON/Parquet/Excel/PDF + DB URI + REST; OCR deferred |
| FR-02 | Goal conditions **all computations**, not post-hoc | ✅ conditioned in plan **and execution** (mining + viz honor directives) |
| FR-03 | Every recommendation carries a RAG citation | ✅ |
| FR-04 | Expertise level visibly alters output language | 🟡 2 of 3 registers, deterministic |
| FR-05 | End-to-end < 60s for <100k rows | 🟡 likely OK (local, no LLM) — **not benchmarked** |
| FR-06 | Every agent step logged/inspectable | ✅ |
| NFR-01 | Horizontal scaling via containerization | 🟡 Docker yes; K8s stretch |
| NFR-02 | Incremental vector indexing (no full rebuild) | ❌ verify |
| NFR-03 | LLM API retry w/ exponential backoff | ❌ n/a until LLM wired |
| NFR-04 | Files sandboxed + deleted post-session | ❌ verify |

---

## 3. Cross-Cutting Gaps (highest leverage)

### A. ✅ Goal-conditioning now executes (RESOLVED 2026-07-24)
`planner.py` emits per-task `computations` (e.g. `dbscan`, `isolation_forest`, `class_balance`) and `charts` (e.g. `grouped_bar`, `pairplot`). These are now **honored at execution time**:
- `MiningAgent.run()` selects computations from `directives["computations"]` — clustering goals run KMeans + DBSCAN, anomaly goals run Isolation Forest + IQR, classification runs class-balance, regression runs linearity, etc. New computations (Isolation Forest, DBSCAN, class balance, linearity) added.
- `VisualizationAgent.run()` builds the chart types named in `directives["charts"]`, falling back to the default set only when none are feasible.
- **Backward compatible:** with no directives (standalone use), both agents run the full default profile — existing tests unchanged.
- **Verified:** `tests/agents/test_goal_conditioning.py` proves two different goals on the same dataset produce different `computations_run` and different chart sets. This makes the same-dataset/different-goal thesis demonstrable and unblocks the evaluation harness (§3-D).

### B. ⚠️ LLM backbone stubbed
- Classification & recommendation text are deterministic/template-based.
- `config.py` has empty `openai_api_key` / `anthropic_api_key`; `langchain-openai` installed, `anthropic` not.
- No `.messages.create` / `.chat.completions` call anywhere.
- **When wired:** goal classification gets model reasoning; recommendations get real synthesis + true per-audience rewriting.

### C. ❌ No live/async execution layer
- `run_analysis` awaits the orchestrator **synchronously**; response returns only when the whole pipeline finishes.
- No WebSocket → the "Live Analysis Dashboard" (reason/act/observe streaming) can't exist yet.
- No Celery/Redis → no async job management.

### D. ❌ Empirical evaluation harness
- Objective 7 / Plan of Work: same-dataset/different-goal vs. `ydata-profiling` baseline, measuring task-relevant output lift.
- **No benchmark/evaluation code exists.** Needed for the project's validation claim — and blocked by gap A (nothing to measure until conditioning actually changes output).

### E. Data model vs. ER diagram
- Implemented tables: `users`, `datasets`, `analysis_runs` (3).
- ER diagram (slides) has ~11 normalized tables (`findings`, `recommendations`, `citations`, `agent_steps`, `visualizations`, `run_memory`, `sessions`, `kb_documents`).
- Currently these are collapsed into JSON blobs on `analysis_runs`. Fine for now; normalize if the report/eval needs relational queries.

---

## 4. Suggested Priority Order

1. ~~**Execute goal-conditioning** in MiningAgent + VisualizationAgent~~ — ✅ **DONE 2026-07-24** (unlocks the core claim *and* the evaluation).
2. **Empirical evaluation harness** vs. ydata-profiling — *now unblocked by #1*; next-highest value for the report's validation claim.
3. **LLM provider wiring** (Gensara — custom `/api/chat`), opt-in with deterministic fallback. *Deferred by decision in favour of core-thesis work.*
4. **WebSocket live streaming** + **Celery/Redis** async jobs (the missing headline UX).
5. **Remaining M3/M4 gaps**: time-series decomposition; violin/line/choropleth + dedicated grouped-bar/box_by_class builders.
6. **SHAP/LIME** explainability (explicit FR with no code).
7. ~~**Extra ingestion**: PDF tables, DB URI, REST~~ — ✅ **DONE 2026-07-24** (OCR still deferred).
8. **3rd output register** (Analyst) + verify NFR-04 file sandboxing/retention.

---

## 5. Test Coverage (already in place)
`tests/agents/` (classifier, ingestion, mining, orchestrator, planner, qa, recommendation, visualization) ·
`tests/backend/` (analysis persistence, auth, export, health) ·
`tests/rag/` (embeddings, knowledge_loader, vector_store). Good baseline; extend as gaps above are closed.

---

## 6. Databases & Data Stores

| Store | Type | Used for | Status |
|-------|------|----------|--------|
| **PostgreSQL** | Relational (async) | Users, datasets, analysis runs | ✅ active |
| **ChromaDB** (default) / FAISS | Vector | RAG knowledge-base embeddings + semantic retrieval | ✅ active (embedded) |
| SQLite | Relational | Dependency-free test fallback only | 🟡 fallback |
| Redis | In-memory / broker | Celery job queue (per docs) | ❌ not implemented |

### PostgreSQL — primary relational DB
- **Engine:** async SQLAlchemy over `asyncpg` — `backend/core/database.py`
- **Connection:** `postgresql+asyncpg://mage:mage@localhost:5432/mage` (docker-compose `postgres` service)
- **Purpose:** all application/account state, persisted per user across restarts.
- **Tables (3 today):**
  - `users` — accounts, hashed passwords, role, `default_expertise_level` (`backend/models/user.py`)
  - `datasets` — uploaded-dataset metadata/references (`backend/models/dataset.py`)
  - `analysis_runs` — goal, task_type, status; findings/recommendations/citations stored as **JSON blobs** (`backend/models/analysis_run.py`)
- Tables auto-created at startup via `init_db()` (`Base.metadata.create_all`).
- Browse locally: `pgweb --host=localhost --port=5432 --user=mage --pass=mage --db=mage`
- **Gap:** ER diagram (slides) specifies ~11 normalized tables (`findings`, `citations`, `agent_steps`, `sessions`, `visualizations`, `run_memory`, `kb_documents`); only 3 exist — rest are JSON on `analysis_runs`. See §3-E.

### Vector store — RAG (ChromaDB default, FAISS alternative)
- **Abstraction:** `rag/vector_store.py` — swappable via `VECTOR_STORE_BACKEND` (`chroma` | `faiss`), default `chroma`.
- **Purpose:** stores embeddings of the EDA-methodology knowledge base; powers citation-grounded recommendations.
- **Chroma runs embedded/in-process** (`PersistentClient`) → persisted to `./data/chroma_db/` (`chroma.sqlite3` + collection dir). **No separate Chroma server/container.**
- **FAISS** alternative persists to `./data/faiss_index`.
- Embeddings: local `all-MiniLM-L6-v2` (384-dim) — no API key.
- **Cleanup done (2026-07-24):** removed vestigial `chroma_host`/`chroma_port` from `config.py` (embedded Chroma needs no host/port); `CHROMA_HOST`/`CHROMA_PORT` already dropped from `.env`.

### SQLite — fallback only
- `database.py` still accepts `sqlite+aiosqlite` (with `check_same_thread` handling) for dependency-free test runs. Not used in normal operation now that `DATABASE_URL` targets Postgres.
- (Chroma's internal `chroma.sqlite3` is a separate storage detail of the vector index, unrelated to the app DB.)

### Redis — in docs, not in code
- Proposal lists Redis as the Celery broker. **No Redis usage anywhere** — analysis runs synchronously. Pending, tied to the async/Celery gap (§3-C).
