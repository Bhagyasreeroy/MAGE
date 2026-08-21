# Prompt to open a new session with

Copy everything in the box below.

---

I'm continuing work on MAGE, a multi-agent goal-conditioned EDA system at
`/Users/22sarthak/MAGE`, branch `feat/m2-goal-orchestrator` (HEAD `f8283f2`,
in sync with origin, working tree clean). **Do not touch `main`** — it stays at
`b64ed19` until I say otherwise.

**Read these three files first, and nothing else:**
- `docs/BUILD_LOG_2026-08-21.md` — everything done on 21 Aug, twelve sections, the
  most current picture of the system
- `docs/REMAINING_PLAN_2026-08-18.md` — what's left, costed and sequenced. **Its
  F1, F2, F3 and B1 are all done**; B2, the MUST list and the PROBABLY-NOT
  reasoning still stand
- `docs/LIVE_EVALUATION_2026-08-18.md` — the live browser evaluation. Its one open
  limitation (temporal goal classification) was closed on 21 Aug

Skip `docs/CONTINUATION_PLAN.md` (57 KB, superseded) and `docs/SECTION_B_PLAN.md`
(Tier 1 complete). **Ignore `docs/PROJECT_PROGRESS.md` entirely** — it is the
24 Jul audit and is now actively wrong: it lists WebSocket streaming, SHAP and the
ReAct loop as absent, and all three exist.

**State — take these as given, don't re-derive them:**
- **839 tests pass, 1 skipped.** `tests/backend/` needs Postgres:
  `docker compose up -d postgres`. `duckdb`, `slowapi` and `statsmodels` must be
  installed.
- Evaluation harness stable at **0.836 / 0.920 / 0.883 / 1.000**, re-run four times
  on 21 Aug with every metric identical each time. **Quote 0.836, never 0.940** —
  and give the reason in the same breath (`BUILD_LOG_2026-08-17.md` §7).
- NFR-02/03/04 and M8 rate limiting closed 18 Aug. **NFR-01 is the only unmet
  requirement**: Docker yes, but `infra/k8s/` contains a README and no manifests.
- Gemini is wired and live via `GEMINI_API_KEY` in `.env`, opt-in only.
- The stack is `docker compose` (backend :8000, frontend :3000, postgres :5432).
  Only `./data` is mounted — **source is baked into the image**, so any code change
  needs `docker compose up -d --build backend` before it exists in the browser.

**Conventions I expect you to follow:**
- **Test-driven**: write the test first, watch it fail *for the right reason*, then
  implement. **Vacuous tests are this project's recurring failure** — four caught so
  far, the last two on 21 Aug. If a test guards existing behaviour, prove it by
  breaking the feature and watching that test go red.
- **Log everything** to `docs/BUILD_LOG_<today>.md`, in the style of the existing ones.
- Commit with the reasoning in the message; push to `feat/m2-goal-orchestrator` only.
- Any new dependency goes in `backend/pyproject.toml` **and** must be verified
  importable inside the container.
- Re-run the harness if anything touches `planner.py` computations, goal
  classification, or the chart sets — and record the movement plus its cause.
- **Never edit `TASK_RELEVANT_COMPUTATIONS` in `evaluation/metrics.py`** to make a
  number better. Its own change log explains why that invalidates the metric.
- UI work can be verified in a real browser: Playwright drives the installed Chrome
  via `channel="chrome"`, no download needed. Sign in through the actual form —
  `middleware.ts` reads a `mage_token` cookie, so injecting localStorage just
  bounces to `/signin`.

**My task for this session:** <SAY EXACTLY WHAT YOU WANT HERE>

---

## What's actually left

### 1. B2 — DB / REST ingestion endpoint (M1) · ~3h · no new deps

The last unbuilt feature. `load_from_database` and `load_from_url` already work
with `classify_source` routing; only the HTTP surface is missing. Makes FR-01
demonstrable rather than "supported internally".

⚠️ **The one item with real security risk.** A user-supplied URI fetched by the
*server* is SSRF. Block `localhost` / `127.0.0.1` / `::1` / link-local / private
ranges and `file://`, allow-list schemes, cap size and timeout. A DB URI carries
credentials — never log it, never echo it in an error. Write all of that as tests.
The SQL console sets the bar: eight escape attempts blocked, nothing leaked.

### 2. The RAG guidance prose is still flattened · ~2h

`_strip_markdown_structure` in `agents/recommendation_agent.py` strips headings,
emphasis and tables from retrieved chunks, so a chunk that is a heading plus
bullets renders as run-on prose — *"Outlier Detection IQR (Interquartile Range)
method For a single numeric column…"*. The 21 Aug card work gave the guidance a
place to breathe; the guidance itself is still a paragraph. This is the most
visible remaining polish item, and it is what would make the RAG answer clearly
*better* than the LLM's rather than merely equal.

Its existing tests encode the current flattening on purpose — read them before
changing it.

### 3. Test and code debt, all small

- **Three filter tests hand-write the attribution literal**
  (`tests/agents/test_recommendation_agent.py`). They pass and act as wording
  pins, but they are *named* as filter tests and would fail for a phrasing reason.
- **`LOW_ATTRIBUTION_MODEL_SCORE` is an absolute cutoff**, not one relative to a
  majority-class baseline. 0.3 accuracy over 10 balanced classes is unremarkable;
  the same 0.3 over 2 classes is barely above chance. Documented in-line in
  `agents/orchestrator.py` as a known follow-up.
- **One unexplained suite flake.** A full run once returned `12 failed, 9 errors`
  in agent tests untouched by the work; every one passed in isolation and the next
  run was clean. Likely contention over `data/chroma_db`, which the running
  container also mounts. Unproven. A suite that fails positionally is the expensive
  kind of flake.
- **`docs/PROJECT_PROGRESS.md` should be rewritten or deleted.** It is the most
  misleading file in the repo.
- **`middleware.ts` → `proxy.ts`** is a Next 16 deprecation that still builds. Note
  it, don't chase it.
- **An unused `ArrowRightIcon` import** in the analysis page — a lint warning.

### 4. A palette finding, deliberately not acted on

`SERIES_COLORS` in `frontend/app/components/charts.tsx` fails a chroma-floor check
(all six read as near-grays), and the pair `#6d6875` / `#4a4e69` separates at only
**ΔE 9.9** for normal vision — below the 15 floor. Harmless today because no chart
places those two together, but any future chart that does would be unreadable.
**This is a brand decision, not one to make inside a chart component** — raise it,
don't quietly change it.

### 5. MUST — before submission, not optional

1. **Merge to `main`.** It still sits at `b64ed19` while the branch carries
   everything — **98+ commits**, including the whole of 21 Aug. The submitted
   default branch should be the real state. Nobody else will do this.
2. **Freeze the code**, run the harness one final time, and use *those* numbers
   everywhere.
3. **Regenerate `MAGE_Project_Report.pdf`** via `scripts/generate_project_report.py`
   — it is gitignored, so this is a deliberate step and teammates must run it
   themselves.
4. **`docker compose up --build` from clean** once more after the last change.
5. **Rehearse the divergence answer**: 0.836, not 0.940, *with the reason in the
   same breath*. `BUILD_LOG_2026-08-17.md` §7 has the three measured options.

### 6. Deliberately not doing — the reasoning still holds

Celery + Redis (the WebSocket already delivers the visible benefit), LIME
(re-inflates the image that was cut 9.87 GB → 3.5 GB, and it is *local*
attribution against SHAP's *global* — not interchangeable), choropleths (boundary
files and a projection library for one chart), OCR (needs the `tesseract` system
binary, a stated exclusion reads better than a rushed inclusion), and the 3-view
wizard (pure UI churn in the flow every demo depends on).

---

## What changed on 21 Aug, in one paragraph

Merged Bhagyasree's observe-and-replan ReAct loop after verifying it by mutation
rather than trusting its commit message, and closed the two weaknesses that found.
Fixed a chartless report rendering as blank space; refused analysis runs with no
dataset (which had been returning a full five-recommendation report built from
goal-only retrieval, plus a 500 on an unresolvable `dataset_id`); gave RAG answers
their structure back as cards, so a grounded answer no longer reads worse than the
LLM's; enforced the classifier's dead confidence floor and taught the lexicon
temporal vocabulary; stopped one error message covering two unrelated CSV problems;
and built seasonal decomposition, gated on the data so the harness could not move.
Tests 750 → 839. Harness unchanged throughout, verified rather than assumed.
