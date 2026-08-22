# Prompt to open a new session with

Copy everything in the box below.

---

I'm continuing work on MAGE, a multi-agent goal-conditioned EDA system at
`/Users/22sarthak/MAGE`, branch `feat/m2-goal-orchestrator` (HEAD `4effca5`,
in sync with origin, working tree clean). **Do not touch `main`** — it stays at
`b64ed19` until I say otherwise.

**Read these first, and nothing else:**
- `docs/BUILD_LOG_2026-08-21.md` — the twelve sections of 21 Aug work, still the
  fullest picture of the system's architecture
- `docs/REMAINING_PLAN_2026-08-18.md` — what's left, costed and sequenced. **Its
  F1, F2, F3 and B1 are done**; B2, the MUST list and the PROBABLY-NOT reasoning
  still stand
- `docs/LIVE_EVALUATION_2026-08-18.md` — the live browser evaluation. Its one open
  limitation (temporal goal classification) was closed on 21 Aug
- **The "What's actually left" section below** — it supersedes those three wherever
  they disagree, and carries the 22 Aug work they predate

Skip `docs/CONTINUATION_PLAN.md` (57 KB, superseded) and `docs/SECTION_B_PLAN.md`
(Tier 1 complete). **Ignore `docs/PROJECT_PROGRESS.md` entirely** — it is the
24 Jul audit and is now actively wrong: it lists WebSocket streaming, SHAP and the
ReAct loop as absent, and all three exist.

**State — take these as given, don't re-derive them:**
- **984 tests pass, 1 skipped.** `tests/backend/` needs Postgres:
  `docker compose up -d postgres`. `duckdb`, `slowapi` and `statsmodels` must be
  installed.
- Evaluation harness stable at **0.836 / 0.920 / 0.883 / 1.000**, re-run on 22 Aug
  after all of that day's changes with every metric identical (only the timestamp
  and sub-second runtimes moved). **Quote 0.836, never 0.940** — and give the
  reason in the same breath (`BUILD_LOG_2026-08-17.md` §7).
- NFR-02/03/04 and M8 rate limiting closed 18 Aug. **NFR-01 is the only unmet
  requirement**: Docker yes, but `infra/k8s/` contains a README and no manifests.
- Gemini is wired and live via `GEMINI_API_KEY` in `.env`, opt-in only.
- The stack is `docker compose` (backend :8000, frontend :3000, postgres :5432).
  Only `./data` is mounted — **source is baked into the image**, so any code change
  needs `docker compose up -d --build backend frontend` before it exists in the
  browser. The frontend is a production build; nothing hot-reloads.
- **There is no frontend test framework.** No vitest, no jest, no jsdom. Anything
  React-behavioural has to be verified with a throwaway harness (`npm i jsdom
  react react-dom` in a scratch dir works) or in a real browser.

**Conventions I expect you to follow:**
- **Test-driven**: write the test first, watch it fail *for the right reason*, then
  implement. **Vacuous tests are this project's recurring failure** — four caught so
  far. If a test guards existing behaviour, prove it by breaking the feature and
  watching that test go red.
- **Measure thresholds, don't guess them.** Two retrieval floors were set on 22 Aug
  from measured score distributions, and the measurement is recorded in the
  constant's own comment. Re-measure before changing either.
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

## What changed on 22 Aug

Three reported bugs, all fixed, all found by using the running app rather than by
reading code.

**Follow-up chat returned the first answer every time.** Recommendations were built
one per MiningAgent finding; findings are a property of the dataset, and a real
dataset yields more than the five available slots, so the goal-led retrieval pass
never ran. Fixed with a slot budget (`_Budget` — the opening report is unchanged, a
follow-up reserves slots for the question), real conversation state
(`agents/conversation.py` and a `conversation` field on `POST /analysis/run`), and
relevance filtering of findings against the question.

**The spreadsheet showed page 1 whatever page you were on.** Cells were
uncontrolled inputs; React sets the DOM dirty-value flag on mount, so
`defaultValue` stops driving the display. A second bug found alongside it was
worse: pending edits were keyed by screen position and saved with the *current*
offset, so an edit made on page 1 and saved from page 2 wrote to a different row.

**Editing a numeric cell 500'd.** Grid values are strings; pandas 3 raises
`TypeError` assigning one into an `int64` column, and TypeError isn't the
`ProcessingError` the router maps to 400. Values are coerced per-dtype now, with
widening reported and impossible values returned as a 400 naming the column.

Then, probing the live app with a Spotify dataset, *"tell me about drake artist"*
came back with a card about SHAP and LIME — a genuine RAG hallucination. Three
grounding fixes followed: the broad fill pass no longer pads a follow-up it cannot
answer, goal-led cards carry a measured 0.30 confidence floor, and `QAAgent` gained
a row lookup that reads the dataframe already on the pipeline context.

Tests 839 → 984. Harness re-run after everything: unchanged.

---

## What's actually left

### 1. Chat: stop re-running the whole pipeline per turn · ~3h

Every chat message runs ingestion + mining + visualization again **and** writes an
`AnalysisRun` row and a `run_memory` row. Follow-ups are slower than they need to
be, and the chat pollutes both the history list and the run-memory corpus with
turns like "why?".

The prior run's `steps` are persisted, so mining and ingestion output can be reused
and only classification + recommendation re-run. **One product question first:** the
demo currently shows each follow-up as its own entry in history. If that matters,
keep the persistence and skip only the recomputation.

### 2. Chat: aggregates and group filters · size unknown, decide before building

The row lookup answers "tell me about Drake". It deliberately refuses **groups and
aggregates** — "which country has the most artists", "which artists are the
outliers in Total Streams" — because a country names many rows and the question is
a computation over them. Those go to the dataset Query tab's ask-in-English → SQL,
which already answers them properly, and chat says so.

"Which artists are the outliers…" is a question an examiner may well type, and it
currently gets the right *finding* (3 outliers in Total Streams, 8.6% of rows) but
not the *names*. Deciding whether chat should answer it is a scope call, not a bug
fix. The NL→SQL path exists and needs `GEMINI_API_KEY`.

### 3. Chat: nothing explains the previous answer · ~2h

`"why?"` now resolves against the previous question and retrieves *differently* —
but nothing actually explains the last answer. There is no explain-the-last-turn
path; the closest thing is `ExplainAgent` behind the per-card "Explain further"
button. Worth wiring for the demo, because "why?" is the most natural follow-up
anyone types.

### 4. The RAG-vs-LLM comparison has a confound · document it, ~1h

This matters for the report more than for the code.

RAG mode has **three** paths, not one: `QAAgent` (deterministic, computed from the
data, runs first and short-circuits everything), then RAG retrieval, then nothing.
LLM mode has one. So "tell me about Drake" is answered by path 1 — it never touches
retrieval, and the card carries no citations. Presenting that screenshot as "RAG
beats LLM" invites the obvious question: is that RAG winning, or the deterministic
QA layer?

Measured on 22 Aug against the live stack: asked the same question, **Gemini did not
hallucinate.** It declined cleanly and correctly, explaining it had only summary
statistics. So that example does *not* support "RAG doesn't hallucinate" — both
modes were honest, and the difference came from a tool one mode has.

Two things worth doing:
- **Sharpen the claim.** Not "RAG doesn't hallucinate" (today's bug *was* a RAG
  hallucination — irrelevant retrieved content served as an answer) but **"RAG knows
  what it doesn't know."** Retrieval returns a score, so the system can threshold and
  abstain; free-form generation has no such number. That is now implemented and
  measurable, and owning the bug alongside it is more credible than claiming it
  can't happen.
- **Find a real hallucination example.** The realistic LLM failure with this prompt
  isn't invented facts, it's unwarranted causal claims off correlations.
  `% of Solo Streams` vs `% of Collaborative Streams` has r = −1.00 — they are
  complements, arithmetically forced, not a finding. If Gemini narrates that as an
  insight, that's the slide.

Row lookup is deliberately **not** in LLM mode. Adding it either makes the two modes
identical for that question, or turns LLM mode into RAG by putting retrieved rows in
the prompt. The honest LLM-side equivalent is tool use, which is a third arm of the
comparison, not a modification of the second.

### 5. B2 — DB / REST ingestion endpoint (M1) · ~3h · no new deps

The last unbuilt feature. `load_from_database` and `load_from_url` already work
with `classify_source` routing; only the HTTP surface is missing. Makes FR-01
demonstrable rather than "supported internally".

⚠️ **The one item with real security risk.** A user-supplied URI fetched by the
*server* is SSRF. Block `localhost` / `127.0.0.1` / `::1` / link-local / private
ranges and `file://`, allow-list schemes, cap size and timeout. A DB URI carries
credentials — never log it, never echo it in an error. Write all of that as tests.
The SQL console sets the bar: eight escape attempts blocked, nothing leaked.

### 6. The RAG guidance prose is still flattened · ~2h

`_strip_markdown_structure` in `agents/recommendation_agent.py` strips headings,
emphasis and tables from retrieved chunks, so a chunk that is a heading plus
bullets renders as run-on prose — *"Outlier Detection IQR (Interquartile Range)
method For a single numeric column…"*. The card work gave the guidance a place to
breathe; the guidance itself is still a paragraph.

Its existing tests encode the current flattening on purpose — read them before
changing it.

### 7. Test and code debt, all small

- **No frontend test framework.** The pagination fix has no committed regression
  test — it was verified with a throwaway jsdom harness. Adding vitest + jsdom is a
  real infra decision, not a tweak.
- **Three filter tests hand-write the attribution literal**
  (`tests/agents/test_recommendation_agent.py`). They pass and act as wording pins,
  but they are *named* as filter tests and would fail for a phrasing reason.
- **`LOW_ATTRIBUTION_MODEL_SCORE` is an absolute cutoff**, not one relative to a
  majority-class baseline. 0.3 accuracy over 10 balanced classes is unremarkable;
  the same 0.3 over 2 classes is barely above chance. Documented in-line in
  `agents/orchestrator.py`.
- **The suite flake did not recur.** The full suite ran clean six times on 22 Aug.
  The `data/chroma_db` contention theory is still unproven and still the best guess.
- **`docs/PROJECT_PROGRESS.md` should be rewritten or deleted.** It is the most
  misleading file in the repo.
- **`middleware.ts` → `proxy.ts`** is a Next 16 deprecation that still builds. Note
  it, don't chase it.
- **Five pre-existing eslint errors** in `frontend/app` (two `setState`-in-effect,
  two unescaped entities, one unused `ArrowRightIcon`). Unchanged on 22 Aug —
  the count is the regression check.

### 8. A palette finding, deliberately not acted on

`SERIES_COLORS` in `frontend/app/components/charts.tsx` fails a chroma-floor check
(all six read as near-grays), and the pair `#6d6875` / `#4a4e69` separates at only
**ΔE 9.9** for normal vision — below the 15 floor. Harmless today because no chart
places those two together, but any future chart that does would be unreadable.
**This is a brand decision, not one to make inside a chart component** — raise it,
don't quietly change it.

### 9. MUST — before submission, not optional

1. **Merge to `main`.** It still sits at `b64ed19` while the branch carries
   everything — **103 commits**. The submitted default branch should be the real
   state. Nobody else will do this.
2. **Freeze the code**, run the harness one final time, and use *those* numbers
   everywhere.
3. **Regenerate `MAGE_Project_Report.pdf`** via `scripts/generate_project_report.py`
   — it is gitignored, so this is a deliberate step and teammates must run it
   themselves.
4. **`docker compose up --build` from clean** once more after the last change.
5. **Rehearse the divergence answer**: 0.836, not 0.940, *with the reason in the
   same breath*. `BUILD_LOG_2026-08-17.md` §7 has the three measured options.
6. **Rehearse the RAG-vs-LLM answer** too — see item 4. The confound is the question
   an examiner is most likely to find.

### 10. Deliberately not doing — the reasoning still holds

Celery + Redis (the WebSocket already delivers the visible benefit), LIME
(re-inflates the image that was cut 9.87 GB → 3.5 GB, and it is *local*
attribution against SHAP's *global* — not interchangeable), choropleths (boundary
files and a projection library for one chart), OCR (needs the `tesseract` system
binary, a stated exclusion reads better than a rushed inclusion), and the 3-view
wizard (pure UI churn in the flow every demo depends on).
