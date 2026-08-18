# MAGE — Remaining Features: Implementation Plan

> **Written:** 18 Aug 2026 · **Submission:** 25 Aug 2026 · **Working days left: ~5** (reserve 23–25
> for the write-up, rehearsal and freeze)
>
> Every dependency below was `pip install --dry-run`-checked on 18 Aug. **None downgrade the pinned
> stack** (pandas 3.0.3 / numpy 2.4.6 / scipy 1.18.0 / sklearn 1.9.0). That was the main risk and it
> is cleared.

---

## 0. Two rules that govern everything here

**Rule 1 — anything touching `planner.py` computations moves the evaluation numbers.**
Computation divergence is the project's headline claim. Only **T6 (time-series)** adds a computation.
Decide its relevance-set treatment *before* re-running the harness, and record it, exactly as was
done for `shap_attribution` (11 Aug) and the reporting broadening (17 Aug). Everything else in this
plan is evaluation-neutral — verify that with a re-run rather than assuming it.

**Rule 2 — every new dependency must land in three places.**
`backend/pyproject.toml` → rebuild the Docker image → confirm the import *inside* the container.
The Dockerfile's silent fallback is gone (18 Aug), so a missing dep now fails the build loudly
instead of shipping a broken image. Do not skip the container check; that is exactly how the
`GEMINI_API_KEY` bug survived.

---

## 1. Tier 1 — do these first (≈1.5 days, closes 3 NFRs + 1 module bullet)

Highest marks-per-hour in the project. Each is self-contained, low-risk, and converts a ❌ in the
requirements scoreboard into a ✅.

### T1. NFR-02 — incremental vector indexing · ~2h · **no new deps**

**This is a real bug, not just an unverified requirement.** Two findings on 18 Aug:

1. `VectorStore.add_documents` assigns `str(uuid.uuid4())` per document. The docstring says *upsert*;
   it always **inserts**. Re-adding the same chunk duplicates it.
2. Both agents guard with `_ensure_kb_loaded`, whose test is *"does retrieving 'eda methodology'
   return anything"* — i.e. **is the store non-empty**. So on a populated store nothing is ever
   indexed again.

Consequence: **add a 15th knowledge-base document and it is silently never indexed.** That directly
undercuts the reported "citation quality improves with corpus size" result, because the corpus can
no longer grow.

**Fix.** Derive the id from a content hash — `sha256(source + chunk_index + text)` — so re-adding is
a true upsert, then replace the emptiness guard with a diff: index only chunks whose id is absent.

**Tests.** Re-loading an unchanged KB leaves the count at 54. Adding one document adds only its
chunks. Editing a document replaces its chunk rather than duplicating it. Deleting is out of scope —
say so.

**Report line:** NFR-02 ❌ → ✅, with a measured before/after count.

---

### T2. Rate limiting · ~2h · `slowapi` *(clean: 4 small packages)*

Closes a named M8 bullet. `backend/routers/auth.py:11` already concedes it is only "recommended via a
reverse proxy" — replace that comment with the real thing.

**Scope it tightly.** Limit the endpoints where abuse is cheap and costly:
`POST /auth/login` and `/auth/register` (credential stuffing), `POST /analysis/run` (CPU),
`POST /analysis/datasets/{id}/query/nl` and `/analysis/explain` (they spend Gemini quota).
Leave reads unlimited.

Key by authenticated user id where available, IP otherwise — IP-only would let one user behind a NAT
lock out a whole lab.

**Tests.** N+1 requests returns 429; the limit is per-key, so a second user is unaffected; reads are
never limited. Use a high limit in the test config so the suite does not become timing-dependent.

⚠️ **Watch:** the WebSocket route and the existing 672 tests hammer these endpoints. Make the limit
configurable via `settings` and set it permissively under test, or you will spend the afternoon
debugging 429s in unrelated suites.

---

### T3. NFR-03 — LLM retry with backoff · ~1.5h · **no new deps**

An LLM now exists, so this moved from *"moot"* to *"unmet"*. `agents/llm_client.py` raises `LLMError`
on the first failure.

Add bounded exponential backoff (3 attempts, 1s/2s/4s + jitter) around the `httpx.post`. Retry only
what is retryable: timeouts, connection errors, HTTP 429, 500, 502, 503, 504. **Never retry 400 or
403** — a malformed prompt or a bad key will fail identically three times and just triples the wait.

A hand-rolled loop is ~15 lines and avoids a dependency; `tenacity` is the alternative if you prefer
the declarative form.

**Tests.** Monkeypatch `httpx.post` (the module-level call is already faked this way in
`test_llm_client.py`): succeeds on attempt 2 after one 503; gives up after 3 with `LLMError`; a 400
is not retried; total sleep is bounded. Patch `time.sleep` so the suite stays fast.

**Report line:** NFR-03 ❌ → ✅.

---

### T4. NFR-04 — session sandboxing / dataset TTL · ~3h · **no new deps**

Current retention is "keep everything forever" — 238 datasets are sitting in Postgres right now.

**Minimum defensible version:** a `expires_at` column on `datasets` (default `created_at + N days`,
`N` from settings), a `DELETE /analysis/datasets/expired` sweep, and a startup/background task that
runs it. Do **not** build a scheduler; an on-startup sweep plus a manual endpoint is honest and
demonstrable.

⚠️ **This deletes user data.** Guard it hard: only rows past `expires_at`, never touch a dataset
referenced by a completed `analysis_run` (or the run's report breaks), and default the TTL generously
(30 days). Test the negative cases — an unexpired dataset survives, a referenced one survives.

⚠️ Needs a migration on the existing DB, same as the lineage columns. Add it to
`docs/TEAM_NOTE_pull_this_branch.md`.

---

### T5. `MAX_REACT_STEPS` honesty · ~15min · **no new deps**

`orchestrator.py:41` defines it and line 154 checks it, but the plan is always exactly 4 steps, so it
can never fire. Two honest options: delete it, or keep it and describe it accurately as a
defensive bound that the current static planner cannot reach. **Do not describe it as an active
safety guard** — an examiner reading `planner.py` will see the plan length is fixed.

Cheapest credibility win available.

---

## 2. Tier 2 — real capability (≈1 day)

### T6. Time-series decomposition — M3 · ~4h · `statsmodels` *(clean: +patsy only)*

The best-supported remaining feature, for three reasons already in place:
- `data/knowledge_base/time_series_analysis.md` **already exists**, so findings have methodology to
  cite. (The SHAP work had to add a KB doc mid-flight; this one does not.)
- `VisualizationAgent._datetime_series()` already detects a time axis **including text dates** —
  built for the line chart on 17 Aug and directly reusable.
- The `line` chart already renders the result's shape.

**Implementation.** Follow the strictly-opt-in pattern exactly (`docs/CONTINUATION_PLAN.md` §10):
add a `time_series_decomposition` token to `planner.py`, gate on `"time_series_decomposition" in
computations` in `MiningAgent` bypassing `wants()`, add `_compute_time_series()` returning
trend/seasonal/residual summaries plus the detected period, and surface it as `line` chart specs.
Test that it runs for the right goal and **not** for the wrong one.

**Design questions to settle first:**
- Which task type asks for it? A new `time_series` TaskType is a bigger change (classifier lexicon,
  planner, relevance sets, harness) than adding the token to `reporting` when a date column exists.
  **Recommend: `reporting` only**, conditional on a detected time axis.
- `seasonal_decompose` needs a `period`. Infer from the median timestamp delta (hourly → 24, daily →
  7, monthly → 12) and **emit the inferred period in the output** so a reader can judge it — same
  honesty principle as SHAP's `method` and `model_score` fields.
- It needs a regular series. Real uploads have gaps. Resample or bail cleanly, and say which.

⚠️ **This is the one item that moves the harness.** Adding it to `reporting` widens reporting's
computation set again — the same direction that already took divergence 0.940 → 0.836. Decide and
record the relevance-set treatment *before* re-running. If divergence drops materially, that is a
legitimate reason to reconsider the feature, not to quietly adjust the ground truth.

---

### T7. DB / REST ingestion endpoint — M1 · ~3h · **no new deps**

`data_pipeline/ingestion.py` already implements `load_from_database` and `load_from_url` with
`classify_source` routing. **The engine works; only the HTTP surface is missing** — `POST
/analysis/ingest` accepts a file upload and nothing else. Cheap way to make FR-01 fully demonstrable
rather than "supported internally."

Add `POST /analysis/ingest/source` taking `{source_type, uri, table|query}`, reusing
`dataset_service` for persistence so the rest of the pipeline is unchanged.

⚠️ **Security — this is the one item with real risk.** A user-supplied URI that the *server* fetches
is server-side request forgery. Block `localhost`/`127.0.0.1`/`::1`/link-local/private ranges and
`file://`, cap response size and timeout, and allow-list schemes. A DB URI additionally carries
credentials — never log it, never echo it back in an error. Write these as tests, not intentions.

---

## 3. Tier 3 — only if Tier 1 and 2 land early

### T8. Celery + Redis — M8 / NFR-01 · ~1 day · `celery[redis]` *(clean deps, real infra cost)*

Dependency-wise clean. The cost is **operational**: a Redis container, a worker service in compose, a
task-state endpoint, and frontend polling. Then all of it has to come up reliably on demo day, on a
laptop that ran out of disk yesterday.

Be clear-eyed about the payoff: **the WebSocket already delivers the entire visible benefit.** A user
sees live agent steps today. Celery buys architectural completeness and a truthful NFR-01 claim, not
a better demo.

*If you build it:* keep the synchronous path as the default and make the queue opt-in via settings,
so a broken worker cannot take the demo down. Add `redis` to the compose healthchecks.

### T9. LIME — M6 · ~4h · `lime` *(⚠️ pulls 11 packages incl. matplotlib + scikit-image)*

SHAP already provides the scaffolding: target detection, the fitted model, the `feature_attribution`
output key, the `method` field that names what actually ran. LIME slots in beside it.

**The honest framing matters more than the code.** SHAP is global attribution; LIME is a *local*
explanation of one prediction. Presenting them as interchangeable is wrong. Either explain a single
representative row, or do not add it. "We implemented two attribution methods" is worth nothing if
the report cannot say why they differ.

⚠️ Heaviest dependency here, and it re-inflates the Docker image you just cut to 3.5 GB.

---

## 4. Tier 4 — recommend against, and say so in the report

| Item | Why not |
|---|---|
| **Choropleth (M4)** | **No sample dataset has geographic data.** `region` is East/West/North/South — labels, not geography. You would need a new dataset, a projection library, and boundary files, to demo one chart. Highest effort, lowest payoff. Document as future work. |
| **OCR (M1)** | Needs the `tesseract` **system binary**, so a Dockerfile change on top of the Python dep. Already explicitly deferred in the proposal — keep it deferred and say so; a deliberate, stated exclusion reads better than a rushed one. |
| **3-view onboarding wizard (M7)** | Pure UI churn. The single-page form already does the job. No new capability, real regression risk in the flow every demo depends on. |
| **`middleware.ts` → `proxy.ts`** | Next 16 deprecation. Still builds. Cosmetic; note it, do not chase it. |

---

## 5. Suggested sequence

| Day | Work | Outcome |
|---|---|---|
| **19 Aug** | T1, T5, T3 | NFR-02 ✅, NFR-03 ✅, honesty item closed |
| **20 Aug** | T2, T4 | Rate limiting ✅, NFR-04 ✅ — **all four NFRs closed** |
| **21 Aug** | T6 | Time-series decomposition; re-run harness, record the effect |
| **22 Aug** | T7, then T8/T9 only if genuinely ahead | FR-01 fully demonstrable over HTTP |
| **23–25 Aug** | Freeze · final harness · regenerate report PDF · rehearse | Submission |

**After Tier 1 alone the scoreboard reads:** NFR-01 🟡 (Docker yes, K8s documented) · NFR-02 ✅ ·
NFR-03 ✅ · NFR-04 ✅. That is three ❌ → ✅ for about a day and a half, and it is a far stronger
report line than a half-finished Celery integration.

---

## 6. Definition of done — applies to every item

1. Written **test-first**; the test fails before the implementation exists.
2. Full suite green (baseline **672 passed, 1 skipped**; needs Postgres + duckdb).
3. Dependency added to `backend/pyproject.toml`, image rebuilt, **import verified inside the
   container**.
4. E2E re-run against the containers, not the host.
5. Harness re-run; if any number moved, record the movement *and the reason* in the build log.
6. Docs updated: `PROJECT_PROGRESS.md` scoreboard, `CONTINUATION_PLAN.md` honesty guide, the day's
   build log, and `TEAM_NOTE` if a migration is involved.
7. Committed with the reasoning in the message, pushed to `feat/m2-goal-orchestrator`.
