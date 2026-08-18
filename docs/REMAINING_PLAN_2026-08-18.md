# MAGE — What's Left: Fixes and Builds

> **Written:** 18 Aug 2026, after the live evaluation · **Submission:** 25 Aug
> **Working days: ~5** (reserve 23–25 for freeze, report and rehearsal)
>
> Supersedes the tiering in `SECTION_B_PLAN.md`, whose Tier 1 is now complete.
> Every claim below was checked against the code today, not carried forward.

**State:** 750 tests green · NFR-02/03/04 + rate limiting closed · live
evaluation 118/128 with one bug fixed · harness stable at
0.836 / 0.920 / 0.883 / 1.000.

---

## 0. The one rule that still governs sequencing

**Only `B1` (time-series decomposition) touches `planner.py` computations, so
only `B1` can move the headline divergence number.** Everything in the FIX tier
is evaluation-neutral — and that is verified, not assumed: all five harness goals
classify at 0.78–0.95, comfortably clear of any confidence floor.

Re-run the harness after `B1` and record the movement *and its reason*, the way
`shap_attribution` (11 Aug) and the reporting broadening (17 Aug) were recorded.

---

## 1. FIX — defects found in the live evaluation (~4.5h, do first)

### F1. The classifier's confidence floor is dead code · ~1.5h · **highest value**

`goal_classifier.py:37` defines `INCONCLUSIVE_BELOW = 0.5`, and line 280 returns
early when a provider clears it. But when *nothing* clears it, line 288 does:

```python
if best is not None:
    return best          # ← returns the sub-threshold guess anyway
```

The `DEFAULT_TASK_TYPE` fallback beneath it is reachable only when no provider
returns anything at all. **So a constant named `INCONCLUSIVE_BELOW` never causes
anything to be treated as inconclusive.** Measured today:

| Goal | Returned | Confidence |
|---|---|---|
| "show me the trend in passengers over time" | `clustering` | **0.33** |
| "analyse the seasonality of passengers" | `clustering` | **0.28** |
| "how do passengers change month over month" | `clustering` | **0.19** |

A 0.19 guess presented as a decision. This is the same shape of problem as
`MAX_REACT_STEPS`: a guard that reads as active and is not.

**Fix.** When the best confidence is below the floor, return `DEFAULT_TASK_TYPE`
(`reporting` — already documented as "the goal-agnostic default"), keeping the
detected target column and stating the reason in the rationale.

**Why it is the highest-value item left:** it converts three wrong answers into
one honest one, it makes the *displayed* confidence meaningful, and `reporting`
already emits the `line` chart — so a temporal goal starts producing the right
artefact without any new computation.

**Verified safe:** the five harness goals score 0.95 / 0.78 / 0.95 / 0.95 / 0.95.
Re-run the harness to confirm rather than trusting this note.

**Tests:** a sub-threshold goal returns `reporting` with a rationale naming the
uncertainty; a confident goal is untouched; the target column survives the
fallback; the five harness goals still classify as themselves.

### F2. No temporal vocabulary anywhere in the lexicon · ~1h

"trend", "over time", "seasonality", "month over month", "forecast" appear in
**no** keyword list, which is *why* F1's cases fall through to embeddings. Add
them to `reporting` (plus prototypes). Pairs naturally with F1 — F1 stops the bad
guess, F2 produces a positively correct one.

⚠️ Deliberately **not** a new `TaskType`. That would touch the enum, planner,
relevance sets and harness — a much larger change than the remaining time
justifies. Say so in the report: temporal goals are served by the reporting
profile plus a trend chart, not by a dedicated pipeline.

### F3. A legitimate single-column CSV is rejected · ~2h

`only\n1\n2\n3` → 400 *"Delimiter detection failure: Could not determine
delimiter"*. A one-column upload is a reasonable thing to do. The same message
also appears for ragged rows, where the real problem is inconsistent field
counts, so the diagnosis misleads in both directions.

**Fix.** Treat "no delimiter found but exactly one column parses" as a valid
single-column frame; give ragged rows their own message naming the row and the
expected field count.

---

## 2. BUILD — genuine features (~7h)

### B1. Time-series decomposition — M3 · ~4h · `statsmodels` *(clean: +patsy)*

**Do F1 and F2 first.** Today "analyse the seasonality of passengers" classifies
as *clustering*; adding `seasonal_decompose` before fixing that would compute a
decomposition for a goal the system believes is clustering — a feature that only
ever fires by accident.

Three things are already in place: `time_series_analysis.md` exists in the
knowledge base (so findings have methodology to cite), `_datetime_series()`
already detects a time axis **including text dates**, and the `line` chart
renders the shape.

Follow the strictly-opt-in pattern: a `time_series_decomposition` token in
`planner.py`, gated in `MiningAgent` bypassing `wants()`, emitting
trend/seasonal/residual plus **the inferred period** — quote the period in the
output the way SHAP quotes `method` and `model_score`, so a reader can judge it.
Real series have gaps: resample or decline cleanly, and say which.

⚠️ **The only item that moves the harness.** Decide the relevance-set treatment
before re-running, not after seeing the score.

### B2. DB / REST ingestion endpoint — M1 · ~3h · no new deps

`load_from_database` and `load_from_url` already work with `classify_source`
routing; only the HTTP surface is missing. Makes FR-01 demonstrable rather than
"supported internally".

⚠️ **The one item with real security risk.** A user-supplied URI fetched by the
*server* is SSRF. Block `localhost`/`127.0.0.1`/`::1`/link-local/private ranges
and `file://`, allow-list schemes, cap size and timeout. A DB URI carries
credentials — never log it, never echo it in an error. Write those as tests.
The SQL console already sets the bar here: eight escape attempts blocked, nothing
leaked.

---

## 3. PROBABLY NOT — the honest recommendation

| Item | Cost | Why I would skip it |
|---|---|---|
| **Celery + Redis** | ~1 day | Deps are clean, but it adds a Redis container and a worker to the stack you must bring up on demo day — on a laptop that ran out of disk yesterday. **The WebSocket already delivers the entire visible benefit.** Architectural completeness only. |
| **LIME** | ~4h | Pulls 11 packages incl. matplotlib + scikit-image, re-inflating the image just cut 9.87 GB → 3.5 GB. And SHAP is *global* attribution while LIME is *local*: presenting them as interchangeable is wrong, and "we implemented two methods" is worth nothing if the report cannot say why they differ. |
| **Choropleth** | high | `california_housing` does have real lat/long, so a dataset now exists — but it still needs boundary files and a projection library for one chart. Worst effort-to-payoff on the list. |
| **OCR** | high | Needs the `tesseract` **system binary**, i.e. a Dockerfile change too. Already deferred in the proposal; a stated exclusion reads better than a rushed inclusion. |
| **3-view wizard** | medium | Pure UI churn, real regression risk in the flow every demo depends on. |
| **`middleware.ts` → `proxy.ts`** | low | Next 16 deprecation; still builds. Note it, don't chase it. |

---

## 4. MUST — before submission, not optional

1. **Freeze the code**, then run the harness one final time and use *those*
   numbers everywhere.
2. **Regenerate `MAGE_Project_Report.pdf`** after the freeze — it is gitignored,
   so this is a deliberate step; teammates must run the script themselves.
3. **`docker compose up --build` from clean** once more after the last change.
4. **Merge to `main`.** It still sits at `b64ed19` while the branch carries
   everything — the submitted default branch should be the real state.
5. **Rehearse the divergence answer**: 0.836, not 0.940, *and the reason in the
   same breath*. `BUILD_LOG_2026-08-17.md` §7 has the three measured options.

---

## 5. Suggested sequence

| Day | Work |
|---|---|
| **19 Aug** | F1 + F2 (classifier honesty + temporal vocabulary), harness control run |
| **20 Aug** | F3, then B1 time-series decomposition |
| **21 Aug** | B1 finish; harness re-run and **record the movement** |
| **22 Aug** | B2 DB/REST endpoint with SSRF tests |
| **23–25 Aug** | Freeze · final harness · report PDF · merge to main · rehearse |

**If only one day is available, do F1 and F2.** They fix three wrong answers,
make a displayed confidence mean something, and cost nothing on the evaluation.

---

## 6. Definition of done (unchanged)

Test-first · full suite green (**750 passed, 1 skipped**; needs Postgres +
duckdb) · dependency in `backend/pyproject.toml` **and verified importable inside
the container** · E2E re-run against the containers · harness re-run with any
movement recorded and explained · docs updated (`PROJECT_PROGRESS.md`
scoreboard, honesty guide, the day's build log, `TEAM_NOTE` if a migration is
involved) · committed with the reasoning and pushed.
