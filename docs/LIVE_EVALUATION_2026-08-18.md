# MAGE — Live End-to-End Evaluation, 18 August 2026

> **What this is:** the whole product driven through Chrome against the
> containerised stack, plus API-level probes, on **real datasets from public
> repositories** and deliberate edge cases. Not a unit-test run — the actual site.
>
> **Stack:** `docker compose` (backend + frontend + postgres), Gemini key live.
> **Scope:** 128 recorded checks across 8 phases. Raw log:
> `.browser-check/eval/results.jsonl`; screenshots alongside it.

---

## 1. Verdict

**118 pass · 1 real bug (fixed) · 1 real limitation · 8 harness errors of my own.**

Nothing in the product is broken. The single genuine defect found was a wasted
chart card on an all-null column, fixed the same session with tests. One real
limitation surfaced in goal classification. Everything else that showed red was
my test code addressing the UI wrongly, each retested and passing.

| Phase | Checks | Result |
|---|---|---|
| 1 — pages, routing, auth guard | 21 | **21 pass** |
| 2 — analysis matrix, 9 real datasets | 37 | **36 pass, 1 warn** |
| 3 — edge-case data | 10 | 1 real bug, rest graceful |
| 4 — exports, explain, chat, workbench | 27 | all pass after selector fixes |
| 5 — security & isolation | 28 | **28 pass** |
| 6 — scale | 3 | **3 pass** |

---

## 2. Datasets used — real, from public repositories

Not synthetic. Downloaded live from `mwaskom/seaborn-data` on GitHub and
scikit-learn's copies of UCI / StatLib:

| Dataset | Shape | Why it earns its place |
|---|---|---|
| `titanic` | 891 × 15 | **real missing values**, mixed types |
| `penguins` | 344 × 7 | missing values, natural clusters |
| `iris` | 150 × 5 | canonical multiclass |
| `wine` | 178 × 14 | multiclass, all-numeric |
| `breast_cancer` | 569 × 31 | wide, binary |
| `diabetes` | 442 × 11 | genuine regression |
| `tips` | 244 × 7 | small, mixed |
| `mpg` | 398 × 9 | outliers, missing horsepower |
| `flights` | 144 × 3 | **time series** |
| `california_housing` | 20 640 × 9 | large; real lat/long |
| `diamonds` | 53 940 × 10 | large, high-cardinality categoricals |

Plus 11 hand-built edge cases: empty file, headers-only, single column, all-null
column, constant column, `<revenue>`/`a&b`/`naïve_café_ü` headers, ragged rows,
id column, 60 000 rows, and a `.txt` that is not tabular at all.

---

## 3. What works

### Goal conditioning holds on real data
Eight of nine goals classified exactly as intended across genuinely different
datasets, and the chart set changed with the goal every time — the project's
central claim, observed rather than asserted:

| Dataset | Goal | Classified | Charts |
|---|---|---|---|
| iris | classify species | `classification` | grouped_bar, box_by_class |
| titanic | predict survival | `classification` | 3 charts |
| penguins | natural groupings | `clustering` | cluster_scatter, pairplot |
| diabetes | predict progression | `regression` | scatter, heatmap, importance |
| mpg | unusual cars | `anomaly_detection` | boxplots + highlighted_scatter |
| wine | general profile | `reporting` | **14 charts** |

### Everything else that passed
- **All 5 pages** render; **all 4 protected routes** redirect when signed out;
  sidebar navigation works; **zero JS console errors anywhere**, across every phase.
- **All three exports** download with correct filenames
  (`mage-report-<id>.pdf`, `…json`, `…bib`).
- **Explain further** returns Gemini-synthesised text carrying **7 citation chips**.
- **Workbench**: Overview profiling (verified against real penguins statistics —
  `bill_length_mm` mean 43.92, `sex` 11 missing), Spreadsheet grid, Clean
  operations, Query console.
- **Follow-up chat** answers in both RAG and LLM mode.
- **Three expertise registers measurably differ**: 1 461 / 2 553 / 3 029
  characters on one dataset (FR-04).
- **Every recommendation carries sources** (FR-03).
- **FR-05 comfortably met**: 60 000 rows profiled in **3.4 s**; diamonds
  (53 940 × 10) in 4.1 s; california_housing in 1.3 s.

### Security — 28/28, nothing leaked
Nine endpoints probed as another user: **all 404**. Six probed anonymously: **all
401**. Three tampered tokens (garbage, alg-none-style, truncated): **all 401**.
The victim retained access throughout.

**Eight SQL escape attempts, all blocked**, and critically **no file contents
leaked**:
`DROP TABLE` · `DELETE` · `UPDATE` · `INSERT` · stacked `SELECT 1; DROP TABLE` ·
`read_csv_auto('/etc/passwd')` · `COPY … TO` · `ATTACH '/etc/passwd'`.

That last group matters most: the NL→SQL feature lets Gemini author SQL, and the
guard holds regardless of who wrote it.

### Edge data fails *gracefully*, with usable messages
| Input | Response |
|---|---|
| empty file | 400 — "Empty file: the provided dataset has no content." |
| headers only | 400 — "the loaded dataset has 0 rows." |
| `.txt` | 400 — "Unsupported extension '.txt': supported types are …" |
| ragged rows | 400 — delimiter detection failure |
| weird headers (`<revenue>`, `a&b`, unicode) | **works**, 5 charts, export intact |
| constant column | **works** — correctly no violin |
| 60 000 rows | **works** — 3.4 s |

---

## 4. Findings

### 4.1 🐛 All-null column produced a chart of nothing — **FIXED**

An entirely-NaN column is still typed *numeric*, with every statistic `None`.
`_boxplot_specs(force_numeric=True)` — the fallback the `box` directive uses when
no column has outliers — selected it anyway, producing a card headed
*"Distribution of 'allnull'"* containing the words *"Not enough data for a box
plot."*

Graceful, but wasted, and on a narrow frame it displaces a chart that would have
said something. Missingness already has its own chart.

Fixed with 4 tests. **Worth recording: my first version of those tests passed
vacuously.** I built the fixture with `[None] * 4`, which pandas makes an
*object* column typed *categorical* — so it never reached the boxplot path and
the test proved nothing. CSV ingestion produces `float64` NaN, typed *numeric*.
Corrected to `np.full(n, np.nan)`, the tests failed properly, then passed. That
is the second vacuous-test near-miss in this project's history; the first is in
the 12 Aug log.

### 4.2 ⚠️ A time-series goal misclassifies — **open, real**

> `flights.csv` · *"show me the trend in passengers over time"* → **`clustering`**

There is no `time_series` task type, so a trend goal falls through to the
embedding fallback and lands somewhere arbitrary. The run still produces useful
output, but the classification is wrong and an examiner asking for a temporal
analysis would see it.

Directly relevant to the planned Tier 2 time-series work: **the classifier needs
the vocabulary before decomposition is worth adding.** Adding `seasonal_decompose`
without fixing this would compute a decomposition for a goal the system thinks is
clustering.

### 4.3 ⚠️ A legitimate single-column CSV is rejected

`only\n1\n2\n3` → 400 *"Delimiter detection failure: Could not determine
delimiter"*. Defensible — a one-column file genuinely has no delimiter — but a
single-column upload is a reasonable thing to do, and the message blames the
wrong thing. Same misleading message appears for ragged rows, where the real
problem is inconsistent field counts.

### 4.4 ℹ️ Rate limiting throttles heavy testing

My own harness hit 429s while creating users in bulk. **The feature working
correctly**, but worth knowing before a demo that hammers signup: the default is
20/minute per host, and login/register are necessarily host-keyed.

---

## 5. Eight red results that were my fault, not the product's

Recorded because the raw log shows them, and because they are a fair caution
about browser-test evidence:

| What I did wrong | Reality |
|---|---|
| Looked for placeholder `"Ask a follow-up question..."` | The UI uses a Unicode ellipsis `…` |
| Clicked the dataset filename to open the workbench | The row has an explicit **Open** link |
| Expected `table td` on the workbench landing tab | It opens on **Overview** — profiling *cards* |
| Searched for "SQL"/"Version" text | The tab is **Query**; version is a **v1** badge |
| `get_by_role("button", name="Run")` | Matched **"Translate & Run"**, which is disabled until the NL box has text |
| Cleared a textarea then `type()`d | React state stayed empty, so **Run** stayed disabled; `fill()` works |
| `full_name: "E"` on register | Correctly rejected — min length 2 |
| `full_page` screenshot of the report | The page has an **inner scroll container** |

Each was retested with the correct approach and passed.

---

## 6. Still absent (unchanged, and deliberately so)

Verified absent by grep, not assumed: **Celery/Redis job queue**, **LIME**,
**time-series decomposition**, **choropleths**, **OCR**, **DB/REST ingestion over
HTTP**, **3-view onboarding wizard**. `middleware.ts` → `proxy.ts` remains a live
Next 16 deprecation that still builds.

`california_housing` does carry real latitude/longitude, so a choropleth now has
*a* dataset — but it would still need boundary files and a projection library,
and it remains the worst effort-to-payoff item on the list.

---

## 7. Bottom line

The product holds up under real data, adversarial input, and scale. Security is
the strongest area — 28/28 with nothing leaking, including against LLM-authored
SQL. The one genuine bug was cosmetic and is fixed. The one genuine limitation is
goal classification on temporal goals, which should be settled **before** any
time-series feature work, not after.

Suite after this session: **750 passed, 1 skipped.**
