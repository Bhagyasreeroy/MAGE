# Prompt to open a new session with

Copy everything in the box below.

---

I'm continuing work on MAGE, a multi-agent goal-conditioned EDA system at
`/Users/22sarthak/MAGE`, branch `feat/m2-goal-orchestrator` (HEAD `e85391c`,
in sync with origin, working tree clean). **Do not touch `main`** — it stays at
`b64ed19` until I say otherwise.

**Read these three files first, and nothing else:**
- `docs/REMAINING_PLAN_2026-08-18.md` — what's left, costed and sequenced
- `docs/BUILD_LOG_2026-08-18.md` — what was done on 17–18 Aug and why
- `docs/LIVE_EVALUATION_2026-08-18.md` — the live browser evaluation and its findings

Skip `docs/CONTINUATION_PLAN.md` (57 KB, largely superseded) and
`docs/SECTION_B_PLAN.md` (its Tier 1 is complete) unless something points you there.

**State — take these as given, don't re-derive them:**
- 750 tests pass, 1 skipped. `tests/backend/` needs Postgres:
  `docker compose up -d postgres`. `duckdb` and `slowapi` must be installed.
- NFR-02, NFR-03, NFR-04 and M8 rate limiting are all closed as of 18 Aug.
- Evaluation harness is stable at **0.836 / 0.920 / 0.883 / 1.000**.
  **Quote 0.836, never 0.940** — and give the reason in the same breath
  (`BUILD_LOG_2026-08-17.md` §7 explains it).
- Gemini is wired and live via `GEMINI_API_KEY` in `.env`, opt-in only.

**Conventions I expect you to follow:**
- **Test-driven**: write the test first, watch it fail for the right reason, then
  implement. Beware vacuous tests — this project has had two near-misses, both
  recorded in the build logs.
- **Log everything** to `docs/BUILD_LOG_<today>.md`, in the style of the existing ones.
- Commit with the reasoning in the message; push to `feat/m2-goal-orchestrator` only.
- Any new dependency goes in `backend/pyproject.toml` **and** must be verified
  importable inside the container.
- Re-run the harness only if `planner.py` computations changed, and record any
  movement plus its cause.

**My task for this session:** <SAY EXACTLY WHAT YOU WANT HERE>

---

## Suggested first tasks, in the order I'd do them

Pick **one** per session and say it explicitly:

1. **"Do F1 and F2 from the remaining plan"** — the classifier confidence floor is
   dead code (`INCONCLUSIVE_BELOW = 0.5` is defined but bypassed, so a 0.19-confidence
   guess is returned as a decision), plus temporal vocabulary. ~2.5h, evaluation-neutral,
   highest value of the small items.
2. **"Plan the model-driven ReAct loop"** — the biggest gap between the project's
   title and its code; the orchestrator is a fixed 4-step `for` loop today.
3. **"Build the ablation study"** — turns "the system diverges" into "here is what
   each component contributes".
4. **"Do F3"** — a legitimate single-column CSV is rejected with a misleading
   delimiter error.
