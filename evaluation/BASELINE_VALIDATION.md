# Baseline validation — pinned spec vs. live ydata-profiling

The evaluation harness scores MAGE against a generic AutoEDA baseline whose computation set is
declared as a pinned constant (`BASELINE_COMPUTATIONS` in `evaluation/baseline.py`) rather than
recomputed on every run. This document records the check that the constant is faithful to the real
tool, so the declarative path is a *verified* shortcut rather than an assumed one.

## What was run

`evaluation.baseline.profile_computations()` executes a **full** `ProfileReport` (not `minimal=True`,
which would suppress correlations, interactions, and duplicate detection and thereby understate the
baseline) and derives the computation set from the report's own description sections.

This was executed against all five datasets in the pinned suite.

| Environment | Value |
|---|---|
| ydata-profiling | 4.18.4 |
| Date | 11 Aug 2026 |
| Datasets | `customer_orders`, `iris`, `wine`, `breast_cancer`, `diabetes` |

Reproduce with:

```bash
PYTHONPATH=. python -m evaluation.harness --live-baseline
```

## Result

Every dataset produced this set, and only this set:

```
cardinality, correlation, descriptive_profile, distribution,
distribution_tails, duplicates, interactions, missingness
```

| Check | Outcome |
|---|---|
| Live set identical across all 5 datasets | ✅ **True** |
| Live set equals the pinned `BASELINE_COMPUTATIONS`, per dataset | ✅ **True** |
| Tokens live produced that the spec lacks | none |
| Tokens the spec claims that live did not produce | none |

**Two conclusions, both load-bearing for the evaluation:**

1. **The pinned spec is accurate.** Zero discrepancies against the real tool, so running the harness
   in its default declarative mode measures the same baseline a live run would.
2. **The baseline is invariant.** Its computation set did not change across five datasets of
   differing shape, dtype mix, and domain — and it cannot change with the goal, because
   `ProfileReport` has no parameter through which a goal could be expressed. The baseline's 0.000
   cross-goal divergence in the results is therefore structural, not an artefact of how it was
   modelled.

## Why the spec is the default path

`ydata-profiling` is **not** a project dependency, and is deliberately not added to
`backend/pyproject.toml`. Installing it forces `pandas<3`, which downgrades the project's pinned
`pandas 3.0.3 → 2.3.3` (along with `numpy 2.5.0 → 2.3.5` and `scipy 1.18.0 → 1.16.3`) and breaks
`tests/agents/test_ingestion_agent.py`, which asserts pandas 3's `str` dtype where pandas 2 reports
`object`. It also requires `setuptools<81` for `pkg_resources`, removed in setuptools 81.

Carrying that downgrade permanently, to re-derive a constant that has been shown to be exact, would
trade the project's numerics stack for nothing. The harness therefore defaults to the pinned spec
and treats `--live-baseline` as an opt-in validation path that degrades gracefully — 
`ydata_profiling_available()` returns `False` when the package is absent and the harness falls back
to the spec with a warning rather than failing.
