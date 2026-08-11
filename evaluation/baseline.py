"""
evaluation/baseline.py
───────────────────────
The generic AutoEDA baseline MAGE is measured against.

The proposal names **ydata-profiling** as the comparison point. Its defining
property — the one the experiment turns on — is that its computation set is a
property of the *tool*, not of the user's question: profiling the same frame
twice with two different analytical intentions produces byte-identical output,
because there is nowhere to express an intention.

Two paths are provided:

  • ``baseline_computations()`` (default) returns a **pinned declarative spec**
    of what ydata-profiling computes, expressed in MAGE's own computation
    vocabulary so the two are directly comparable. No extra dependency, no
    network, deterministic, fast.
  • ``profile_computations(df)`` (opt-in, ``--live-baseline``) actually runs
    ydata-profiling and derives the set from the report it produces, which
    validates that the pinned spec matches the real tool.

The declarative path is not a shortcut around the comparison. Goal-invariance
is *structural*: `ProfileReport(df)` has no goal parameter. The live path
exists so that claim is verified rather than asserted.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ── The pinned spec ──────────────────────────────────────────────────────────
#
# ydata-profiling's per-report computation set, mapped onto MAGE's computation
# tokens (agents/planner.py) so Jaccard comparisons are like-for-like.
#
# Sources — ydata-profiling report sections, v4.x:
#   "Overview"            → dataset shape, duplicate rows, per-column cardinality
#   "Variables"           → descriptive statistics per column
#   "Variables → missing" → per-column missing counts, plus the Missing values section
#   "Variables → histogram" / "Common values" → univariate distributions
#   "Correlations"        → Pearson / Spearman / Kendall / Phik matrices
#   "Extreme values"      → per-column smallest/largest values (tail inspection)
#
# Deliberately ABSENT from the baseline, because ydata-profiling does not
# perform them at all: kmeans, dbscan, silhouette, standardize,
# isolation_forest, class_balance, linearity_check, feature_importance.
BASELINE_COMPUTATIONS: frozenset[str] = frozenset(
    {
        "descriptive_profile",
        "missingness",
        "distribution",
        "correlation",
        "distribution_tails",
        # Baseline-only tokens — outside MAGE's vocabulary. Kept in the set
        # rather than dropped, so the comparison does not quietly discard work
        # the baseline genuinely does.
        "duplicates",
        "cardinality",
        "interactions",
    }
)

# Report sections whose presence implies each token, used by the live path to
# derive the set from an actual ProfileReport description.
#
# Keyed on the section being *present*, never on it being non-empty: an empty
# `duplicates` frame means the duplicate check ran and found nothing, which is
# evidence the computation happened, not evidence that it didn't.
_SECTION_TOKENS: dict[str, str] = {
    "table": "descriptive_profile",
    "variables": "descriptive_profile",
    "missing": "missingness",
    "correlations": "correlation",
    "scatter": "interactions",
    "duplicates": "duplicates",
}


def baseline_computations(_df: pd.DataFrame | None = None, goal: str | None = None) -> set[str]:
    """
    Return the computations the AutoEDA baseline performs.

    Both parameters are accepted and both are ignored — that is the point.
    The signature deliberately mirrors a goal-conditioned call so the harness
    can invoke baseline and MAGE through the same shape, and so the
    goal-invariance is visible at the call site rather than buried.

    Parameters
    ----------
    _df : pd.DataFrame, optional
        Ignored. The baseline's computation set does not vary by data.
    goal : str, optional
        Ignored. The baseline has no mechanism to express an analytical goal.

    Returns
    -------
    set[str]
        A copy of :data:`BASELINE_COMPUTATIONS`.
    """
    if goal:
        logger.debug("Baseline ignoring goal %r — computation set is goal-invariant.", goal)
    return set(BASELINE_COMPUTATIONS)


# ── Live validation path (opt-in) ────────────────────────────────────────────


def ydata_profiling_available() -> bool:
    """True if ydata-profiling is importable, so the harness can degrade gracefully."""
    try:
        import ydata_profiling  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "unavailable"
        return False
    return True


def profile_computations(df: pd.DataFrame, goal: str | None = None) -> set[str]:
    """
    Derive the baseline's computation set by actually running ydata-profiling.

    Used by ``--live-baseline`` to verify :data:`BASELINE_COMPUTATIONS` against
    the real tool. Raises if ydata-profiling is not installed — the caller is
    expected to have checked :func:`ydata_profiling_available` first.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to profile.
    goal : str, optional
        Ignored — accepted only to match :func:`baseline_computations`.

    Returns
    -------
    set[str]
        Computation tokens implied by the sections the report produced.
    """
    from ydata_profiling import ProfileReport

    if goal:
        logger.debug("Live baseline ignoring goal %r — ProfileReport takes no goal.", goal)

    # The FULL report, deliberately — `minimal=True` disables correlations,
    # interactions, and duplicate detection, so deriving the spec from a
    # minimal run would understate what the baseline actually does and make the
    # comparison unfairly favourable to MAGE.
    report = ProfileReport(df, minimal=False, progress_bar=False)
    description: Any = report.get_description()
    if not isinstance(description, dict):
        # v4.6+ returns a BaseDescription dataclass rather than a plain dict.
        description = vars(description)

    found: set[str] = set()
    for key, token in _SECTION_TOKENS.items():
        if key in description and description[key] is not None:
            found.add(token)

    # Tokens implied by per-variable output rather than a top-level section:
    # ydata-profiling computes a histogram, value counts, and extreme values
    # for every variable it profiles.
    if description.get("variables"):
        found.update({"descriptive_profile", "distribution", "cardinality", "distribution_tails"})

    logger.info("Live ydata-profiling baseline derived %d computation token(s).", len(found))
    return found
