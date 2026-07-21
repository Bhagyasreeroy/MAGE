"""
agents/mining_agent.py
───────────────────────
MiningAgent — statistical profiling and pattern discovery.

Responsibilities:
    • Compute descriptive statistics: mean, median, std-dev, quantiles,
      skewness, kurtosis for numerical features.
    • Profile categorical features: cardinality, top-k frequencies.
    • Detect missing-value patterns, outliers (IQR / Z-score), and
      duplicate records.
    • Perform correlation analysis and mutual information ranking.
    • Identify temporal patterns if a datetime column exists (trend,
      seasonality, stationarity).
    • Run goal-conditioned feature selection to surface the most relevant
      variables for the user's analytical objective.

Wiring (future milestones):
    - Will consume the canonical DataFrame produced by IngestionAgent.
    - Will call DataProcessingEngine from /data_pipeline/processing.py.
    - Will emit a StatisticalProfile Pydantic model consumed by
      VisualizationAgent and RecommendationAgent.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MiningAgent:
    """
    Performs statistical profiling and goal-conditioned pattern discovery.

    Input context keys consumed:
        - ``IngestionAgent_output`` : normalised data from IngestionAgent
        - ``goal``                  : analytical goal
        - ``expertise_level``       : adapts depth of profiling

    Output keys produced:
        - ``statistics``  : descriptive stats per feature
        - ``correlations``: pairwise correlation matrix (serialised)
        - ``outliers``    : list of detected outlier records
        - ``patterns``    : discovered patterns / insights
    """

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute statistical profiling and pattern mining.

        Parameters
        ----------
        context : dict, optional
            Shared pipeline context forwarded by the Orchestrator.

        Returns
        -------
        dict
            Mining result payload.

        TODO:
            - Ingest canonical DataFrame from context.
            - Compute real descriptive stats with Pandas.
            - Detect outliers via IQR / Z-score.
            - Compute correlation matrix.
            - Apply goal-conditioned feature ranking.
        """
        context = context or {}
        directives = context.get("directives", {}) or {}
        task_type = directives.get("task_type") or context.get("task_type")
        computations = directives.get("computations", [])
        logger.info(
            "MiningAgent.run() | goal=%r task_type=%s computations=%s",
            context.get("goal"), task_type, computations,
        )

        # ── Stub output ───────────────────────────────────────────────────────
        # Module 2 wires goal-conditioning through: the agent echoes the
        # directives it was told to run so the conditional pipeline is visible
        # and testable. Module 3 replaces the stubs with real computations.
        return {
            "task_type": task_type,
            "planned_computations": computations,       # directives from the orchestrator
            "target_column": directives.get("target_column"),
            "statistics": {},    # TODO(M3): {column: {mean, std, min, max, …}}
            "correlations": {},  # TODO(M3): serialised correlation matrix
            "outliers": [],      # TODO(M3): list of outlier record indices
            "patterns": [],      # TODO(M3): list of discovered pattern strings
            "message": (
                f"[STUB] MiningAgent conditioned for '{task_type}' — "
                f"would run: {', '.join(computations) or 'default profile'}."
            ),
        }
