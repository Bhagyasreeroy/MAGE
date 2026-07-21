"""
agents/visualization_agent.py
──────────────────────────────
VisualizationAgent — goal-aware chart selection.

Consumes MiningAgent's statistical profile (and the raw DataFrame, for
histogram binning) and emits a small set of chart specs. Specs are plain
JSON — {type, title, ...data} — rendered by the frontend without any
particular charting library, following the same selection rules
documented in data/knowledge_base/distribution_profiling.md so the
agent's behavior matches what the RAG layer cites.

Chart types emitted:
    - "correlation_heatmap" : numeric x numeric Pearson matrix
    - "feature_importance"  : bar chart of PCA-loading feature ranks
    - "cluster_scatter"     : 2D PCA projection colored by KMeans cluster
    - "histogram"           : binned distribution for a numeric column
    - "boxplot"             : five-number summary for a numeric column
    - "bar"                 : top category frequencies for a categorical column
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MAX_HISTOGRAMS = 3
MAX_CATEGORICAL_BARS = 2
HISTOGRAM_BINS = 12


class VisualizationAgent:
    """
    Selects and specifies EDA charts conditioned on the analytical goal.

    Input context keys consumed:
        - ``dataframe``          : canonical DataFrame from IngestionAgent
        - ``MiningAgent_output`` : statistical profile + patterns
        - ``goal``                : analytical goal (biases chart priority)

    Output keys produced:
        - ``viz_specs``: list of chart spec dicts (type, title, …data)
    """

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        goal = str(context.get("goal", "")).lower()
        logger.info("VisualizationAgent.run() | goal=%r", goal)

        mining = context.get("MiningAgent_output") or {}
        df: pd.DataFrame | None = context.get("dataframe")

        if not mining or df is None:
            return {
                "viz_specs": [],
                "message": "No mining output or dataset available — visualization skipped.",
            }

        specs: list[dict[str, Any]] = []

        correlations = mining.get("correlations") or {}
        if correlations:
            cols = list(correlations.keys())
            specs.append(
                {
                    "type": "correlation_heatmap",
                    "title": "Correlation Matrix",
                    "columns": cols,
                    "matrix": [[correlations[c].get(o) for o in cols] for c in cols],
                }
            )

        clustering = mining.get("clustering")
        if clustering:
            specs.append(
                {
                    "type": "cluster_scatter",
                    "title": f"Clustering (k={clustering['k']}, silhouette={clustering['silhouette_score']})",
                    "points": clustering["points"],
                }
            )

        feature_importance = mining.get("feature_importance") or []
        if feature_importance:
            specs.append(
                {
                    "type": "feature_importance",
                    "title": "Feature Importance (PCA loading)",
                    "items": [{"label": f["feature"], "value": f["score"]} for f in feature_importance],
                }
            )

        statistics = mining.get("statistics") or {}
        data_quality = mining.get("data_quality") or {}
        outliers = mining.get("outliers") or {}

        # Prioritize numeric columns with outliers or high goal-relevance for
        # the limited histogram/boxplot slots, per the goal-conditioning
        # guidance in data/knowledge_base/distribution_profiling.md. Skip
        # near-unique columns (row_id-like) — a histogram of an id column
        # carries no distributional signal.
        numeric_cols = [
            c
            for c, s in statistics.items()
            if s.get("type") == "numeric" and data_quality.get(c, {}).get("uniqueness_pct", 0) < 90
        ]
        numeric_cols.sort(key=lambda c: (c not in outliers, c.lower() not in goal))

        for col in numeric_cols[:MAX_HISTOGRAMS]:
            hist = self._histogram_spec(df, col)
            if hist:
                specs.append(hist)

        for col in outliers:
            stat = statistics.get(col, {})
            if not stat:
                continue
            specs.append(
                {
                    "type": "boxplot",
                    "title": f"Distribution of '{col}' (outliers flagged)",
                    "column": col,
                    "min": stat.get("min"),
                    "q1": stat.get("q1"),
                    "median": stat.get("median"),
                    "q3": stat.get("q3"),
                    "max": stat.get("max"),
                    "outlier_bounds": outliers[col].get("bounds"),
                }
            )

        categorical_cols = [c for c, s in statistics.items() if s.get("type") == "categorical"]
        for col in categorical_cols[:MAX_CATEGORICAL_BARS]:
            top_values = statistics[col].get("top_values", [])
            if not top_values:
                continue
            specs.append(
                {
                    "type": "bar",
                    "title": f"Top values in '{col}'",
                    "items": [{"label": v["value"], "value": v["count"]} for v in top_values],
                }
            )

        return {
            "viz_specs": specs,
            "message": f"Selected {len(specs)} chart(s) for this goal.",
        }

    def _histogram_spec(self, df: pd.DataFrame, column: str) -> dict[str, Any] | None:
        series = df[column].dropna()
        if len(series) < 2 or series.nunique() < 2:
            return None
        try:
            counts, edges = pd.cut(series, bins=HISTOGRAM_BINS, retbins=True, duplicates="drop")
            binned = counts.value_counts(sort=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Histogram binning failed for %s: %s", column, exc)
            return None

        bins = [
            {
                "label": f"{self._format_bin_edge(interval.left)}–{self._format_bin_edge(interval.right)}",
                "count": int(count),
            }
            for interval, count in binned.items()
        ]
        return {"type": "histogram", "title": f"Distribution of '{column}'", "column": column, "bins": bins}

    @staticmethod
    def _format_bin_edge(value: float) -> str:
        """Format a bin edge with enough precision to stay distinct from
        its neighbor — %.2g collapses to identical labels for numbers
        like 1001-1006 (e.g. "1e+03–1e+03")."""
        if value == 0:
            return "0"
        magnitude = abs(value)
        if magnitude >= 1000:
            return f"{value:,.0f}"
        if magnitude >= 1:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{value:.3g}"
