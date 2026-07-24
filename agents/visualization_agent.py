"""
agents/visualization_agent.py
──────────────────────────────
VisualizationAgent — goal-conditioned chart selection.

Consumes MiningAgent's statistical profile (and the raw DataFrame, for
histogram binning) and emits a small set of chart specs. Specs are plain
JSON — {type, title, ...data} — rendered by the frontend without any
particular charting library, following the same selection rules
documented in data/knowledge_base/distribution_profiling.md so the
agent's behavior matches what the RAG layer cites.

The chart *set* is conditioned on the planner's ``charts`` directive
(Module 2 / M4 → FR-02): a clustering goal yields cluster/scatter views, an
anomaly goal yields box/highlighted-scatter views, a regression goal yields
scatter + heatmap, and so on. When no directive is supplied the agent falls
back to the full default chart set, so it remains useful standalone.

Chart types emitted:
    - "correlation_heatmap" : numeric x numeric Pearson matrix
    - "feature_importance"  : bar chart of PCA-loading feature ranks
    - "cluster_scatter"     : 2D PCA projection colored by KMeans cluster
    - "scatter"             : two most-correlated numeric columns
    - "histogram"           : binned distribution for a numeric column
    - "boxplot"             : five-number summary for a numeric column
    - "bar"                 : top category frequencies for a categorical column
    - "missingness_matrix"  : per-column missing-value percentages
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MAX_HISTOGRAMS = 3
MAX_CATEGORICAL_BARS = 2
HISTOGRAM_BINS = 12
MAX_SCATTER_POINTS = 500


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
        directives = context.get("directives", {}) or {}
        charts = [str(c) for c in (directives.get("charts", []) or [])]
        logger.info("VisualizationAgent.run() | goal=%r charts=%s", goal, charts)

        mining = context.get("MiningAgent_output") or {}
        df: pd.DataFrame | None = context.get("dataframe")

        if not mining or df is None:
            return {
                "viz_specs": [],
                "message": "No mining output or dataset available — visualization skipped.",
            }

        # Conditioned path: build the chart types the plan asked for. If none
        # are feasible for this data, fall back to the default chart set so a
        # run is never left with no visualisation at all.
        if charts:
            specs = self._conditioned_specs(charts, df, mining, goal)
            if specs:
                return {
                    "viz_specs": specs,
                    "message": f"Selected {len(specs)} goal-conditioned chart(s) for {charts}.",
                }

        specs = self._default_specs(df, mining, goal)
        return {
            "viz_specs": specs,
            "message": f"Selected {len(specs)} chart(s) for this goal.",
        }

    # ── Chart-set assembly ───────────────────────────────────────────────────

    def _default_specs(self, df: pd.DataFrame, mining: dict[str, Any], goal: str) -> list[dict[str, Any]]:
        """The full unconditioned chart set (standalone / reporting default)."""
        specs: list[dict[str, Any]] = []
        for spec in (self._heatmap_spec(mining), self._cluster_scatter_spec(mining),
                     self._feature_importance_spec(mining)):
            if spec:
                specs.append(spec)
        specs.extend(self._histogram_specs(df, mining, goal))
        specs.extend(self._boxplot_specs(mining))
        specs.extend(self._categorical_bar_specs(mining))
        return specs

    def _conditioned_specs(
        self, charts: list[str], df: pd.DataFrame, mining: dict[str, Any], goal: str
    ) -> list[dict[str, Any]]:
        """Build the specific chart types named in the planner's directive."""
        specs: list[dict[str, Any]] = []
        seen_types: set[str] = set()

        for chart in charts:
            built: list[dict[str, Any]] = []
            if chart in ("correlation_heatmap",):
                spec = self._heatmap_spec(mining)
                built = [spec] if spec else []
            elif chart in ("cluster_scatter", "pairplot"):
                spec = self._cluster_scatter_spec(mining)
                built = [spec] if spec else []
            elif chart in ("scatter", "highlighted_scatter"):
                spec = self._scatter_spec(df, mining)
                built = [spec] if spec else []
            elif chart in ("histograms", "histogram", "distribution"):
                built = self._histogram_specs(df, mining, goal)
            elif chart in ("box", "box_by_class", "boxplot"):
                # Prefer outlier columns; otherwise box the top numeric columns.
                built = self._boxplot_specs(mining) or self._boxplot_specs(mining, force_numeric=True)
            elif chart in ("grouped_bar", "bar"):
                built = self._categorical_bar_specs(mining)
            elif chart in ("missingness_matrix", "missingness"):
                spec = self._missingness_spec(mining)
                built = [spec] if spec else []
            elif chart in ("feature_importance",):
                spec = self._feature_importance_spec(mining)
                built = [spec] if spec else []

            for spec in built:
                # De-dup by (type, title) so repeated directives don't stack.
                key = f"{spec['type']}::{spec.get('title', '')}"
                if key not in seen_types:
                    seen_types.add(key)
                    specs.append(spec)

        return specs

    # ── Individual spec builders ─────────────────────────────────────────────

    def _heatmap_spec(self, mining: dict[str, Any]) -> dict[str, Any] | None:
        correlations = mining.get("correlations") or {}
        if not correlations:
            return None
        cols = list(correlations.keys())
        return {
            "type": "correlation_heatmap",
            "title": "Correlation Matrix",
            "columns": cols,
            "matrix": [[correlations[c].get(o) for o in cols] for c in cols],
        }

    def _cluster_scatter_spec(self, mining: dict[str, Any]) -> dict[str, Any] | None:
        clustering = mining.get("clustering")
        if not clustering:
            return None
        return {
            "type": "cluster_scatter",
            "title": f"Clustering (k={clustering['k']}, silhouette={clustering['silhouette_score']})",
            "points": clustering["points"],
        }

    def _feature_importance_spec(self, mining: dict[str, Any]) -> dict[str, Any] | None:
        feature_importance = mining.get("feature_importance") or []
        if not feature_importance:
            return None
        return {
            "type": "feature_importance",
            "title": "Feature Importance (PCA loading)",
            "items": [{"label": f["feature"], "value": f["score"]} for f in feature_importance],
        }

    def _scatter_spec(self, df: pd.DataFrame, mining: dict[str, Any]) -> dict[str, Any] | None:
        """Scatter of the two most strongly correlated numeric columns."""
        correlations = mining.get("correlations") or {}
        cols = list(correlations.keys())
        best: tuple[str, str, float] | None = None
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = correlations[a].get(b)
                if r is None:
                    continue
                if best is None or abs(r) > abs(best[2]):
                    best = (a, b, r)
        if best is None:
            return None
        x_col, y_col, r = best
        pair = df[[x_col, y_col]].dropna().head(MAX_SCATTER_POINTS)
        return {
            "type": "scatter",
            "title": f"'{x_col}' vs '{y_col}' (r={r:.2f})",
            "x_label": x_col,
            "y_label": y_col,
            "points": [{"x": float(x), "y": float(y)} for x, y in zip(pair[x_col], pair[y_col])],
        }

    def _histogram_specs(self, df: pd.DataFrame, mining: dict[str, Any], goal: str) -> list[dict[str, Any]]:
        statistics = mining.get("statistics") or {}
        data_quality = mining.get("data_quality") or {}
        outliers = mining.get("outliers") or {}
        # Prioritize numeric columns with outliers or high goal-relevance for
        # the limited histogram slots. Skip near-unique (row_id-like) columns —
        # a histogram of an id column carries no distributional signal.
        numeric_cols = [
            c
            for c, s in statistics.items()
            if s.get("type") == "numeric" and data_quality.get(c, {}).get("uniqueness_pct", 0) < 90
        ]
        numeric_cols.sort(key=lambda c: (c not in outliers, c.lower() not in goal))
        specs: list[dict[str, Any]] = []
        for col in numeric_cols[:MAX_HISTOGRAMS]:
            hist = self._histogram_spec(df, col)
            if hist:
                specs.append(hist)
        return specs

    def _boxplot_specs(self, mining: dict[str, Any], force_numeric: bool = False) -> list[dict[str, Any]]:
        statistics = mining.get("statistics") or {}
        outliers = mining.get("outliers") or {}
        if force_numeric and not outliers:
            columns = [c for c, s in statistics.items() if s.get("type") == "numeric"][:MAX_HISTOGRAMS]
        else:
            columns = list(outliers.keys())
        specs: list[dict[str, Any]] = []
        for col in columns:
            stat = statistics.get(col, {})
            if not stat:
                continue
            flagged = col in outliers
            specs.append(
                {
                    "type": "boxplot",
                    "title": f"Distribution of '{col}'" + (" (outliers flagged)" if flagged else ""),
                    "column": col,
                    "min": stat.get("min"),
                    "q1": stat.get("q1"),
                    "median": stat.get("median"),
                    "q3": stat.get("q3"),
                    "max": stat.get("max"),
                    "outlier_bounds": outliers.get(col, {}).get("bounds"),
                }
            )
        return specs

    def _categorical_bar_specs(self, mining: dict[str, Any]) -> list[dict[str, Any]]:
        statistics = mining.get("statistics") or {}
        categorical_cols = [c for c, s in statistics.items() if s.get("type") == "categorical"]
        specs: list[dict[str, Any]] = []
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
        return specs

    def _missingness_spec(self, mining: dict[str, Any]) -> dict[str, Any] | None:
        """Per-column missing-value percentages (reporting goals)."""
        data_quality = mining.get("data_quality") or {}
        if not data_quality:
            return None
        items = [
            {"label": col, "value": round(100.0 - dq.get("completeness_pct", 100.0), 1)}
            for col, dq in data_quality.items()
        ]
        return {
            "type": "missingness_matrix",
            "title": "Missing Values by Column (%)",
            "items": items,
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
