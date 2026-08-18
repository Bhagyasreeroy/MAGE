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
    - "correlation_heatmap"  : numeric x numeric Pearson matrix
    - "feature_importance"   : bar chart of PCA-loading feature ranks
    - "cluster_scatter"      : 2D PCA projection colored by KMeans cluster
    - "scatter"              : two most-correlated numeric columns
    - "histogram"            : binned distribution for a numeric column
    - "boxplot"              : five-number summary for a numeric column
    - "bar"                  : top category frequencies for a categorical column
    - "missingness_matrix"   : per-column missing-value percentages
    - "grouped_bar"          : categorical feature cross-tabulated by target class
    - "box_by_class"         : five-number summary of one numeric column per class
    - "pairplot"             : pairwise scatter panels over the top numeric columns
    - "highlighted_scatter"  : scatter with IQR-outlier rows flagged
    - "violin"               : binned density profile of a numeric column
    - "line"                 : a numeric column's trend over a datetime column

The last six are *dedicated builders*. Four of them (`grouped_bar`,
`box_by_class`, `pairplot`, `highlighted_scatter`) were previously routed to
approximations — a grouped bar was served by a plain frequency bar, a
box-by-class by a boxplot that ignored the class, a pairplot by a single
cluster scatter, and a highlighted scatter by an unmarked one. The chart the
planner asked for and the chart the reader saw were different charts.

Each dedicated builder returns ``None`` when the data cannot support it (no
target column, too few numeric columns, no outliers, no datetime axis), and the
router then falls back to the old stand-in. That keeps a chart slot from ever
coming back empty, while making the *real* chart the default whenever it is
drawable.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MAX_HISTOGRAMS = 3
MAX_CATEGORICAL_BARS = 2
HISTOGRAM_BINS = 12
MAX_SCATTER_POINTS = 500

# Dedicated-builder limits. These keep a chart readable rather than complete —
# a 40-category grouped bar or a 10x10 pairplot is a wall, not a chart.
MAX_GROUPED_BAR_CATEGORIES = 8
# Above this many distinct values a column is a label, not a grouping.
MAX_GROUPED_BAR_DISTINCT = 25
# Above this share of distinct values, counting occurrences says nothing.
MAX_CATEGORY_DISTINCT_RATIO = 0.5
MAX_CLASS_SERIES = 5
MIN_ROWS_PER_CLASS = 3
MAX_PAIRPLOT_COLUMNS = 4
MAX_PAIRPLOT_POINTS = 200
VIOLIN_BANDS = 9
MAX_LINE_POINTS = 200
# Text-date detection: how much of the column must parse, and how long its
# values must typically be before a parse is even attempted.
_MIN_DATE_PARSE_RATIO = 0.9
_MIN_DATE_TEXT_LENGTH = 6

# Column names that denote a record identifier rather than a measurement.
# Deliberately duplicated from MiningAgent's richer check rather than imported:
# specialist agents must not import each other (see docs/CONTINUATION_PLAN.md
# §4). A name-based check is all the visualization layer needs.
_ID_NAME_EXACT = {"id", "index", "idx", "row", "row_id", "key", "uuid", "pk"}


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
            target = self._resolve_target(mining, directives)
            specs = self._conditioned_specs(charts, df, mining, goal, target)
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
        specs.extend(self._categorical_bar_specs(mining, df))
        return specs

    def _resolve_target(self, mining: dict[str, Any], directives: dict[str, Any]) -> str | None:
        """
        Find the target column for the class-aware charts.

        Read from MiningAgent's output first — ``class_balance`` and
        ``feature_attribution`` both name the target they actually used, so
        this stays consistent with what was computed rather than with what was
        requested. The directive is the fallback; note the orchestrator injects
        ``target_column`` into the *mining* step's directives only, so in a
        normal pipeline run the mining output is the only source available.
        """
        for source in (mining.get("class_balance"), mining.get("feature_attribution")):
            if isinstance(source, dict) and source.get("target"):
                return str(source["target"])
        target = (directives or {}).get("target_column")
        return str(target) if target else None

    def _conditioned_specs(
        self,
        charts: list[str],
        df: pd.DataFrame,
        mining: dict[str, Any],
        goal: str,
        target: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the specific chart types named in the planner's directive.

        Where a dedicated builder exists it is tried first; the older stand-in
        is kept only as the fallback for data that cannot support the real
        chart, so a directive never yields an empty slot.
        """
        specs: list[dict[str, Any]] = []
        seen_types: set[str] = set()

        for chart in charts:
            built: list[dict[str, Any]] = []
            if chart in ("correlation_heatmap",):
                spec = self._heatmap_spec(mining)
                built = [spec] if spec else []
            elif chart in ("cluster_scatter",):
                spec = self._cluster_scatter_spec(mining)
                built = [spec] if spec else []
            elif chart == "pairplot":
                spec = self._pairplot_spec(df, mining) or self._cluster_scatter_spec(mining)
                built = [spec] if spec else []
            elif chart == "highlighted_scatter":
                spec = self._highlighted_scatter_spec(df, mining) or self._scatter_spec(df, mining)
                built = [spec] if spec else []
            elif chart in ("scatter",):
                spec = self._scatter_spec(df, mining)
                built = [spec] if spec else []
            elif chart in ("histograms", "histogram", "distribution"):
                built = self._histogram_specs(df, mining, goal)
            elif chart == "box_by_class":
                spec = self._box_by_class_spec(df, mining, target)
                built = [spec] if spec else (
                    self._boxplot_specs(mining) or self._boxplot_specs(mining, force_numeric=True)
                )
            elif chart in ("box", "boxplot"):
                # Prefer outlier columns; otherwise box the top numeric columns.
                built = self._boxplot_specs(mining) or self._boxplot_specs(mining, force_numeric=True)
            elif chart == "grouped_bar":
                spec = self._grouped_bar_spec(df, mining, target)
                built = [spec] if spec else self._categorical_bar_specs(mining, df)
            elif chart == "bar":
                built = self._categorical_bar_specs(mining, df)
            elif chart == "violin":
                spec = self._violin_spec(df, mining)
                built = [spec] if spec else []
            elif chart == "line":
                spec = self._line_spec(df, mining)
                built = [spec] if spec else []
            elif chart in ("missingness_matrix", "missingness"):
                spec = self._missingness_spec(mining)
                built = [spec] if spec else []
            elif chart in ("feature_importance",):
                spec = self._feature_importance_spec(mining)
                built = [spec] if spec else []
            elif chart in ("feature_attribution", "shap"):
                # Falls back to the PCA ranking when attribution did not run
                # (no usable target), so the slot is never left empty.
                spec = self._feature_attribution_spec(mining) or self._feature_importance_spec(mining)
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

    def _feature_attribution_spec(self, mining: dict[str, Any]) -> dict[str, Any] | None:
        """
        Bar chart of supervised feature attribution (SHAP).

        Reuses the ``feature_importance`` render type — the frontend and the PDF
        exporter both already draw it — but carries a distinct title naming the
        target and the method, so it is never mistaken for the unsupervised
        PCA ranking sitting next to it in the same report.
        """
        attribution = mining.get("feature_attribution") or {}
        items = attribution.get("attributions") or []
        if not items:
            return None
        method = attribution.get("method", "attribution")
        target = attribution.get("target", "target")
        return {
            "type": "feature_importance",
            "title": f"Feature Attribution for '{target}' ({method})",
            "items": [{"label": f["feature"], "value": f["score"]} for f in items],
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

    def _categorical_bar_specs(
        self, mining: dict[str, Any], df: pd.DataFrame | None = None
    ) -> list[dict[str, Any]]:
        """
        Frequency bars for the genuinely categorical columns.

        Selection skips the time axis and near-unique columns. Found in the
        browser: an uploaded CSV leaves dates as text, the profiler types them
        *categorical*, and this builder charted "Top values in 'order_date'" —
        five unique timestamps, each with a count of 1 — in the default
        reporting view. Counting occurrences only says something when values
        actually repeat, so a column whose values are nearly all distinct is
        not a bar chart, and no chart beats a meaningless one.
        """
        statistics = mining.get("statistics") or {}
        time_col = None
        if df is not None:
            found = self._datetime_series(df)
            time_col = found[0] if found else None

        categorical_cols = []
        for col, stat in statistics.items():
            if stat.get("type") != "categorical" or col == time_col:
                continue
            if df is not None and col in df.columns:
                non_null = df[col].dropna()
                distinct = non_null.nunique()
                if distinct > MAX_GROUPED_BAR_DISTINCT or (
                    len(non_null) and distinct / len(non_null) > MAX_CATEGORY_DISTINCT_RATIO
                ):
                    continue
            categorical_cols.append(col)

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

    # ── Dedicated builders (previously approximated, or absent) ──────────────

    def _numeric_columns(self, df: pd.DataFrame, mining: dict[str, Any]) -> list[str]:
        """
        Numeric columns worth plotting, in dataset order.

        Identifier columns are dropped by name. Note this deliberately does not
        reuse the ``uniqueness_pct < 90`` filter the histogram selector applies:
        continuous measurements are near-unique too, so that rule would discard
        exactly the columns a scatter or pairplot is for.
        """
        statistics = mining.get("statistics") or {}
        return [
            col
            for col, stat in statistics.items()
            if stat.get("type") == "numeric"
            and col in df.columns
            and not self._is_identifier_name(col)
        ]

    @staticmethod
    def _is_identifier_name(column: str) -> bool:
        name = str(column).strip().lower()
        return name in _ID_NAME_EXACT or name.endswith(("_id", "_key", "_uuid"))

    @staticmethod
    def _five_number(series: pd.Series) -> dict[str, float]:
        """Five-number summary, rounded only as far as display needs."""
        return {
            "min": round(float(series.min()), 6),
            "q1": round(float(series.quantile(0.25)), 6),
            "median": round(float(series.median()), 6),
            "q3": round(float(series.quantile(0.75)), 6),
            "max": round(float(series.max()), 6),
        }

    def _class_labels(self, df: pd.DataFrame, target: str) -> list[Any]:
        """The target's most populated classes, capped for readability."""
        counts = df[target].dropna().value_counts()
        counts = counts[counts >= MIN_ROWS_PER_CLASS]
        return list(counts.head(MAX_CLASS_SERIES).index)

    def _grouped_bar_spec(
        self, df: pd.DataFrame, mining: dict[str, Any], target: str | None
    ) -> dict[str, Any] | None:
        """
        A categorical feature cross-tabulated against the target class.

        This is the chart a classification goal actually asks for: how the
        feature's categories are distributed *across the outcome*. The plain
        frequency bar it used to fall back to shows the categories but not the
        outcome, which is the entire question.
        """
        if not target or target not in df.columns:
            return None
        # A groupable feature has few enough categories to compare by eye.
        # The exclusions matter in practice: an uploaded CSV leaves the date
        # column as text, so the profiler types it categorical and it would
        # otherwise be picked — a grouped bar over hundreds of timestamps,
        # truncated to its busiest few, shows nothing.
        statistics = mining.get("statistics") or {}
        time_axis = self._datetime_series(df)
        time_col = time_axis[0] if time_axis else None
        candidates = [
            col
            for col, stat in statistics.items()
            if stat.get("type") == "categorical"
            and col != target
            and col != time_col
            and col in df.columns
            and 2 <= df[col].nunique(dropna=True) <= MAX_GROUPED_BAR_DISTINCT
        ]
        if not candidates:
            return None

        # Fewest categories first — the most readable comparison.
        feature = min(candidates, key=lambda c: df[c].nunique(dropna=True))
        pair = df[[feature, target]].dropna()
        if pair.empty:
            return None
        table = pd.crosstab(pair[feature], pair[target])
        if table.empty or table.shape[1] < 2:
            return None

        # Keep the busiest categories and classes, then restore a stable order.
        top_categories = table.sum(axis=1).sort_values(ascending=False).index[:MAX_GROUPED_BAR_CATEGORIES]
        top_classes = table.sum(axis=0).sort_values(ascending=False).index[:MAX_CLASS_SERIES]
        table = table.loc[top_categories, top_classes].sort_index()

        return {
            "type": "grouped_bar",
            "title": f"'{feature}' by '{target}'",
            "x_label": feature,
            "group_label": target,
            "categories": [str(c) for c in table.index],
            "series": [
                {"name": str(cls), "values": [int(v) for v in table[cls]]} for cls in table.columns
            ],
        }

    def _box_by_class_spec(
        self, df: pd.DataFrame, mining: dict[str, Any], target: str | None
    ) -> dict[str, Any] | None:
        """
        One box per target class for a single numeric column.

        The column is chosen by how far the class medians separate, scaled by
        the column's own spread — the column that best distinguishes the
        classes is the one worth the chart slot. The old fallback boxed a
        column without splitting by class at all, which cannot show separation
        even when it exists.
        """
        if not target or target not in df.columns:
            return None
        numeric_cols = [c for c in self._numeric_columns(df, mining) if c != target]
        if not numeric_cols:
            return None
        classes = self._class_labels(df, target)
        if len(classes) < 2:
            return None

        best_col: str | None = None
        best_separation = -1.0
        for col in numeric_cols:
            medians = [
                float(m)
                for m in (df.loc[df[target] == cls, col].dropna().median() for cls in classes)
                if not pd.isna(m)
            ]
            if len(medians) < 2:
                continue
            scale = float(df[col].dropna().std() or 0.0) or 1.0
            separation = (max(medians) - min(medians)) / scale
            if separation > best_separation:
                best_col, best_separation = col, separation
        if best_col is None:
            return None

        groups: list[dict[str, Any]] = []
        for cls in classes:
            series = df.loc[df[target] == cls, best_col].dropna()
            if len(series) < MIN_ROWS_PER_CLASS:
                continue
            groups.append({"label": str(cls), "count": int(len(series)), **self._five_number(series)})
        if len(groups) < 2:
            return None

        return {
            "type": "box_by_class",
            "title": f"'{best_col}' by '{target}'",
            "column": best_col,
            "target": target,
            "groups": groups,
        }

    def _pairplot_spec(self, df: pd.DataFrame, mining: dict[str, Any]) -> dict[str, Any] | None:
        """
        Every pairwise scatter over the top numeric columns.

        A clustering goal asks for a pairplot to see structure across *several*
        projections; the single cluster scatter it used to fall back to shows
        one. Panels are capped at ``MAX_PAIRPLOT_COLUMNS`` columns (6 panels).
        """
        numeric_cols = self._numeric_columns(df, mining)[:MAX_PAIRPLOT_COLUMNS]
        if len(numeric_cols) < 2:
            return None

        pairs: list[dict[str, Any]] = []
        for x_col, y_col in combinations(numeric_cols, 2):
            pair = df[[x_col, y_col]].dropna()
            if len(pair) < 2:
                continue
            r = pair[x_col].corr(pair[y_col])
            sample = pair.head(MAX_PAIRPLOT_POINTS)
            pairs.append(
                {
                    "x_label": x_col,
                    "y_label": y_col,
                    "r": 0.0 if pd.isna(r) else round(float(r), 3),
                    "points": [
                        {"x": float(x), "y": float(y)}
                        for x, y in zip(sample[x_col], sample[y_col])
                    ],
                }
            )
        if not pairs:
            return None

        return {
            "type": "pairplot",
            "title": f"Pairwise Relationships ({len(pairs)} panels)",
            "columns": numeric_cols,
            "pairs": pairs,
        }

    def _highlighted_scatter_spec(
        self, df: pd.DataFrame, mining: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        A scatter with the IQR-outlier rows flagged.

        For an anomaly goal the flagged points *are* the finding, so the plain
        scatter this used to fall back to omitted the only thing the chart was
        asked to show. Flagging is on the x column's IQR bounds — the same
        bounds MiningAgent reports — so the chart and the numbers agree.
        """
        outliers = mining.get("outliers") or {}
        if not outliers:
            return None
        numeric_cols = self._numeric_columns(df, mining)

        # x is the column with the most flagged rows; y is any other numeric.
        x_col = max(
            (c for c in outliers if c in numeric_cols),
            key=lambda c: outliers[c].get("count", 0),
            default=None,
        )
        if x_col is None:
            return None
        y_col = next((c for c in numeric_cols if c != x_col), None)
        if y_col is None:
            return None

        bounds = outliers[x_col].get("bounds") or []
        if len(bounds) != 2 or bounds[0] is None or bounds[1] is None:
            return None
        lower, upper = float(bounds[0]), float(bounds[1])

        pair = df[[x_col, y_col]].dropna()
        flagged_mask = (pair[x_col] < lower) | (pair[x_col] > upper)
        # Keep every flagged row when sampling — dropping the anomalies from an
        # anomaly chart would defeat it.
        keep = pd.concat([pair[flagged_mask], pair[~flagged_mask].head(MAX_SCATTER_POINTS)])
        if keep.empty:
            return None

        points = [
            {"x": float(x), "y": float(y), "outlier": bool(x < lower or x > upper)}
            for x, y in zip(keep[x_col], keep[y_col])
        ]
        highlighted = sum(1 for p in points if p["outlier"])
        if highlighted == 0:
            return None

        return {
            "type": "highlighted_scatter",
            "title": f"'{x_col}' vs '{y_col}' — {highlighted} outlier(s) highlighted",
            "x_label": x_col,
            "y_label": y_col,
            "highlighted_count": highlighted,
            "bounds": [round(lower, 6), round(upper, 6)],
            "points": points,
        }

    def _violin_spec(self, df: pd.DataFrame, mining: dict[str, Any]) -> dict[str, Any] | None:
        """
        A binned density profile plus the five-number summary.

        Emitted as bands rather than a kernel estimate: the counts are exactly
        the data, so the renderer draws a shape that can be checked against the
        frame, and there is no bandwidth parameter to justify in a viva.
        ``width`` is the count normalised to the modal band, which is all a
        renderer needs to mirror the shape about its axis.
        """
        for column in self._numeric_columns(df, mining):
            series = df[column].dropna()
            if len(series) < 2 or series.nunique() < 2:
                continue
            try:
                binned = pd.cut(series, bins=VIOLIN_BANDS, include_lowest=True).value_counts(sort=False)
            except Exception as exc:  # noqa: BLE001 - a bad column must not kill the chart set
                logger.warning("Violin binning failed for %s: %s", column, exc)
                continue

            peak = int(binned.max()) or 1
            bands = [
                {
                    "center": round((interval.left + interval.right) / 2, 6),
                    "count": int(count),
                    "width": round(int(count) / peak, 4),
                }
                for interval, count in binned.items()
            ]
            return {
                "type": "violin",
                "title": f"Distribution shape of '{column}'",
                "column": column,
                "bands": bands,
                **self._five_number(series),
            }
        return None

    def _datetime_series(self, df: pd.DataFrame) -> tuple[str, pd.Series] | None:
        """
        The dataset's time axis, if it has one.

        Parsed datetime columns are taken as-is. Text columns are *also*
        considered, because that is what a CSV upload actually delivers — the
        ingestion layer does not infer date dtypes, so a datetime64-only check
        would mean the trend chart never fires outside hand-built test frames.

        The guard against reading category codes as dates is a minimum string
        length: "2026-01-04" and "04/01/2026" clear it, "3" does not.
        """
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col, df[col]

        for col in df.columns:
            series = df[col].dropna()
            if series.empty or not (
                pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
            ):
                continue
            text = series.astype(str)
            if float(text.str.len().median()) < _MIN_DATE_TEXT_LENGTH:
                continue
            try:
                parsed = pd.to_datetime(text, errors="coerce", format="mixed")
            except Exception:  # noqa: BLE001 - an unparseable column is simply not the axis
                continue
            if parsed.notna().mean() < _MIN_DATE_PARSE_RATIO or parsed.nunique() < 3:
                continue
            return col, parsed.reindex(df.index)
        return None

    def _line_spec(self, df: pd.DataFrame, mining: dict[str, Any]) -> dict[str, Any] | None:
        """
        A numeric column's trend over the dataset's datetime column.

        Only built when a genuine time axis exists — inventing one from row
        order would draw a trend that is not in the data. Repeated timestamps
        are averaged so the line stays single-valued.
        """
        found = self._datetime_series(df)
        if found is None:
            return None
        x_col, x_values = found
        y_col = next((c for c in self._numeric_columns(df, mining) if c != x_col), None)
        if y_col is None:
            return None

        pair = pd.DataFrame({x_col: x_values, y_col: df[y_col]}).dropna()
        if len(pair) < 2:
            return None
        trend = pair.groupby(x_col)[y_col].mean().sort_index()
        if len(trend) < 2:
            return None
        if len(trend) > MAX_LINE_POINTS:
            step = len(trend) // MAX_LINE_POINTS + 1
            trend = trend.iloc[::step]

        return {
            "type": "line",
            "title": f"'{y_col}' over '{x_col}'",
            "x_label": x_col,
            "y_label": y_col,
            "points": [
                {"x": pd.Timestamp(idx).isoformat(), "y": round(float(value), 6)}
                for idx, value in trend.items()
            ],
        }

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
