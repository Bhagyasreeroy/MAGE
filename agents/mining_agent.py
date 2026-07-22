"""
agents/mining_agent.py
───────────────────────
MiningAgent — statistical profiling and pattern discovery.

Consumes the canonical DataFrame that IngestionAgent places into the
shared pipeline context and computes:
    • Per-column descriptive statistics (numeric and categorical).
    • Data-quality metrics (completeness, uniqueness) per column.
    • A Pearson correlation matrix over numeric columns.
    • IQR-based outlier counts per numeric column.
    • PCA-based feature importance (|loading| on the first principal
      component) — an unsupervised, goal-agnostic ranking since no
      target column / predictive model is assumed.
    • KMeans clustering (silhouette-selected k) when there are enough
      numeric columns and rows for it to be meaningful.
    • A list of human-readable pattern strings, which the
      RecommendationAgent folds into its RAG retrieval query so
      recommendations are grounded in what was actually found.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Below this many numeric columns or rows, clustering/PCA are too
# under-determined to produce a meaningful result — skip rather than
# report a number that doesn't mean anything.
MIN_NUMERIC_COLUMNS_FOR_PCA = 2
MIN_ROWS_FOR_CLUSTERING = 10
MAX_CLUSTER_K = 6
MAX_SCATTER_POINTS = 500

# A correlation at or above this magnitude is called out as a pattern.
STRONG_CORRELATION_THRESHOLD = 0.6


class MiningAgent:
    """
    Performs statistical profiling and goal-conditioned pattern discovery.

    Input context keys consumed:
        - ``dataframe``     : canonical pandas DataFrame from IngestionAgent
        - ``goal``          : analytical goal (logged for traceability)

    Output keys produced:
        - ``statistics``         : descriptive stats per feature
        - ``data_quality``       : completeness/uniqueness per feature
        - ``correlations``       : pairwise Pearson correlation matrix
        - ``outliers``            : IQR outlier counts per numeric feature
        - ``feature_importance`` : PCA-loading-based feature ranking
        - ``clustering``          : KMeans result, or None if not applicable
        - ``patterns``            : list of human-readable insight strings
    """

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        directives = context.get("directives", {}) or {}
        task_type = directives.get("task_type") or context.get("task_type")
        computations = directives.get("computations", [])
        logger.info(
            "MiningAgent.run() | goal=%r task_type=%s computations=%s",
            context.get("goal"), task_type, computations,
        )

        df: pd.DataFrame | None = context.get("dataframe")
        if df is None or df.empty:
            return {
                "statistics": {},
                "data_quality": {},
                "correlations": {},
                "outliers": {},
                "feature_importance": [],
                "clustering": None,
                "patterns": [],
                "message": "No ingested dataset available — profiling skipped.",
            }

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        statistics = self._compute_statistics(df, numeric_cols)
        data_quality = self._compute_data_quality(df)
        correlations = self._compute_correlations(df, numeric_cols)
        outliers = self._compute_outliers(df, numeric_cols)
        feature_importance = self._compute_feature_importance(df, numeric_cols)
        clustering = self._compute_clustering(df, numeric_cols)

        patterns = self._build_patterns(
            correlations=correlations,
            outliers=outliers,
            statistics=statistics,
            clustering=clustering,
            row_count=len(df),
        )

        return {
            "statistics": statistics,
            "data_quality": data_quality,
            "correlations": correlations,
            "outliers": outliers,
            "feature_importance": feature_importance,
            "clustering": clustering,
            "patterns": patterns,
            "message": f"Profiled {len(df)} rows across {len(df.columns)} columns.",
        }

    # ── Statistics ────────────────────────────────────────────────────────

    def _compute_statistics(self, df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for col in df.columns:
            series = df[col]
            if col in numeric_cols:
                non_null = series.dropna()
                stats[col] = {
                    "type": "numeric",
                    "mean": self._safe_float(non_null.mean()),
                    "median": self._safe_float(non_null.median()),
                    "std": self._safe_float(non_null.std()),
                    "min": self._safe_float(non_null.min()),
                    "max": self._safe_float(non_null.max()),
                    "q1": self._safe_float(non_null.quantile(0.25)),
                    "q3": self._safe_float(non_null.quantile(0.75)),
                    "skew": self._safe_float(non_null.skew()) if len(non_null) > 2 else None,
                }
            else:
                value_counts = series.value_counts().head(5)
                stats[col] = {
                    "type": "categorical",
                    "cardinality": int(series.nunique()),
                    "top_values": [
                        {"value": str(v), "count": int(c)} for v, c in value_counts.items()
                    ],
                }
        return stats

    def _compute_data_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        row_count = len(df)
        quality: dict[str, Any] = {}
        for col in df.columns:
            series = df[col]
            missing = int(series.isna().sum())
            completeness_pct = round((1 - missing / row_count) * 100, 1) if row_count else 0.0
            uniqueness_pct = round((series.nunique() / row_count) * 100, 1) if row_count else 0.0
            quality[col] = {
                "completeness_pct": completeness_pct,
                "uniqueness_pct": uniqueness_pct,
                "missing_count": missing,
            }
        return quality

    # ── Correlation ───────────────────────────────────────────────────────

    def _compute_correlations(self, df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
        if len(numeric_cols) < 2:
            return {}
        corr = df[numeric_cols].corr(method="pearson")
        return {
            col: {other: self._safe_float(corr.loc[col, other]) for other in numeric_cols}
            for col in numeric_cols
        }

    # ── Outliers (IQR) ───────────────────────────────────────────────────

    def _compute_outliers(self, df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
        outliers: dict[str, Any] = {}
        row_count = len(df)
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flagged = series[(series < lower) | (series > upper)]
            if len(flagged) == 0:
                continue
            outliers[col] = {
                "count": int(len(flagged)),
                "pct": round(len(flagged) / row_count * 100, 1) if row_count else 0.0,
                "method": "iqr",
                "bounds": [self._safe_float(lower), self._safe_float(upper)],
            }
        return outliers

    # ── Feature importance (unsupervised, PCA-based) ─────────────────────

    def _compute_feature_importance(self, df: pd.DataFrame, numeric_cols: list[str]) -> list[dict[str, Any]]:
        if len(numeric_cols) < MIN_NUMERIC_COLUMNS_FOR_PCA:
            return []

        matrix = df[numeric_cols].dropna()
        if len(matrix) < MIN_NUMERIC_COLUMNS_FOR_PCA:
            return []

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(matrix)
            pca = PCA(n_components=1, random_state=42)
            pca.fit(scaled)
            loadings = np.abs(pca.components_[0])
            total = loadings.sum()
            if total == 0:
                return []
            scores = loadings / total
        except Exception as exc:  # noqa: BLE001
            logger.warning("PCA feature importance failed: %s", exc)
            return []

        ranked = sorted(
            zip(numeric_cols, scores), key=lambda pair: pair[1], reverse=True
        )
        return [{"feature": col, "score": round(float(score), 4)} for col, score in ranked]

    # ── Clustering (goal-agnostic, KMeans with silhouette-selected k) ────

    def _compute_clustering(self, df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any] | None:
        if len(numeric_cols) < MIN_NUMERIC_COLUMNS_FOR_PCA:
            return None

        matrix = df[numeric_cols].dropna()
        if len(matrix) < MIN_ROWS_FOR_CLUSTERING:
            return None

        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(matrix)

            max_k = min(MAX_CLUSTER_K, len(matrix) - 1)
            best: dict[str, Any] | None = None
            for k in range(2, max_k + 1):
                labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scaled)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(scaled, labels)
                if best is None or score > best["score"]:
                    best = {"k": k, "labels": labels, "score": score}

            if best is None:
                return None

            projection = PCA(n_components=2, random_state=42).fit_transform(scaled)
            points = [
                {"x": self._safe_float(x), "y": self._safe_float(y), "cluster": int(c)}
                for (x, y), c in zip(projection[:MAX_SCATTER_POINTS], best["labels"][:MAX_SCATTER_POINTS])
            ]
            cluster_sizes = [int((best["labels"] == c).sum()) for c in range(best["k"])]

            return {
                "k": best["k"],
                "silhouette_score": round(float(best["score"]), 3),
                "cluster_sizes": cluster_sizes,
                "points": points,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Clustering failed: %s", exc)
            return None

    # ── Pattern summaries (feed the RAG retrieval query) ─────────────────

    def _build_patterns(
        self,
        correlations: dict[str, Any],
        outliers: dict[str, Any],
        statistics: dict[str, Any],
        clustering: dict[str, Any] | None,
        row_count: int,
    ) -> list[str]:
        patterns: list[str] = []

        seen_pairs: set[frozenset[str]] = set()
        for col, row in correlations.items():
            for other, r in row.items():
                if other == col or r is None:
                    continue
                pair = frozenset((col, other))
                if pair in seen_pairs or abs(r) < STRONG_CORRELATION_THRESHOLD:
                    continue
                seen_pairs.add(pair)
                direction = "positively" if r > 0 else "negatively"
                patterns.append(f"'{col}' and '{other}' are strongly {direction} correlated (r={r:.2f}).")

        for col, info in outliers.items():
            patterns.append(f"{info['count']} outlier(s) detected in '{col}' via IQR ({info['pct']}% of rows).")

        for col, stat in statistics.items():
            if stat.get("type") == "numeric" and stat.get("skew") is not None and abs(stat["skew"]) > 1:
                patterns.append(f"'{col}' is strongly skewed (skew={stat['skew']:.2f}).")

        if clustering is not None:
            patterns.append(
                f"Data separates into {clustering['k']} clusters "
                f"(silhouette score={clustering['silhouette_score']})."
            )

        return patterns

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert to float, mapping NaN/inf to None (invalid JSON otherwise)."""
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if np.isfinite(f) else None
