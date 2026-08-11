"""
agents/mining_agent.py
───────────────────────
MiningAgent — goal-conditioned statistical profiling and pattern discovery.

The set of computations run is **conditioned on the planner's directives**
(Module 2 → FR-02): the goal decides which statistics are computed, not just
which are surfaced. Two different goals on the same dataset therefore produce
different mining output — the same-dataset/different-goal behaviour the project
is evaluated on.

Computations, keyed by the directive tokens the PipelinePlanner emits:
    • ``correlation``                         → Pearson correlation matrix
    • ``iqr_outliers`` / ``distribution_tails`` → IQR outlier counts
    • ``isolation_forest``                    → Isolation Forest outlier detection
    • ``feature_importance``                  → PCA-loading feature ranking
    • ``kmeans`` / ``silhouette`` / ``standardize`` → KMeans (silhouette-selected k)
    • ``dbscan``                              → DBSCAN density clustering
    • ``class_balance``                       → target class distribution / imbalance
    • ``linearity_check``                     → feature↔target Pearson linearity

Per-column descriptive statistics and data-quality metrics are always
computed. When no directives are supplied (e.g. a direct ``MiningAgent().run``
call outside the orchestrator), the agent falls back to the full unconditioned
profile — correlation, IQR outliers, feature importance, and KMeans — so it is
useful standalone.

Human-readable pattern strings summarise whatever was computed; the
RecommendationAgent folds these into its RAG retrieval query so recommendations
are grounded in what was actually found.
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
# Attribution fits a model, so it needs more rows than a summary statistic does;
# below this the explanation describes noise rather than the target.
MIN_ROWS_FOR_ATTRIBUTION = 20
# Above this many distinct values a numeric target is read as continuous.
MAX_CLASSES_FOR_ATTRIBUTION = 20
MAX_CLUSTER_K = 6
MAX_SCATTER_POINTS = 500

# Identifier columns (order_id, row index) carry no distributional signal and
# skew distance/variance-based models (PCA, KMeans, DBSCAN, Isolation Forest),
# so they are excluded from those models. Detected by name or by a
# sequential-integer structure — NOT by uniqueness alone, since continuous
# measurements (revenue, price) are also near-unique but are real features.
_ID_NAME_EXACT = {"id", "index", "idx", "key", "uuid", "guid", "rowid", "row_id", "row_number", "sno"}

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
        computations = {str(c) for c in (directives.get("computations", []) or [])}
        target_column = directives.get("target_column")
        # No directive tokens → run the full unconditioned profile (standalone
        # use, and the reporting default). Tokens present → run only what the
        # goal's plan asked for.
        conditioned = bool(computations)
        logger.info(
            "MiningAgent.run() | goal=%r task_type=%s conditioned=%s computations=%s",
            context.get("goal"), task_type, conditioned, sorted(computations),
        )

        df: pd.DataFrame | None = context.get("dataframe")
        if df is None or df.empty:
            return self._empty_result("No ingested dataset available — profiling skipped.")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        def wants(*tokens: str) -> bool:
            """True if unconditioned (full default) or any token was requested."""
            return not conditioned or any(t in computations for t in tokens)

        # Always computed.
        statistics = self._compute_statistics(df, numeric_cols)
        data_quality = self._compute_data_quality(df)

        # Conditioned computations. Correlation is also needed by linearity.
        correlations = (
            self._compute_correlations(df, numeric_cols)
            if wants("correlation", "linearity_check") else {}
        )
        outliers = (
            self._compute_outliers(df, numeric_cols)
            if wants("iqr_outliers", "distribution_tails") else {}
        )
        feature_importance = (
            self._compute_feature_importance(df, numeric_cols)
            if wants("feature_importance") else []
        )
        clustering = (
            self._compute_clustering(df, numeric_cols)
            if wants("kmeans", "silhouette", "standardize") else None
        )

        # Opt-in computations — only when explicitly requested by the plan.
        isolation_forest = (
            self._compute_isolation_forest(df, numeric_cols)
            if "isolation_forest" in computations else {}
        )
        dbscan = (
            self._compute_dbscan(df, numeric_cols)
            if "dbscan" in computations else None
        )
        class_balance = (
            self._compute_class_balance(df, target_column)
            if "class_balance" in computations else {}
        )
        linearity = (
            self._compute_linearity(df, numeric_cols, target_column)
            if "linearity_check" in computations else []
        )
        feature_attribution = (
            self._compute_feature_attribution(df, numeric_cols, target_column)
            if "shap_attribution" in computations else {}
        )

        patterns = self._build_patterns(
            correlations=correlations,
            outliers=outliers,
            statistics=statistics,
            clustering=clustering,
            row_count=len(df),
            isolation_forest=isolation_forest,
            dbscan=dbscan,
            class_balance=class_balance,
            linearity=linearity,
            feature_attribution=feature_attribution,
        )

        return {
            "statistics": statistics,
            "data_quality": data_quality,
            "correlations": correlations,
            "outliers": outliers,
            "feature_importance": feature_importance,
            "clustering": clustering,
            "isolation_forest": isolation_forest,
            "dbscan": dbscan,
            "class_balance": class_balance,
            "linearity": linearity,
            "feature_attribution": feature_attribution,
            "patterns": patterns,
            "task_type": task_type,
            "computations_run": sorted(computations) if conditioned else ["<default profile>"],
            "message": f"Profiled {len(df)} rows across {len(df.columns)} columns.",
        }

    @staticmethod
    def _empty_result(message: str) -> dict[str, Any]:
        """Shape-stable empty result (all keys present) for no-data paths."""
        return {
            "statistics": {},
            "data_quality": {},
            "correlations": {},
            "outliers": {},
            "feature_importance": [],
            "clustering": None,
            "isolation_forest": {},
            "dbscan": None,
            "class_balance": {},
            "linearity": [],
            "feature_attribution": {},
            "patterns": [],
            "task_type": None,
            "computations_run": [],
            "message": message,
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

    # ── Model feature selection (drop identifier columns) ────────────────

    def _model_feature_columns(self, df: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
        """Numeric columns minus identifier columns, which skew distance/
        variance-based models (PCA, KMeans, DBSCAN, Isolation Forest)."""
        return [c for c in numeric_cols if not self._is_identifier_column(df, c)]

    def _is_identifier_column(self, df: pd.DataFrame, col: str) -> bool:
        """True if `col` is an identifier — by name (id/index/…/*_id) or by a
        strictly-increasing, fully-unique integer structure (a row id). Not
        based on uniqueness alone: continuous measurements are near-unique too."""
        n = len(df)
        if n == 0:
            return False
        name = str(col).strip().lower()
        if name in _ID_NAME_EXACT or name.endswith(("_id", "_key", "_uuid")):
            return True
        series = df[col].dropna()
        if len(series) == n and series.nunique() == n and pd.api.types.is_integer_dtype(df[col]):
            arr = series.to_numpy()
            if len(arr) > 1 and bool((arr[1:] > arr[:-1]).all()):  # strictly increasing → sequential id
                return True
        return False

    # ── Feature importance (unsupervised, PCA-based) ─────────────────────

    def _compute_feature_importance(self, df: pd.DataFrame, numeric_cols: list[str]) -> list[dict[str, Any]]:
        numeric_cols = self._model_feature_columns(df, numeric_cols)
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

    # ── Feature attribution (supervised, SHAP) ───────────────────────────

    def _compute_feature_attribution(
        self, df: pd.DataFrame, numeric_cols: list[str], target_column: str | None
    ) -> dict[str, Any]:
        """
        Per-feature attribution against the actual target (Objective 5, FR).

        Unlike ``_compute_feature_importance`` — which ranks features by PCA
        loading and so describes *variance*, not *the target* — this fits a
        small gradient-boosted tree to predict ``target_column`` and explains it
        with ``shap.TreeExplainer``. Mean absolute SHAP value per feature is
        reported, normalised to sum to 1 so the numbers read as shares of the
        explanation.

        Two honesty measures are built into the output:

        • ``method`` names what actually ran. If SHAP is unavailable or fails,
          this falls back to ``sklearn.inspection.permutation_importance`` and
          says so, rather than silently presenting one method's numbers under
          the other's name.
        • ``model_score`` is the fitted model's train R²/accuracy. Attributions
          from a model that cannot predict the target are not meaningful, and
          this is what lets a reader judge that rather than take the ranking on
          trust.

        Returns ``{}`` when there is no usable target, which is also what a
        non-supervised goal gets, since the directive is never issued there.
        """
        if not target_column or target_column not in df.columns:
            return {}

        features = [c for c in self._model_feature_columns(df, numeric_cols) if c != target_column]
        if len(features) < 1:
            return {}

        frame = df[[*features, target_column]].dropna()
        if len(frame) < MIN_ROWS_FOR_ATTRIBUTION:
            return {}

        X = frame[features]
        y = frame[target_column]

        # Discrete, low-cardinality targets are treated as classes; anything
        # else is regression. Mirrors how a practitioner would read the column.
        is_classification = (
            not pd.api.types.is_numeric_dtype(y) or y.nunique() <= MAX_CLASSES_FOR_ATTRIBUTION
        )
        if is_classification and y.nunique() < 2:
            return {}

        try:
            model, score = self._fit_attribution_model(X, y, is_classification)
        except Exception as exc:  # noqa: BLE001 - attribution is additive, never fatal
            logger.warning("Feature-attribution model fit failed: %s", exc)
            return {}

        scores, method = self._shap_values(model, X)
        if scores is None:
            scores, method = self._permutation_values(model, X, y)
        if scores is None:
            return {}

        total = float(np.sum(scores))
        if total <= 0:
            return {}

        ranked = sorted(
            zip(features, (float(s) / total for s in scores)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return {
            "target": target_column,
            "method": method,
            "task": "classification" if is_classification else "regression",
            "model": "GradientBoosting",
            "model_score": round(float(score), 4),
            "n_samples": int(len(frame)),
            "attributions": [
                {"feature": col, "score": round(share, 4)} for col, share in ranked
            ],
        }

    @staticmethod
    def _fit_attribution_model(X: pd.DataFrame, y: pd.Series, is_classification: bool):
        """Fit the small tree model SHAP explains, returning (model, train score)."""
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        # Deliberately small: the model exists to be explained, not deployed.
        # A shallow, few-estimator fit keeps this well inside the FR-05 budget.
        kwargs = {"n_estimators": 40, "max_depth": 3, "random_state": 42}
        model = (
            GradientBoostingClassifier(**kwargs)
            if is_classification
            else GradientBoostingRegressor(**kwargs)
        )
        model.fit(X, y)
        return model, model.score(X, y)

    @staticmethod
    def _shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray | None, str]:
        """Mean |SHAP value| per feature, or (None, "") if SHAP cannot run."""
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X)
            array = np.asarray(values)
            # Multiclass returns (n_samples, n_features, n_classes); average the
            # magnitude across classes so every feature gets one comparable number.
            if array.ndim == 3:
                array = np.abs(array).mean(axis=2)
            return np.abs(array).mean(axis=0), "shap.TreeExplainer"
        except Exception as exc:  # noqa: BLE001 - fall through to permutation
            logger.warning("SHAP attribution unavailable (%s); using permutation importance.", exc)
            return None, ""

    @staticmethod
    def _permutation_values(model, X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray | None, str]:
        """Fallback attribution when SHAP is absent — honestly labelled as such."""
        try:
            from sklearn.inspection import permutation_importance

            result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            # Permutation importance can go negative (a feature that actively
            # hurt the shuffled model); clip so shares stay interpretable.
            return np.clip(result.importances_mean, 0, None), "permutation_importance"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Permutation importance failed: %s", exc)
            return None, ""

    # ── Clustering (goal-agnostic, KMeans with silhouette-selected k) ────

    def _compute_clustering(self, df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any] | None:
        numeric_cols = self._model_feature_columns(df, numeric_cols)
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

    # ── Isolation Forest (multivariate outliers, anomaly goals) ──────────

    def _compute_isolation_forest(self, df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
        numeric_cols = self._model_feature_columns(df, numeric_cols)
        if len(numeric_cols) < 1:
            return {}
        matrix = df[numeric_cols].dropna()
        if len(matrix) < MIN_ROWS_FOR_CLUSTERING:
            return {}
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(matrix)
            model = IsolationForest(random_state=42, contamination="auto")
            preds = model.fit_predict(scaled)  # -1 = outlier, 1 = inlier
            n_outliers = int((preds == -1).sum())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Isolation Forest failed: %s", exc)
            return {}

        return {
            "method": "isolation_forest",
            "n_outliers": n_outliers,
            "pct": round(n_outliers / len(matrix) * 100, 1) if len(matrix) else 0.0,
            "n_samples": int(len(matrix)),
            "features": list(numeric_cols),
        }

    # ── DBSCAN (density clustering, clustering goals) ────────────────────

    def _compute_dbscan(self, df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any] | None:
        numeric_cols = self._model_feature_columns(df, numeric_cols)
        if len(numeric_cols) < MIN_NUMERIC_COLUMNS_FOR_PCA:
            return None
        matrix = df[numeric_cols].dropna()
        if len(matrix) < MIN_ROWS_FOR_CLUSTERING:
            return None
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(matrix)
            min_samples = min(5, len(matrix) - 1)
            # eps via the k-distance heuristic: the median distance to each
            # point's min_samples-th nearest neighbour is a robust default.
            nn = NearestNeighbors(n_neighbors=min_samples).fit(scaled)
            distances, _ = nn.kneighbors(scaled)
            eps = float(np.median(distances[:, -1]))
            if not np.isfinite(eps) or eps <= 0:
                eps = 0.5

            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
            unique = set(labels)
            n_clusters = len(unique - {-1})
            n_noise = int((labels == -1).sum())
        except Exception as exc:  # noqa: BLE001
            logger.warning("DBSCAN failed: %s", exc)
            return None

        return {
            "method": "dbscan",
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "eps": round(eps, 3),
            "min_samples": int(min_samples),
        }

    # ── Class balance (classification goals) ─────────────────────────────

    def _compute_class_balance(self, df: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
        if not target_column or target_column not in df.columns:
            return {}
        counts = df[target_column].value_counts(dropna=True)
        total = int(counts.sum())
        if total == 0:
            return {}
        distribution = [
            {"class": str(cls), "count": int(cnt), "pct": round(cnt / total * 100, 1)}
            for cls, cnt in counts.head(20).items()
        ]
        majority, minority = int(counts.max()), int(counts.min())
        return {
            "target": target_column,
            "n_classes": int(counts.nunique()),
            "distribution": distribution,
            "imbalance_ratio": round(majority / minority, 2) if minority > 0 else None,
        }

    # ── Linearity check (regression goals) ───────────────────────────────

    def _compute_linearity(
        self, df: pd.DataFrame, numeric_cols: list[str], target_column: str | None
    ) -> list[dict[str, Any]]:
        # Linearity is feature↔target Pearson correlation; needs a numeric target.
        if not target_column or target_column not in numeric_cols:
            return []
        result: list[dict[str, Any]] = []
        for col in numeric_cols:
            if col == target_column:
                continue
            pair = df[[col, target_column]].dropna()
            if len(pair) < 3:
                continue
            r = pair[col].corr(pair[target_column])
            if pd.isna(r):
                continue
            r = float(r)
            strength = "strong" if abs(r) >= 0.6 else "moderate" if abs(r) >= 0.3 else "weak"
            result.append({"feature": col, "pearson_r": round(r, 3), "linear_strength": strength})
        result.sort(key=lambda d: abs(d["pearson_r"]), reverse=True)
        return result

    # ── Pattern summaries (feed the RAG retrieval query) ─────────────────

    def _build_patterns(
        self,
        correlations: dict[str, Any],
        outliers: dict[str, Any],
        statistics: dict[str, Any],
        clustering: dict[str, Any] | None,
        row_count: int,
        isolation_forest: dict[str, Any] | None = None,
        dbscan: dict[str, Any] | None = None,
        class_balance: dict[str, Any] | None = None,
        linearity: list[dict[str, Any]] | None = None,
        feature_attribution: dict[str, Any] | None = None,
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

        if isolation_forest:
            patterns.append(
                f"Isolation Forest flagged {isolation_forest['n_outliers']} multivariate "
                f"anomaly(ies) ({isolation_forest['pct']}% of rows)."
            )

        if dbscan is not None:
            patterns.append(
                f"DBSCAN found {dbscan['n_clusters']} density-based cluster(s) "
                f"with {dbscan['n_noise']} noise point(s) (eps={dbscan['eps']})."
            )

        if class_balance:
            ratio = class_balance.get("imbalance_ratio")
            ratio_txt = f" (imbalance ratio {ratio}:1)" if ratio and ratio > 1.5 else ""
            patterns.append(
                f"Target '{class_balance['target']}' has {class_balance['n_classes']} "
                f"class(es){ratio_txt}."
            )

        if linearity:
            top = linearity[0]
            patterns.append(
                f"'{top['feature']}' has a {top['linear_strength']} linear relationship "
                f"with the target (r={top['pearson_r']})."
            )

        if feature_attribution and feature_attribution.get("attributions"):
            top = feature_attribution["attributions"][0]
            # The method and model score travel with the claim so the reader can
            # weigh it — an attribution from a model that fits poorly is weak
            # evidence, and hiding that would overstate the finding.
            patterns.append(
                f"'{top['feature']}' contributes most to predicting "
                f"'{feature_attribution['target']}' ({top['score']:.0%} of total attribution, "
                f"via {feature_attribution['method']}, model score "
                f"{feature_attribution['model_score']})."
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
