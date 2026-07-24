"""
agents/qa_agent.py
────────────────────
QAAgent — direct, computed answers to specific factual questions about the
ingested dataset.

"Which column has the most missing values?", "what imputation should I use?",
"what's the correlation between price and revenue?" — these have exact
answers sitting in MiningAgent's already-computed output. Routing them
through a full RAG-recommendation pass (retrieve methodology chunks, dump
them as "recommendations") is why the chat felt like it was ignoring the
actual question. This agent pattern-matches the question and answers it
directly from computed data — no retrieval, no re-analysis.

Not a general NLU system — a bounded set of question shapes recognized via
regex. Returns None (falls through to RecommendationAgent's RAG path) for
anything broader, like "what should I investigate about this dataset."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class QAAnswer:
    text: str
    # Knowledge-base source path to cite alongside the computed answer, if
    # the question touches on a methodology (e.g. imputation strategy).
    source: str | None = None


def _columns_mentioned(goal_lower: str, columns: list[str]) -> list[str]:
    """Column names that appear as whole words in the goal text, longest first
    (so 'monthly_revenue' matches before a coincidental 'revenue' substring)."""
    found = [c for c in columns if c and re.search(rf"(?<!\w){re.escape(c.lower())}(?!\w)", goal_lower)]
    return sorted(found, key=len, reverse=True)


class QAAgent:
    def try_answer(
        self,
        goal: str,
        mining_output: dict[str, Any] | None,
        ingestion_output: dict[str, Any] | None,
    ) -> QAAnswer | None:
        mining_output = mining_output or {}
        ingestion_output = ingestion_output or {}
        goal_lower = goal.lower().strip()

        data_quality: dict[str, Any] = mining_output.get("data_quality") or {}
        statistics: dict[str, Any] = mining_output.get("statistics") or {}
        outliers: dict[str, Any] = mining_output.get("outliers") or {}
        correlations: dict[str, Any] = mining_output.get("correlations") or {}
        clustering = mining_output.get("clustering")
        feature_importance = mining_output.get("feature_importance") or []
        columns = list(statistics.keys()) or list(data_quality.keys())

        if not columns and "row_count" not in ingestion_output:
            return None  # nothing computed yet to answer from

        handlers = [
            self._row_column_count,
            self._most_missing,
            self._least_missing,
            self._imputation_advice,
            self._correlation_between,
            self._outliers_for_column,
            self._which_columns_have_outliers,
            self._numeric_vs_categorical,
            self._cluster_summary,
            self._top_feature,
            self._column_summary_stat,
        ]
        for handler in handlers:
            answer = handler(
                goal_lower, columns, data_quality, statistics, outliers,
                correlations, clustering, feature_importance, ingestion_output,
            )
            if answer is not None:
                return answer
        return None

    # ── Handlers (each returns None if its pattern doesn't match) ─────────

    def _row_column_count(self, goal_lower, columns, data_quality, statistics, outliers, correlations, clustering, feature_importance, ingestion_output) -> QAAnswer | None:
        if not re.search(r"how many (rows|records|columns|features|fields)", goal_lower):
            return None
        row_count = ingestion_output.get("row_count")
        column_count = ingestion_output.get("column_count", len(columns))
        if "column" in goal_lower or "feature" in goal_lower or "field" in goal_lower:
            return QAAnswer(f"This dataset has {column_count} column(s): {', '.join(columns) or 'none computed yet'}.")
        if row_count is not None:
            return QAAnswer(f"This dataset has {row_count} row(s) across {column_count} column(s).")
        return None

    def _most_missing(self, goal_lower, columns, data_quality, statistics, outliers, correlations, clustering, feature_importance, ingestion_output) -> QAAnswer | None:
        if not data_quality:
            return None
        if not re.search(r"(most|highest|largest|worst).*missing", goal_lower) and not re.search(r"missing.*(most|highest|largest|worst)", goal_lower):
            return None
        ranked = sorted(data_quality.items(), key=lambda kv: kv[1].get("missing_count", 0), reverse=True)
        top_col, top_info = ranked[0]
        if top_info.get("missing_count", 0) == 0:
            return QAAnswer("No column has any missing values in this dataset — completeness is 100% across the board.")
        return QAAnswer(
            f"'{top_col}' has the most missing values: {top_info['missing_count']} missing "
            f"({100 - top_info.get('completeness_pct', 0):.1f}% of rows)."
        )

    def _least_missing(self, goal_lower, columns, data_quality, statistics, outliers, correlations, clustering, feature_importance, ingestion_output) -> QAAnswer | None:
        if not data_quality:
            return None
        if not re.search(r"(least|fewest|lowest).*missing", goal_lower):
            return None
        ranked = sorted(data_quality.items(), key=lambda kv: kv[1].get("missing_count", 0))
        top_col, top_info = ranked[0]
        return QAAnswer(f"'{top_col}' has the fewest missing values: {top_info.get('missing_count', 0)} missing.")

    def _imputation_advice(self, goal_lower, columns, data_quality, statistics, outliers, correlations, clustering, feature_importance, ingestion_output) -> QAAnswer | None:
        if not re.search(r"imputat|how (should|do|can) i (handle|deal with|fix|treat).*missing", goal_lower):
            return None

        mentioned = _columns_mentioned(goal_lower, columns)
        targets = mentioned or [c for c, q in data_quality.items() if q.get("missing_count", 0) > 0]
        if not targets:
            return QAAnswer(
                "No columns have missing values in this dataset, so no imputation is needed.",
                source="knowledge_base/missing_values.md",
            )

        lines = []
        for col in targets[:5]:
            stat = statistics.get(col, {})
            missing = data_quality.get(col, {}).get("missing_count", 0)
            if missing == 0 and col in mentioned:
                lines.append(f"'{col}' has no missing values.")
                continue
            if stat.get("type") == "numeric":
                skew = stat.get("skew")
                if skew is not None and abs(skew) > 1:
                    lines.append(f"'{col}' ({missing} missing): skewed (skew={skew:.2f}) — use median imputation, not mean.")
                else:
                    lines.append(f"'{col}' ({missing} missing): roughly symmetric — mean imputation is reasonable.")
            elif stat.get("type") == "categorical":
                lines.append(f"'{col}' ({missing} missing): categorical — use mode imputation or an explicit 'Unknown' category.")
            else:
                lines.append(f"'{col}': {missing} missing value(s).")

        return QAAnswer(" ".join(lines), source="knowledge_base/missing_values.md")

    def _correlation_between(self, goal_lower, columns, data_quality, statistics, outliers, correlations, clustering, feature_importance, ingestion_output) -> QAAnswer | None:
        if "correlat" not in goal_lower or not correlations:
            return None
        mentioned = _columns_mentioned(goal_lower, list(correlations.keys()))
        if len(mentioned) < 2:
            return None
        col_a, col_b = mentioned[0], mentioned[1]
        r = correlations.get(col_a, {}).get(col_b)
        if r is None:
            return QAAnswer(f"I don't have a computed correlation between '{col_a}' and '{col_b}'.")
        strength = (
            "very strong" if abs(r) >= 0.7 else
            "strong" if abs(r) >= 0.5 else
            "moderate" if abs(r) >= 0.3 else
            "weak" if abs(r) >= 0.1 else "negligible"
        )
        direction = "positive" if r >= 0 else "negative"
        return QAAnswer(f"The correlation between '{col_a}' and '{col_b}' is r={r:.2f} — a {strength} {direction} relationship.")

    def _outliers_for_column(self, goal_lower, columns, data_quality, statistics, outliers, correlations, clustering, feature_importance, ingestion_output) -> QAAnswer | None:
        if "outlier" not in goal_lower:
            return None
        mentioned = _columns_mentioned(goal_lower, columns)
        if not mentioned:
            return None
        col = mentioned[0]
        info = outliers.get(col)
        if info is None:
            return QAAnswer(f"No outliers were flagged in '{col}' (IQR method).")
        return QAAnswer(
            f"'{col}' has {info['count']} outlier(s) ({info.get('pct', 0)}% of rows), "
            f"flagged via IQR outside [{info['bounds'][0]:.2f}, {info['bounds'][1]:.2f}]."
        )

    def _which_columns_have_outliers(self, goal_lower, columns, data_quality, statistics, outliers, correlations, clustering, feature_importance, ingestion_output) -> QAAnswer | None:
        if not re.search(r"which (columns?|features?) (have|has|contain).*outliers?", goal_lower) and not re.search(r"outliers?.*which columns?", goal_lower):
            return None
        if not outliers:
            return QAAnswer("No columns have flagged outliers (IQR method) in this dataset.")
        parts = [f"'{col}' ({info['count']})" for col, info in outliers.items()]
        return QAAnswer(f"Columns with flagged outliers: {', '.join(parts)}.")

    def _numeric_vs_categorical(self, goal_lower, columns, data_quality, statistics, outliers, correlations, clustering, feature_importance, ingestion_output) -> QAAnswer | None:
        wants_numeric = re.search(r"which (columns?|features?) (are|is).*numeric", goal_lower)
        wants_categorical = re.search(r"which (columns?|features?) (are|is).*categorical", goal_lower)
        if not wants_numeric and not wants_categorical:
            return None
        kind = "numeric" if wants_numeric else "categorical"
        matches = [c for c, s in statistics.items() if s.get("type") == kind]
        if not matches:
            return QAAnswer(f"No {kind} columns were found in this dataset.")
        return QAAnswer(f"{kind.title()} columns: {', '.join(matches)}.")

    def _cluster_summary(self, goal_lower, columns, data_quality, statistics, outliers, correlations, clustering, feature_importance, ingestion_output) -> QAAnswer | None:
        if not re.search(r"how many (clusters|segments|groups)", goal_lower):
            return None
        if not clustering:
            return QAAnswer("Clustering wasn't run or didn't find a meaningful structure for this dataset.")
        sizes = ", ".join(f"cluster {i}: {n} rows" for i, n in enumerate(clustering["cluster_sizes"]))
        return QAAnswer(
            f"The data separates into {clustering['k']} clusters "
            f"(silhouette score {clustering['silhouette_score']}). Sizes — {sizes}.",
            source="knowledge_base/clustering.md",
        )

    def _top_feature(self, goal_lower, columns, data_quality, statistics, outliers, correlations, clustering, feature_importance, ingestion_output) -> QAAnswer | None:
        if not re.search(r"(most|which).*(important|impactful)|feature importance", goal_lower):
            return None
        if not feature_importance:
            return QAAnswer("Feature importance wasn't computed (need at least 2 numeric columns).")
        top = feature_importance[0]
        return QAAnswer(f"'{top['feature']}' has the highest feature importance (PCA loading score {top['score']}).")

    def _column_summary_stat(self, goal_lower, columns, data_quality, statistics, outliers, correlations, clustering, feature_importance, ingestion_output) -> QAAnswer | None:
        stat_word = next((w for w in ("mean", "median", "average", "min", "minimum", "max", "maximum", "std", "stdev") if w in goal_lower), None)
        if stat_word is None:
            return None
        mentioned = _columns_mentioned(goal_lower, columns)
        if not mentioned:
            return None
        col = mentioned[0]
        stat = statistics.get(col, {})
        if stat.get("type") != "numeric":
            return QAAnswer(f"'{col}' isn't numeric, so {stat_word} isn't defined for it.")
        key = {"average": "mean", "minimum": "min", "maximum": "max", "stdev": "std"}.get(stat_word, stat_word)
        value = stat.get(key)
        if value is None:
            return None
        return QAAnswer(f"The {key} of '{col}' is {value:.3g}.")
