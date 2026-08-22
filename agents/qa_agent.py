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

**Row lookup** is the one handler that reads the data itself rather than
MiningAgent's summary of it. "Tell me about Drake" is a question about the
*contents* of the table, and every statistic upstream describes its *shape*, so
nothing computed could answer it — and the knowledge base, which holds EDA
methodology, has nothing to say about an artist either. Asked that against a
Spotify dataset, the chat previously answered with a card about SHAP and LIME.

It is matched by looking for a cell value in the question rather than by
recognising a phrasing, which is what keeps it from being another list of
regexes to extend forever: any of "tell me about Drake", "who is Drake",
"Drake?" finds the same row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# How distinct a column's values must be before one of them can be said to name
# a single row. Below this the value identifies a *group* — "Colombia", "Pop" —
# and a group is a filter or an aggregate, which is a different question and
# one the dataset's Query tab already answers in SQL.
IDENTIFIER_UNIQUENESS = 0.9

# Shortest cell value worth matching against a question. Two characters occur
# in ordinary prose far too often to be evidence of anything.
MIN_ENTITY_CHARS = 3

# Longest phrase, in words, considered as a candidate name. Past this the
# scan costs more than the names it would find are worth.
MAX_ENTITY_WORDS = 6

# Fields listed for one row before the answer stops being readable.
MAX_ROW_FIELDS = 20

_NON_ALPHANUMERIC_RE = re.compile(r"[^a-z0-9]+")


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


def _normalize(text: str) -> str:
    """Lowercase, punctuation flattened to single spaces.

    Applied to both the question and the cell values so they meet on the same
    terms — "Ty Dolla $ign" in the data and "ty dolla $ign" as typed both
    become "ty dolla ign" and match.
    """
    return _NON_ALPHANUMERIC_RE.sub(" ", str(text).lower()).strip()


def _identifier_columns(df: pd.DataFrame) -> list[str]:
    """Text columns distinct enough that one value names one row."""
    row_count = len(df)
    if row_count == 0:
        return []

    identifiers: list[str] = []
    for column in df.columns:
        series = df[column]
        # Numbers and dates are excluded deliberately. A unique integer id is
        # not what anyone types when they ask about a record, and a bare year
        # in a question would match a date column constantly.
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        distinct = int(series.nunique(dropna=True))
        if distinct > 1 and distinct / row_count >= IDENTIFIER_UNIQUENESS:
            identifiers.append(column)
    return identifiers


def _candidate_names(goal_normalized: str) -> list[str]:
    """Every phrase in the question that could be a name, longest first.

    Longest first because the longest match is the right one: in "tell me about
    bad bunny", both "bad bunny" and "bunny" may be present in the data, and
    only one of them is what was asked about.
    """
    words = goal_normalized.split()
    grams: list[str] = []
    for size in range(min(MAX_ENTITY_WORDS, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            grams.append(" ".join(words[start : start + size]))
    return grams


def _display(value: Any) -> str:
    """One cell, as a reader wants to see it."""
    try:
        if value is None or pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass  # arrays and the like are never NA; fall through and format them

    # Checked before the numeric branch, so "2006" stored as text is not
    # reformatted into "2,006".
    if isinstance(value, str):
        return value
    # np.bool_ is not a subclass of bool, and a row pulled out of a
    # mixed-dtype frame keeps numpy's scalar types. Missing it here sent
    # booleans down the numeric branch and rendered True as "1".
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"

    # float() rather than isinstance, so numpy's scalar types are covered
    # without naming each of them.
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        # Four digits and under go ungrouped, so a year reads as 2006 rather
        # than 2,006. Nothing here knows which column is a year, and the
        # convention is right for small counts either way.
        return f"{number:.0f}" if abs(number) < 10000 else f"{number:,.0f}"
    # A fixed 2dp would round a small magnitude away to "0.00".
    return f"{number:.3g}" if abs(number) < 0.01 else f"{number:,.2f}"


class QAAgent:
    def try_answer(
        self,
        goal: str,
        mining_output: dict[str, Any] | None,
        ingestion_output: dict[str, Any] | None,
        dataframe: pd.DataFrame | None = None,
    ) -> QAAnswer | None:
        """Answer `goal` directly, or return None to fall through to RAG.

        `dataframe` is the ingested table, which IngestionAgent already leaves
        on the pipeline context. Only the row lookup uses it; every other
        handler reads MiningAgent's computed summary, and omitting it simply
        turns that one handler off.
        """
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
        # Nothing computed yet means nothing for these handlers to read; the
        # row lookup below needs only the data itself, so it still gets a turn.
        if columns or "row_count" in ingestion_output:
            for handler in handlers:
                answer = handler(
                    goal_lower, columns, data_quality, statistics, outliers,
                    correlations, clustering, feature_importance, ingestion_output,
                )
                if answer is not None:
                    return answer

        # Last, so that every established handler keeps priority. A question
        # about the shape of the data should be answered as one even if some
        # value in the table happens to appear in it.
        return self._row_lookup(goal, dataframe)

    # ── Row lookup ────────────────────────────────────────────────────────

    def _row_lookup(self, goal: str, df: pd.DataFrame | None) -> QAAnswer | None:
        """Answer a question about one record, found by name.

        Returns None unless a value from an identifier-like column appears in
        the question. That deliberately answers nothing for "which country has
        the most artists" — "country" is a group, not a record, and an
        aggregate over a group is a different question.
        """
        if df is None or len(df) == 0:
            return None

        identifiers = _identifier_columns(df)
        if not identifiers:
            return None

        goal_normalized = _normalize(goal)
        if not goal_normalized:
            return None
        names = _candidate_names(goal_normalized)

        best: tuple[int, str, int] | None = None  # (match length, column, row position)
        for column in identifiers:
            lookup: dict[str, int] = {}
            for position, raw in enumerate(df[column].to_numpy()):
                key = _normalize(raw)
                if len(key) >= MIN_ENTITY_CHARS:
                    lookup.setdefault(key, position)
            for name in names:  # longest first
                position = lookup.get(name)
                if position is not None:
                    if best is None or len(name) > best[0]:
                        best = (len(name), column, position)
                    break

        if best is None:
            return None

        _, column, position = best
        return QAAnswer(self._describe_row(df, position, column))

    @staticmethod
    def _describe_row(df: pd.DataFrame, position: int, column: str) -> str:
        """One row, as Markdown the chat card can render."""
        row = df.iloc[position]
        # Column names are stripped for display only. Headers routinely carry
        # stray whitespace (" Artist Type"), and rendering that back looks like
        # our mistake rather than the file's.
        heading = f"**{_display(row[column])}** — {str(column).strip()}, row {position + 1} of {len(df):,}."

        fields: list[str] = []
        remaining = [c for c in df.columns if c != column]
        for name in remaining[:MAX_ROW_FIELDS]:
            fields.append(f"- {str(name).strip()}: {_display(row[name])}")
        hidden = len(remaining) - len(fields)
        if hidden > 0:
            fields.append(f"- …and {hidden} more column(s).")

        return "\n".join([heading, "", *fields])

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
        # "imput" (not "imputat") so it also catches "impute"/"imputing", not
        # just "imputation"/"imputate" — a real user wrote "best way to
        # impute postal" and that fell through to the generic RAG path
        # entirely, since "impute" doesn't contain "imputat" as a substring.
        if not re.search(
            r"imput|(how|what) (should|do|can|to) i? ?(handle|deal with|fix|treat|do (about|with)).*missing"
            r"|(best|good|right) way to (handle|deal with|fix|treat|impute).*missing",
            goal_lower,
        ):
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
        stat_word = next(
            (
                w
                for w in (
                    "mean", "median", "average",
                    "min", "minimum", "lowest", "smallest",
                    "max", "maximum", "highest", "largest", "biggest",
                    "std", "stdev",
                )
                if w in goal_lower
            ),
            None,
        )
        if stat_word is None:
            return None
        mentioned = _columns_mentioned(goal_lower, columns)
        if not mentioned:
            return None
        col = mentioned[0]
        stat = statistics.get(col, {})
        if stat.get("type") != "numeric":
            return QAAnswer(f"'{col}' isn't numeric, so {stat_word} isn't defined for it.")
        key = {
            "average": "mean",
            "minimum": "min", "lowest": "min", "smallest": "min",
            "maximum": "max", "highest": "max", "largest": "max", "biggest": "max",
            "stdev": "std",
        }.get(stat_word, stat_word)
        value = stat.get(key)
        if value is None:
            return None
        # `_display` rather than "%.3g", which rendered a perfectly good median
        # of 53,308.9 as "5.33e+04" — a correct answer that looks broken. One
        # formatter for every number this agent returns.
        return QAAnswer(f"The {key} of '{col}' is {_display(value)}.")
