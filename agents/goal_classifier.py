"""
agents/goal_classifier.py
──────────────────────────
GoalClassifier — turns a natural-language analytical goal into a structured
task-type classification that drives the orchestrator's conditional pipeline
(Module 2).

Design: goal classification sits behind a *swappable provider* interface, as
the proposal describes ("Claude or GPT-4o, abstracted behind a swappable
interface"). The default providers are **local and free** — no API key, no
network, no cost:

    1. RuleBasedProvider    — keyword + column-schema heuristics. Fast,
                              deterministic, handles the clear-cut goals.
    2. EmbeddingProvider    — zero-shot cosine similarity against task-type
                              prototypes, reusing the sentence-transformers
                              model already loaded for the RAG layer. Handles
                              the ambiguous goals the rules can't resolve.

An LLM-backed provider can be plugged in behind the same interface if an API
key is ever configured, but it is never required. When nothing is confident,
the classifier falls back to ``reporting`` (the goal-agnostic default).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from backend.schemas.analysis import GoalClassification, TaskType

logger = logging.getLogger(__name__)

# Below this confidence a provider is treated as inconclusive and the next
# provider in the chain is consulted.
INCONCLUSIVE_BELOW = 0.5

# Final fallback when no provider is confident about anything.
DEFAULT_TASK_TYPE = TaskType.reporting


# ── Keyword lexicon ──────────────────────────────────────────────────────────
# (keyword, weight) per task type. Matched case-insensitively as whole words /
# phrases against the goal text. Higher weight = stronger signal.
_KEYWORDS: dict[TaskType, list[tuple[str, float]]] = {
    TaskType.anomaly_detection: [
        ("anomaly", 3.0), ("anomalies", 3.0), ("outlier", 3.0), ("outliers", 3.0),
        ("unusual", 2.0), ("abnormal", 2.0), ("suspicious", 2.0), ("novelty", 2.0),
        ("fraud detection", 3.0), ("detect fraud", 2.0), ("deviation", 1.5),
    ],
    TaskType.clustering: [
        ("cluster", 3.0), ("clustering", 3.0), ("segment", 3.0), ("segments", 3.0),
        ("segmentation", 3.0), ("cohort", 2.0), ("cohorts", 2.0), ("persona", 2.0),
        ("personas", 2.0), ("group similar", 2.5), ("natural groups", 2.5),
        ("unsupervised", 2.0), ("customer groups", 2.0),
    ],
    TaskType.regression: [
        ("forecast", 3.0), ("regression", 3.0), ("estimate", 2.0), ("how much", 2.0),
        ("project", 1.5), ("continuous", 1.5), ("revenue", 1.0), ("sales amount", 1.5),
        ("price", 1.0), ("predict revenue", 3.0), ("predict sales", 3.0),
        ("predict price", 3.0), ("predict the amount", 3.0),
    ],
    TaskType.classification: [
        ("classify", 3.0), ("classification", 3.0), ("churn", 2.5), ("predict churn", 3.0),
        ("propensity", 2.0), ("likelihood", 2.0), ("probability of", 2.0), ("label", 1.5),
        ("category", 1.5), ("will they", 1.5), ("predict whether", 2.5), ("spam", 2.0),
        ("convert", 1.5), ("default", 1.5),
    ],
    TaskType.reporting: [
        ("describe", 2.0), ("summarize", 2.0), ("summary", 2.0), ("overview", 2.0),
        ("explore", 2.0), ("understand", 1.5), ("profile", 2.0), ("distribution", 1.5),
        ("report", 2.0), ("what does the data", 1.5), ("general analysis", 2.0),
    ],
}

# Short natural-language prototypes per task type for the embedding fallback.
_PROTOTYPES: dict[TaskType, list[str]] = {
    TaskType.classification: [
        "predict a categorical label or class for each record",
        "classify records into discrete categories such as churn yes or no",
    ],
    TaskType.regression: [
        "predict or forecast a continuous numeric value",
        "estimate a quantity such as revenue, price, or demand",
    ],
    TaskType.clustering: [
        "group similar records into clusters or segments without labels",
        "discover natural customer segments or cohorts",
    ],
    TaskType.anomaly_detection: [
        "find anomalies, outliers, or unusual records in the data",
        "detect abnormal or suspicious observations",
    ],
    TaskType.reporting: [
        "describe, summarize, and profile the dataset in general",
        "explore the data and report distributions and quality",
    ],
}


def _find_target_column(goal: str, columns: list[str]) -> str | None:
    """Return a column name that appears in the goal text, if any."""
    goal_lower = goal.lower()
    # Prefer the longest matching column name (most specific).
    matches = [c for c in columns if c and c.lower() in goal_lower]
    return max(matches, key=len) if matches else None


def _extract_columns(dataset_schema: Any) -> list[str]:
    """Pull column names out of an IngestionResult / dict / None."""
    if dataset_schema is None:
        return []
    summary = getattr(dataset_schema, "column_summary", None)
    if summary is None and isinstance(dataset_schema, dict):
        summary = dataset_schema.get("column_summary")
    if not summary:
        return []
    names: list[str] = []
    for col in summary:
        name = getattr(col, "name", None)
        if name is None and isinstance(col, dict):
            name = col.get("name")
        if name:
            names.append(str(name))
    return names


class TaskClassifierProvider(Protocol):
    """A swappable strategy that classifies a goal into a task type."""

    def classify(self, goal: str, columns: list[str]) -> GoalClassification | None:
        """Return a classification, or None if inconclusive."""
        ...


class RuleBasedProvider:
    """Keyword + schema heuristic classifier. Deterministic, offline, instant."""

    def classify(self, goal: str, columns: list[str]) -> GoalClassification | None:
        goal_lower = goal.lower().strip()
        if not goal_lower:
            return None

        scores: dict[TaskType, float] = {}
        for task_type, keywords in _KEYWORDS.items():
            score = 0.0
            for phrase, weight in keywords:
                if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", goal_lower):
                    score += weight
            if score > 0:
                scores[task_type] = score

        if not scores:
            return None

        total = sum(scores.values())
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        # Confidence: blend a floor with the share of signal the winner holds.
        confidence = round(min(0.55 + 0.4 * (best_score / total), 0.97), 3)

        return GoalClassification(
            task_type=best_type,
            target_column=_find_target_column(goal, columns),
            confidence=confidence,
            rationale=f"Matched task-type keywords for '{best_type.value}' (rule-based).",
        )


class EmbeddingProvider:
    """
    Zero-shot classifier: embed the goal and each task-type prototype with the
    local sentence-transformers model, pick the highest cosine similarity.
    Free, local — same model the RAG layer already uses.
    """

    def classify(self, goal: str, columns: list[str]) -> GoalClassification | None:
        goal = goal.strip()
        if not goal:
            return None

        try:
            import numpy as np

            from rag.embeddings import embed_batch, embed_text
        except Exception as exc:  # pragma: no cover - only if deps missing
            logger.warning("EmbeddingProvider unavailable (%s); skipping.", exc)
            return None

        # Flatten prototypes, remembering which task type each belongs to.
        proto_types: list[TaskType] = []
        proto_texts: list[str] = []
        for task_type, texts in _PROTOTYPES.items():
            for text in texts:
                proto_types.append(task_type)
                proto_texts.append(text)

        goal_vec = embed_text(goal)
        proto_vecs = embed_batch(proto_texts)

        def _norm(v: "np.ndarray") -> "np.ndarray":
            n = np.linalg.norm(v)
            return v / n if n else v

        goal_unit = _norm(goal_vec)
        sims = np.array([float(np.dot(goal_unit, _norm(p))) for p in proto_vecs])

        # Best prototype per task type → the task type's similarity.
        per_type: dict[TaskType, float] = {}
        for task_type, sim in zip(proto_types, sims):
            per_type[task_type] = max(per_type.get(task_type, -1.0), sim)

        best_type = max(per_type, key=per_type.get)
        best_sim = per_type[best_type]
        # Cosine sims for MiniLM live roughly in [0.1, 0.7] for short text;
        # rescale into a calibrated-ish confidence and clamp.
        confidence = round(min(max(best_sim, 0.0), 1.0), 3)

        return GoalClassification(
            task_type=best_type,
            target_column=_find_target_column(goal, columns),
            confidence=confidence,
            rationale=f"Closest task-type prototype by embedding similarity ({best_sim:.2f}).",
        )


class GoalClassifier:
    """
    Classifies a natural-language goal into a TaskType by consulting a chain of
    providers in order, returning the first confident result and otherwise
    falling back to the goal-agnostic ``reporting`` task.
    """

    def __init__(self, providers: list[TaskClassifierProvider] | None = None) -> None:
        # Rules first (fast, confident on clear goals), embeddings as fallback
        # for the ambiguous ones. Both are local and free.
        self._providers: list[TaskClassifierProvider] = providers or [
            RuleBasedProvider(),
            EmbeddingProvider(),
        ]

    def classify(self, goal: str, dataset_schema: Any = None) -> GoalClassification:
        """
        Parameters
        ----------
        goal : str
            The user's natural-language analytical goal.
        dataset_schema : IngestionResult | dict | None
            Optional ingestion profile — used to guess a target column.

        Returns
        -------
        GoalClassification
            Always returns a result; defaults to ``reporting`` when nothing is
            confident.
        """
        columns = _extract_columns(dataset_schema)
        best: GoalClassification | None = None

        for provider in self._providers:
            try:
                result = provider.classify(goal, columns)
            except Exception as exc:  # noqa: BLE001 - a provider must never crash the pipeline
                logger.warning("Provider %s failed: %s", type(provider).__name__, exc)
                continue
            if result is None:
                continue
            if best is None or result.confidence > best.confidence:
                best = result
            if result.confidence >= INCONCLUSIVE_BELOW:
                logger.info(
                    "Goal classified as %s (confidence=%.2f) via %s",
                    result.task_type.value, result.confidence, type(provider).__name__,
                )
                return result

        if best is not None:
            return best

        return GoalClassification(
            task_type=DEFAULT_TASK_TYPE,
            target_column=_find_target_column(goal, columns),
            confidence=0.3,
            rationale="No confident task-type signal; defaulting to a general report.",
        )
