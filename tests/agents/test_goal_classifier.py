"""
tests/agents/test_goal_classifier.py
──────────────────────────────────────
Tests for the Module 2 GoalClassifier.

These use the RuleBasedProvider only (no embedding model download) so they run
fast and fully offline. Representative goals for each task type are asserted to
classify correctly, and ambiguous/empty goals fall back to reporting.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.goal_classifier import EmbeddingProvider, GoalClassifier, RuleBasedProvider
from backend.schemas.analysis import (
    ColumnStats,
    ColumnSummary,
    GoalClassification,
    IngestionResult,
    TaskType,
)


@pytest.fixture
def classifier() -> GoalClassifier:
    # Rule-based only — deterministic and offline.
    return GoalClassifier(providers=[RuleBasedProvider()])


class TestGoalClassifier:
    @pytest.mark.parametrize(
        "goal, expected",
        [
            ("Predict which customers will churn next quarter", TaskType.classification),
            ("Classify transactions as spam or not", TaskType.classification),
            # Discriminative / binary-contrast goals that carry no explicit
            # classification keyword — these previously fell through to reporting.
            ("Which factors best separate high vs low value orders", TaskType.classification),
            ("Distinguish fraudulent orders from legitimate ones", TaskType.classification),
            ("Tell apart the customers who differentiate into two tiers", TaskType.classification),
            ("Forecast monthly revenue for the next year", TaskType.regression),
            ("Estimate how much each order will be worth", TaskType.regression),
            ("Segment our customers into distinct groups", TaskType.clustering),
            ("Find natural cohorts in the user base", TaskType.clustering),
            ("Detect anomalies in the sensor readings", TaskType.anomaly_detection),
            ("Flag unusual and suspicious transactions", TaskType.anomaly_detection),
            ("Give me a general summary and overview of the data", TaskType.reporting),
        ],
    )
    def test_classifies_representative_goals(
        self, classifier: GoalClassifier, goal: str, expected: TaskType
    ) -> None:
        result = classifier.classify(goal)
        assert isinstance(result, GoalClassification)
        assert result.task_type == expected
        assert 0.0 <= result.confidence <= 1.0

    def test_bare_separate_into_groups_is_not_classification(
        self, classifier: GoalClassifier
    ) -> None:
        # The discriminative keywords must not hijack a clustering-style goal:
        # "separate ... into groups" is clustering, so bare "separate" is
        # deliberately excluded from the classification lexicon.
        result = classifier.classify("Separate the customers into groups")
        assert result.task_type != TaskType.classification

    def test_empty_goal_falls_back_to_reporting(self, classifier: GoalClassifier) -> None:
        result = classifier.classify("")
        assert result.task_type == TaskType.reporting

    def test_unmatched_goal_falls_back_to_reporting(self, classifier: GoalClassifier) -> None:
        # No task-type keywords at all — rules inconclusive, chain falls back.
        result = classifier.classify("xyzzy foobar qux")
        assert result.task_type == TaskType.reporting

    def test_detects_target_column_from_schema(self, classifier: GoalClassifier) -> None:
        schema = IngestionResult(
            row_count=100,
            column_count=2,
            column_summary=[
                ColumnSummary(name="churn", dtype="int64", missing_count=0, stats=ColumnStats()),
                ColumnSummary(name="tenure", dtype="int64", missing_count=0, stats=ColumnStats()),
            ],
        )
        result = classifier.classify("Predict churn for each customer", dataset_schema=schema)
        assert result.task_type == TaskType.classification
        assert result.target_column == "churn"

    def test_always_returns_classification_object(self, classifier: GoalClassifier) -> None:
        for goal in ["", "cluster the data", "totally unrelated text"]:
            assert isinstance(classifier.classify(goal), GoalClassification)


class TestEmbeddingProvider:
    """
    Covers the local embedding zero-shot fallback. Calls the provider directly
    (rules are bypassed), so wording need not avoid keywords. Loads the
    sentence-transformers model — slower, but exercises the real path.
    """

    @pytest.fixture
    def provider(self) -> EmbeddingProvider:
        # The sentence-transformers model is cached at module scope in
        # rag.embeddings, so re-instantiating per test is cheap.
        return EmbeddingProvider()

    def test_returns_valid_classification(self, provider: EmbeddingProvider) -> None:
        result = provider.classify("understand what this dataset contains", [])
        assert isinstance(result, GoalClassification)
        assert result.task_type in set(TaskType)
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_goal_returns_none(self, provider: EmbeddingProvider) -> None:
        assert provider.classify("   ", []) is None

    @pytest.mark.parametrize(
        "goal, expected",
        [
            ("predict a continuous numeric amount like next month revenue", TaskType.regression),
            ("group similar customers into distinct segments", TaskType.clustering),
            ("find unusual outlier records that stand out", TaskType.anomaly_detection),
        ],
    )
    def test_semantic_routing(self, provider: EmbeddingProvider, goal, expected) -> None:
        assert provider.classify(goal, []).task_type == expected
