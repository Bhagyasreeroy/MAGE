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

from agents.goal_classifier import GoalClassifier, RuleBasedProvider
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
