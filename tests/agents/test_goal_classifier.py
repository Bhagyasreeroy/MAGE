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

from agents.goal_classifier import (
    DEFAULT_TASK_TYPE,
    INCONCLUSIVE_BELOW,
    EmbeddingProvider,
    GoalClassifier,
    RuleBasedProvider,
)
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


class TestInconclusiveFloorIsEnforced:
    """
    F1 — `INCONCLUSIVE_BELOW` was defined and then ignored.

    `classify()` returned the highest-scoring provider result whether or not it
    cleared the floor, so a constant named "inconclusive below" never caused
    anything to be treated as inconclusive. Measured before the fix: "how do
    passengers change month over month" came back as `clustering` at **0.19**,
    presented to the reader as a decision.

    Note where the floor actually bites. RuleBasedProvider's confidence is
    `0.55 + 0.4 * share`, so a keyword match can never score below 0.55 — above
    the floor by construction. The fallback therefore fires in exactly one
    situation: no keyword matched at all, and the embedding provider's guess is
    weak. That is the honest reading of "we do not know".
    """

    def test_a_sub_threshold_guess_falls_back_to_reporting(self) -> None:
        class WeakProvider:
            def classify(self, goal: str, columns: list[str]) -> GoalClassification:
                return GoalClassification(
                    task_type=TaskType.clustering,
                    target_column=None,
                    confidence=0.19,
                    rationale="a weak guess",
                )

        result = GoalClassifier(providers=[WeakProvider()]).classify("something ambiguous")

        assert result.task_type == DEFAULT_TASK_TYPE
        assert result.confidence < INCONCLUSIVE_BELOW

    def test_the_rationale_names_the_uncertainty_and_the_rejected_guess(self) -> None:
        """
        A silent downgrade is its own dishonesty: the reader should be able to
        see that a guess was considered and rejected, not just that they got a
        general report.
        """
        class WeakProvider:
            def classify(self, goal: str, columns: list[str]) -> GoalClassification:
                return GoalClassification(
                    task_type=TaskType.clustering,
                    target_column=None,
                    confidence=0.19,
                    rationale="a weak guess",
                )

        result = GoalClassifier(providers=[WeakProvider()]).classify("something ambiguous")

        assert "clustering" in result.rationale, "name the guess that was rejected"
        assert "0.19" in result.rationale, "name its confidence"

    def test_the_detected_target_column_survives_the_fallback(self) -> None:
        """The target is read off the goal text, not the task type — losing it
        would make the fallback cost information it never needed to cost."""
        class WeakProvider:
            def classify(self, goal: str, columns: list[str]) -> GoalClassification:
                return GoalClassification(
                    task_type=TaskType.clustering, target_column="churned",
                    confidence=0.2, rationale="weak",
                )

        result = GoalClassifier(providers=[WeakProvider()]).classify(
            "something about churned", dataset_schema={"column_summary": [{"name": "churned"}]},
        )

        assert result.target_column == "churned"

    def test_a_confident_classification_is_untouched(self) -> None:
        result = GoalClassifier().classify("find natural clusters and segments in this data")

        assert result.task_type == TaskType.clustering
        assert result.confidence >= INCONCLUSIVE_BELOW

    @pytest.mark.parametrize(
        ("goal", "expected"),
        [
            ("Predict which category each record falls into based on churned", TaskType.classification),
            ("Understand what drives churned and estimate its value from the other fields", TaskType.regression),
            ("Find natural groupings and segments hidden in this data", TaskType.clustering),
            ("Identify unusual or suspicious records that don't fit the pattern", TaskType.anomaly_detection),
            ("Give me a general overview of this dataset and its data quality", TaskType.reporting),
        ],
    )
    def test_the_five_harness_goals_are_unaffected(self, goal: str, expected: TaskType) -> None:
        """The evaluation harness must not move because of this change — all
        five score 0.78-0.95, comfortably clear of the floor."""
        result = GoalClassifier().classify(goal)

        assert result.task_type == expected
        assert result.confidence >= INCONCLUSIVE_BELOW


class TestTemporalVocabulary:
    """
    F2 — "trend", "over time", "seasonality", "month over month" and their
    relatives appeared nowhere in the lexicon, so temporal goals fell through
    the rules entirely and were classified by embedding similarity alone.

    They map to `reporting` because there is no time-series TaskType: reporting
    already emits the `line` chart, so a temporal goal produces the right
    artefact without any new computation. When B1 adds decomposition this is
    the vocabulary it will claim.
    """

    @pytest.mark.parametrize(
        "goal",
        [
            "show me the trend in passengers over time",
            "analyse the seasonality of passengers",
            "how do passengers change month over month",
            "plot passengers over time",
            "is there a seasonal pattern in this data",
            "show the year over year change",
        ],
    )
    def test_temporal_goals_classify_as_reporting_with_confidence(self, goal: str) -> None:
        result = GoalClassifier().classify(goal)

        assert result.task_type == TaskType.reporting, f"{goal!r} -> {result.task_type.value}"
        assert result.confidence >= INCONCLUSIVE_BELOW, f"{goal!r} scored {result.confidence}"

    def test_forecast_still_reads_as_regression(self) -> None:
        """
        'forecast' is temporal *and* already a 3.0-weight regression keyword.
        Adding temporal vocabulary must not drag it into reporting — predicting
        a future value is a regression goal, whatever its time axis.
        """
        result = GoalClassifier().classify("forecast next quarter revenue")

        assert result.task_type == TaskType.regression


class TestPluralKeywordGap:
    """
    Found while measuring F1: 'predict house prices from these features' matched
    *no* keyword and fell to a 0.41 embedding guess — because the lexicon lists
    "price" but not "prices", while it lists both "outlier"/"outliers" and
    "segment"/"segments" elsewhere. A plain omission rather than a design.

    It matters more after F1: before, the weak guess was returned anyway and
    happened to be right; now a sub-threshold score becomes `reporting`, so an
    unmatched plural turns a correct regression answer into a general report.
    """

    def test_a_plural_price_goal_reads_as_regression(self) -> None:
        result = GoalClassifier().classify("predict house prices from these features")

        assert result.task_type == TaskType.regression
        assert result.confidence >= INCONCLUSIVE_BELOW
