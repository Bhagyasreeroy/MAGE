"""
tests/evaluation/test_metrics.py
─────────────────────────────────
Unit tests for the evaluation metrics (Objective 7).

These matter more than typical unit tests: the numbers these functions produce
are the project's validation claim and go into the report. A silently wrong
metric would not fail loudly anywhere else — it would just produce a plausible
result that happens to be false. The edge cases below (empty sets, no
recommendations, single-element input) are the ones where a naive
implementation manufactures a flattering answer.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from evaluation.metrics import (
    TASK_RELEVANT_COMPUTATIONS,
    citation_coverage,
    f1,
    jaccard_distance,
    mean_pairwise_divergence,
    pairwise_divergence_matrix,
    task_relevant_precision,
    task_relevant_recall,
)


class TestJaccardDistance:
    def test_identical_sets_are_zero_distance(self) -> None:
        assert jaccard_distance({"a", "b"}, {"a", "b"}) == 0.0

    def test_disjoint_sets_are_maximum_distance(self) -> None:
        assert jaccard_distance({"a"}, {"b"}) == 1.0

    def test_half_overlap(self) -> None:
        # |∩| = 1 ({b}), |∪| = 3 ({a,b,c}) → 1 - 1/3
        assert jaccard_distance({"a", "b"}, {"b", "c"}) == pytest.approx(2 / 3)

    def test_two_empty_sets_report_no_divergence(self) -> None:
        """Two runs that computed nothing have not been shown to differ."""
        assert jaccard_distance(set(), set()) == 0.0

    def test_empty_against_populated_is_total_divergence(self) -> None:
        assert jaccard_distance(set(), {"a"}) == 1.0

    def test_accepts_any_iterable_and_deduplicates(self) -> None:
        assert jaccard_distance(["a", "a", "b"], ("b", "a")) == 0.0

    def test_is_symmetric(self) -> None:
        a, b = {"a", "b", "c"}, {"c", "d"}
        assert jaccard_distance(a, b) == jaccard_distance(b, a)


class TestMeanPairwiseDivergence:
    def test_single_set_has_no_pair_to_compare(self) -> None:
        assert mean_pairwise_divergence([{"a", "b"}]) == 0.0

    def test_empty_input(self) -> None:
        assert mean_pairwise_divergence([]) == 0.0

    def test_identical_sets_diverge_by_zero(self) -> None:
        """This is the baseline's expected score — it must come out as 0."""
        assert mean_pairwise_divergence([{"a", "b"}] * 4) == 0.0

    def test_fully_disjoint_sets_diverge_by_one(self) -> None:
        assert mean_pairwise_divergence([{"a"}, {"b"}, {"c"}]) == 1.0

    def test_averages_over_all_pairs(self) -> None:
        # Pairs: (ab,ab)=0, (ab,cd)=1, (ab,cd)=1 → mean 2/3
        assert mean_pairwise_divergence(
            [{"a", "b"}, {"a", "b"}, {"c", "d"}]
        ) == pytest.approx(2 / 3)


class TestPairwiseDivergenceMatrix:
    def test_emits_one_record_per_unordered_pair(self) -> None:
        matrix = pairwise_divergence_matrix(
            {"x": {"a"}, "y": {"a"}, "z": {"b"}}
        )
        assert len(matrix) == 3  # 3 choose 2
        assert {(r["a"], r["b"]) for r in matrix} == {("x", "y"), ("x", "z"), ("y", "z")}

    def test_records_the_distance(self) -> None:
        matrix = pairwise_divergence_matrix({"x": {"a"}, "y": {"b"}})
        assert matrix[0]["distance"] == 1.0


class TestTaskRelevantPrecision:
    def test_all_relevant_computations_score_one(self) -> None:
        assert task_relevant_precision(
            {"class_balance", "feature_importance"}, "classification"
        ) == 1.0

    def test_irrelevant_computations_lower_the_score(self) -> None:
        # 1 of 2 relevant to classification ("kmeans" is not).
        assert task_relevant_precision({"class_balance", "kmeans"}, "classification") == 0.5

    def test_empty_computation_set_scores_zero_not_one(self) -> None:
        """A run that computed nothing must not be treated as vacuously perfect."""
        assert task_relevant_precision(set(), "classification") == 0.0

    def test_unknown_task_type_scores_zero(self) -> None:
        assert task_relevant_precision({"correlation"}, "not_a_task_type") == 0.0

    def test_baseline_fixed_set_is_penalised_on_a_specialised_task(self) -> None:
        """The baseline runs generic work on a clustering goal → low precision."""
        from evaluation.baseline import BASELINE_COMPUTATIONS

        assert task_relevant_precision(BASELINE_COMPUTATIONS, "clustering") < 0.5


class TestTaskRelevantRecall:
    def test_full_coverage_scores_one(self) -> None:
        assert task_relevant_recall(
            TASK_RELEVANT_COMPUTATIONS["clustering"], "clustering"
        ) == 1.0

    def test_partial_coverage(self) -> None:
        relevant = TASK_RELEVANT_COMPUTATIONS["classification"]
        assert task_relevant_recall({next(iter(relevant))}, "classification") == pytest.approx(
            1 / len(relevant)
        )

    def test_irrelevant_extras_do_not_inflate_recall(self) -> None:
        relevant = TASK_RELEVANT_COMPUTATIONS["clustering"]
        with_noise = set(relevant) | {"totally_unrelated", "also_unrelated"}
        assert task_relevant_recall(with_noise, "clustering") == 1.0

    def test_unknown_task_type_scores_zero(self) -> None:
        assert task_relevant_recall({"correlation"}, "not_a_task_type") == 0.0


class TestF1:
    def test_both_zero(self) -> None:
        assert f1(0.0, 0.0) == 0.0

    def test_both_one(self) -> None:
        assert f1(1.0, 1.0) == 1.0

    def test_is_the_harmonic_mean(self) -> None:
        assert f1(1.0, 0.5) == pytest.approx(2 / 3)

    def test_penalises_imbalance_relative_to_arithmetic_mean(self) -> None:
        """Perfect precision with poor recall must not score near 1."""
        assert f1(1.0, 0.1) < 0.5


class TestCitationCoverage:
    def test_all_recommendations_cited(self) -> None:
        recs = [{"sources": ["a.md"]}, {"sources": ["b.md", "c.md"]}]
        assert citation_coverage(recs) == 1.0

    def test_partial_coverage(self) -> None:
        assert citation_coverage([{"sources": ["a.md"]}, {"sources": []}]) == 0.5

    def test_missing_sources_key_counts_as_uncited(self) -> None:
        assert citation_coverage([{"insight": "x"}]) == 0.0

    def test_no_recommendations_scores_zero_not_one(self) -> None:
        """Nothing was grounded, so full coverage would be a false claim."""
        assert citation_coverage([]) == 0.0


class TestRelevanceSetsAreWellFormed:
    def test_covers_every_task_type(self) -> None:
        from backend.schemas.analysis import TaskType

        assert set(TASK_RELEVANT_COMPUTATIONS) == {t.value for t in TaskType}

    def test_no_task_type_has_an_empty_relevance_set(self) -> None:
        assert all(len(v) > 0 for v in TASK_RELEVANT_COMPUTATIONS.values())

    def test_task_types_are_not_all_identical(self) -> None:
        """If every task shared one relevance set, precision could not discriminate."""
        distinct = {frozenset(v) for v in TASK_RELEVANT_COMPUTATIONS.values()}
        assert len(distinct) == len(TASK_RELEVANT_COMPUTATIONS)
