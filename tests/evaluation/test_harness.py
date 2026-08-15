"""
tests/evaluation/test_harness.py
─────────────────────────────────
Tests for the evaluation harness (Objective 7).

Two layers:

  • **Extraction** — the harness reads its evidence out of the orchestrator's
    Reason/Act/Observe step log, because the aggregate result flattens
    recommendations to display strings and drops mining internals. If that
    extraction silently returned empty sets, every metric would still compute
    and would report a confident, wrong answer. These tests pin the shapes.
  • **End-to-end** — one real sweep over a single dataset, asserting the
    project's central claim actually holds when measured: different goals on
    the same data produce different computations, while the baseline does not.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from evaluation.datasets import load_eval_datasets
from evaluation.harness import (
    GOAL_TEMPLATES,
    _build_goal,
    _extract_chart_types,
    _extract_computations,
    _extract_recommendations,
    _step_output,
    run_harness,
)


class TestStepLogExtraction:
    @pytest.fixture
    def result(self) -> dict:
        return {
            "task_type": "clustering",
            "rag_sources": ["clustering.md"],
            "steps": [
                {"agent_name": "IngestionAgent", "output": {"row_count": 10, "column_count": 3}},
                {
                    "agent_name": "MiningAgent",
                    "output": {"computations_run": ["kmeans", "dbscan", "standardize"]},
                },
                {
                    "agent_name": "VisualizationAgent",
                    "output": {
                        "viz_specs": [
                            {"type": "cluster_scatter", "title": "A"},
                            {"type": "cluster_scatter", "title": "B"},
                            {"type": "histogram", "title": "C"},
                        ]
                    },
                },
                {
                    "agent_name": "RecommendationAgent",
                    "output": {"recommendations": [{"insight": "x", "sources": ["clustering.md"]}]},
                },
            ],
        }

    def test_finds_a_named_agents_output(self, result: dict) -> None:
        assert _step_output(result, "MiningAgent")["computations_run"] == [
            "kmeans", "dbscan", "standardize"
        ]

    def test_missing_agent_returns_empty_dict(self, result: dict) -> None:
        assert _step_output(result, "NoSuchAgent") == {}

    def test_handles_a_result_with_no_steps(self) -> None:
        assert _step_output({}, "MiningAgent") == {}

    def test_extracts_computations_sorted(self, result: dict) -> None:
        assert _extract_computations(result) == ["dbscan", "kmeans", "standardize"]

    def test_extracts_distinct_chart_types(self, result: dict) -> None:
        """Two cluster_scatter specs are one chart *type*."""
        assert _extract_chart_types(result) == ["cluster_scatter", "histogram"]

    def test_extracts_structured_recommendations_with_sources(self, result: dict) -> None:
        recs = _extract_recommendations(result)
        assert len(recs) == 1
        assert recs[0]["sources"] == ["clustering.md"]

    def test_failed_run_yields_empty_evidence_rather_than_raising(self) -> None:
        assert _extract_computations({}) == []
        assert _extract_chart_types({}) == []
        assert _extract_recommendations({}) == []

    def test_default_profile_placeholder_is_expanded_to_real_tokens(self) -> None:
        """
        MiningAgent reports the literal "<default profile>" when unconditioned.
        Left as-is, Jaccard would read that placeholder as one exotic
        computation and manufacture divergence that never happened.
        """
        expanded = _extract_computations(
            {"steps": [{"agent_name": "MiningAgent",
                        "output": {"computations_run": ["<default profile>"]}}]}
        )
        assert "<default profile>" not in expanded
        assert "descriptive_profile" in expanded
        assert len(expanded) > 1


class TestGoalConstruction:
    def test_one_goal_per_task_type(self) -> None:
        from backend.schemas.analysis import TaskType

        assert {t for t, _ in GOAL_TEMPLATES} == {t.value for t in TaskType}

    def test_target_column_is_interpolated(self) -> None:
        (dataset,) = [d for d in load_eval_datasets() if d.name == "customer_orders"]
        assert "churned" in _build_goal("classification", dataset)

    def test_no_placeholder_survives_into_any_goal(self) -> None:
        for dataset in load_eval_datasets():
            for task_type, _ in GOAL_TEMPLATES:
                assert "{target}" not in _build_goal(task_type, dataset)

    def test_goals_are_distinct_per_task_type(self) -> None:
        (dataset,) = [d for d in load_eval_datasets() if d.name == "iris"]
        goals = [_build_goal(t, dataset) for t, _ in GOAL_TEMPLATES]
        assert len(set(goals)) == len(goals)


@pytest.fixture(scope="module")
def results() -> dict:
    """One real sweep, shared by the end-to-end tests — the pipeline is slow to re-run."""
    return run_harness(dataset_filter=["iris"], live_baseline=False)


class TestHarnessEndToEnd:
    """One real sweep. Slow-ish, but this is the test that proves the claim."""

    def test_every_run_succeeded(self, results: dict) -> None:
        assert results["summary"]["successful_runs"] == results["summary"]["total_runs"]
        assert results["summary"]["total_runs"] == len(GOAL_TEMPLATES)

    def test_mage_computations_diverge_across_goals(self, results: dict) -> None:
        """FR-02 / Objective 7 — the headline claim, measured."""
        assert results["summary"]["mage_computation_divergence"] > 0.5

    def test_baseline_does_not_diverge_across_goals(self, results: dict) -> None:
        assert results["summary"]["baseline_computation_divergence"] == 0.0

    def test_mage_diverges_strictly_more_than_the_baseline(self, results: dict) -> None:
        s = results["summary"]
        assert s["mage_computation_divergence"] > s["baseline_computation_divergence"]

    def test_charts_also_diverge_across_goals(self, results: dict) -> None:
        assert results["summary"]["mage_chart_divergence"] > 0.0

    def test_no_two_goals_ran_an_identical_computation_set(self, results: dict) -> None:
        sets = [frozenset(r["computations_run"]) for r in results["runs"]]
        assert len(set(sets)) == len(sets)

    def test_every_run_actually_computed_something(self, results: dict) -> None:
        assert all(r["computations_run"] for r in results["runs"])

    def test_mage_precision_beats_the_baseline(self, results: dict) -> None:
        s = results["summary"]
        assert s["mean_precision"] > s["mean_baseline_precision"]

    def test_citation_coverage_is_total(self, results: dict) -> None:
        """FR-03 — every recommendation carries a retrievable source."""
        assert results["summary"]["mean_citation_coverage"] == 1.0

    def test_fr05_runtime_under_60_seconds(self, results: dict) -> None:
        assert results["summary"]["max_runtime_seconds"] < 60.0
        assert results["summary"]["fr05_under_60s"] is True

    def test_payload_carries_provenance_for_the_report(self, results: dict) -> None:
        assert results["experiment"]["design"] == "same-dataset / different-goal"
        assert results["environment"]["baseline_mode"] == "declarative-spec"
        assert results["generated_at"]

    def test_unknown_dataset_filter_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            run_harness(dataset_filter=["no_such_dataset"])


class TestReportRendering:
    def test_markdown_contains_the_headline_numbers(self) -> None:
        from evaluation.report import build_markdown

        results = {
            "generated_at": "2026-08-11T00:00:00+00:00",
            "environment": {"python": "3.13.7", "platform": "test",
                            "baseline_mode": "declarative-spec"},
            "experiment": {"design": "same-dataset / different-goal", "datasets": ["iris"],
                           "goals_per_dataset": ["clustering"], "data_dir": "/tmp"},
            "summary": {
                "total_runs": 1, "successful_runs": 1,
                "mage_computation_divergence": 0.95, "baseline_computation_divergence": 0.0,
                "mage_chart_divergence": 0.94, "mean_precision": 1.0, "mean_recall": 0.57,
                "mean_baseline_precision": 0.4, "mean_baseline_recall": 0.57,
                "mean_f1": 0.72, "mean_baseline_f1": 0.47,
                "mean_citation_coverage": 1.0, "goal_classification_accuracy": 1.0,
                "max_runtime_seconds": 1.8, "mean_runtime_seconds": 0.9, "fr05_under_60s": True,
            },
            "per_dataset": [{
                "dataset": "iris", "description": "d", "notes": {},
                "row_count": 150, "column_count": 5,
                "mage_computation_divergence": 0.95, "baseline_computation_divergence": 0.0,
                "mage_chart_divergence": 0.94, "computation_pairs": [], "chart_pairs": [],
                "mean_precision": 1.0, "mean_recall": 0.57,
                "mean_baseline_precision": 0.4, "mean_baseline_recall": 0.57,
                "mean_citation_coverage": 1.0, "max_runtime_seconds": 1.8,
            }],
            "runs": [{
                "dataset": "iris", "intended_task_type": "clustering",
                "classified_task_type": "clustering",
                "computations_run": ["kmeans"], "chart_types": ["cluster_scatter"],
                "runtime_seconds": 1.8, "baseline_computations": ["correlation"],
            }],
        }
        md = build_markdown(results)
        assert "0.950" in md
        assert "PASS" in md
        assert "same-dataset / different-goal" in md
        assert "`kmeans`" in md
