"""
tests/agents/test_prior_run_grounding.py
─────────────────────────────────────────
Tests for prior-run memory reaching the RecommendationAgent.

Recording memory is only half of Objective 4 — a table nobody reads changes
nothing. This covers the other half: prior runs are supplied to the
RecommendationAgent as additional context, which biases retrieval toward the
methodology relevant to what this user has already investigated, and surfaces
what they previously found.

The integrity constraint is the important one. Prior runs are *context*, not
citations: a recommendation's `sources` must still name knowledge-base
documents, because FR-03 promises retrievable methodology and "you found this
last week" is not that. Memory changes what gets retrieved; it never becomes
the citation itself.

Written test-first.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.recommendation_agent import RecommendationAgent

PRIOR_RUNS = [
    {
        "goal": "Segment customers into behavioural groups",
        "task_type": "clustering",
        "findings": ["Data separates into 3 clusters (silhouette=0.52)."],
        "similarity": 0.81,
    },
    {
        "goal": "Check which customers churn",
        "task_type": "classification",
        "findings": ["Target 'churned' has 2 classes (imbalance ratio 3.1:1)."],
        "similarity": 0.55,
    },
]

MINING_OUTPUT = {
    "patterns": ["'units' and 'revenue' are strongly positively correlated (r=0.91)."],
    "statistics": {},
    "data_quality": {},
}


@pytest.fixture(scope="module")
def agent() -> RecommendationAgent:
    return RecommendationAgent()


def _run(agent: RecommendationAgent, prior_runs=None) -> dict:
    context = {
        "goal": "Find natural clusters and segments in this data",
        "MiningAgent_output": dict(MINING_OUTPUT),
    }
    if prior_runs is not None:
        context["prior_runs"] = prior_runs
    return agent.run(context=context)


class TestPriorRunsReachTheQuery:
    def test_prior_findings_are_included_in_the_retrieval_query(
        self, agent: RecommendationAgent
    ) -> None:
        query = agent._build_query(  # noqa: SLF001 - the query is the integration point
            {
                "goal": "Find clusters",
                "MiningAgent_output": dict(MINING_OUTPUT),
                "prior_runs": PRIOR_RUNS,
            }
        )
        assert "silhouette" in query

    def test_the_query_still_leads_with_the_current_goal(
        self, agent: RecommendationAgent
    ) -> None:
        """History informs retrieval; it must not displace the present question."""
        query = agent._build_query(  # noqa: SLF001
            {"goal": "Find clusters", "prior_runs": PRIOR_RUNS}
        )
        assert query.startswith("Find clusters")

    def test_absent_prior_runs_leave_the_query_unchanged(
        self, agent: RecommendationAgent
    ) -> None:
        base = agent._build_query({"goal": "Find clusters"})  # noqa: SLF001
        with_empty = agent._build_query(  # noqa: SLF001
            {"goal": "Find clusters", "prior_runs": []}
        )
        assert base == with_empty


class TestPriorRunsAreSurfaced:
    def test_prior_runs_are_returned_for_display(self, agent: RecommendationAgent) -> None:
        assert _run(agent, PRIOR_RUNS)["prior_runs"]

    def test_the_key_is_present_even_with_no_history(
        self, agent: RecommendationAgent
    ) -> None:
        """A stable output shape — the frontend should not have to guess."""
        assert _run(agent)["prior_runs"] == []

    def test_prior_runs_carry_their_findings(self, agent: RecommendationAgent) -> None:
        surfaced = _run(agent, PRIOR_RUNS)["prior_runs"]
        assert any("silhouette" in f for run in surfaced for f in run["findings"])


class TestCitationIntegrityIsPreserved:
    """FR-03 must survive the addition. Memory is context, never a citation."""

    def test_recommendations_still_cite_knowledge_base_documents(
        self, agent: RecommendationAgent
    ) -> None:
        result = _run(agent, PRIOR_RUNS)
        assert result["recommendations"]
        for rec in result["recommendations"]:
            assert rec["sources"]
            assert all(src.endswith(".md") for src in rec["sources"])

    def test_no_prior_run_is_presented_as_a_source(
        self, agent: RecommendationAgent
    ) -> None:
        result = _run(agent, PRIOR_RUNS)
        for rec in result["recommendations"]:
            for source in rec["sources"]:
                assert "Segment customers" not in source

    def test_rag_sources_remain_knowledge_base_only(
        self, agent: RecommendationAgent
    ) -> None:
        assert all(s.endswith(".md") for s in _run(agent, PRIOR_RUNS)["rag_sources"])


class TestMalformedMemoryIsHarmless:
    def test_prior_runs_without_findings_do_not_break_the_run(
        self, agent: RecommendationAgent
    ) -> None:
        result = _run(agent, [{"goal": "Something", "task_type": "reporting"}])
        assert result["recommendations"]

    def test_non_dict_entries_are_ignored(self, agent: RecommendationAgent) -> None:
        result = _run(agent, ["not a dict", None])
        assert result["recommendations"]

    def test_a_non_list_value_is_ignored(self, agent: RecommendationAgent) -> None:
        result = _run(agent, "not a list")
        assert result["recommendations"]
