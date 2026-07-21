"""
tests/agents/test_orchestrator.py
──────────────────────────────────
Tests for the Module 2 goal-conditioned OrchestratorAgent.

These verify that the orchestrator classifies the goal, builds a *conditional*
pipeline (different goals → different pipelines), runs a ReAct loop, and returns
a well-formed step log. Business-logic assertions on real statistics arrive with
Module 3.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.goal_classifier import GoalClassifier, RuleBasedProvider
from agents.orchestrator import MAX_REACT_STEPS, OrchestratorAgent
from agents.planner import MINING, VISUALIZATION


@pytest.fixture
def orchestrator() -> OrchestratorAgent:
    # Rule-based classifier only — keeps tests fast and offline (no model load).
    return OrchestratorAgent(classifier=GoalClassifier(providers=[RuleBasedProvider()]))


class TestOrchestratorAgent:
    def test_instantiation(self, orchestrator: OrchestratorAgent) -> None:
        assert orchestrator is not None

    def test_run_returns_dict(self, orchestrator: OrchestratorAgent) -> None:
        result = orchestrator.run(goal="Find anomalies in sales data")
        assert isinstance(result, dict)

    def test_run_contains_required_keys(self, orchestrator: OrchestratorAgent) -> None:
        result = orchestrator.run(goal="Identify churn drivers")
        for key in ("goal", "task_type", "classification", "steps",
                    "recommendations", "rag_sources", "summary"):
            assert key in result

    def test_run_echoes_goal(self, orchestrator: OrchestratorAgent) -> None:
        goal = "Find seasonal patterns in revenue"
        result = orchestrator.run(goal=goal)
        assert result["goal"] == goal

    def test_sets_task_type(self, orchestrator: OrchestratorAgent) -> None:
        result = orchestrator.run(goal="Detect anomalies in transactions")
        assert result["task_type"] == "anomaly_detection"

    def test_steps_are_well_formed_react_steps(self, orchestrator: OrchestratorAgent) -> None:
        result = orchestrator.run(goal="Segment customers into groups")
        assert result["steps"]
        for step in result["steps"]:
            for field in ("agent_name", "action", "reasoning", "observation",
                          "status", "latency_ms"):
                assert field in step

    def test_reporting_goal_skips_visualization(self, orchestrator: OrchestratorAgent) -> None:
        result = orchestrator.run(goal="Give me a general summary of the data")
        agents = [s["agent_name"] for s in result["steps"]]
        assert VISUALIZATION not in agents

    def test_clustering_goal_includes_visualization(self, orchestrator: OrchestratorAgent) -> None:
        result = orchestrator.run(goal="Cluster the customers into segments")
        agents = [s["agent_name"] for s in result["steps"]]
        assert VISUALIZATION in agents

    def test_different_goals_produce_different_mining_directives(
        self, orchestrator: OrchestratorAgent
    ) -> None:
        """The FR-02 core claim, at the orchestrator level."""
        anomaly = orchestrator.run(goal="Detect outliers in the readings")
        cluster = orchestrator.run(goal="Segment customers into cohorts")

        def mining_comps(result):
            step = next(s for s in result["steps"] if s["agent_name"] == MINING)
            return step["output"].get("planned_computations")

        assert mining_comps(anomaly) != mining_comps(cluster)

    def test_never_exceeds_step_cap(self, orchestrator: OrchestratorAgent) -> None:
        result = orchestrator.run(goal="Profile customer data")
        assert len(result["steps"]) <= MAX_REACT_STEPS

    @pytest.mark.parametrize("expertise_level", ["beginner", "intermediate", "expert"])
    def test_run_accepts_all_expertise_levels(
        self, orchestrator: OrchestratorAgent, expertise_level: str
    ) -> None:
        result = orchestrator.run(goal="Analyse product metrics", expertise_level=expertise_level)
        assert result["expertise_level"] == expertise_level
