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

    def test_reporting_goal_still_includes_visualization(self, orchestrator: OrchestratorAgent) -> None:
        result = orchestrator.run(goal="Give me a general summary of the data")
        agents = [s["agent_name"] for s in result["steps"]]
        assert VISUALIZATION in agents

    def test_clustering_goal_includes_visualization(self, orchestrator: OrchestratorAgent) -> None:
        result = orchestrator.run(goal="Cluster the customers into segments")
        agents = [s["agent_name"] for s in result["steps"]]
        assert VISUALIZATION in agents

    def test_different_goals_produce_different_mining_directives(
        self, orchestrator: OrchestratorAgent
    ) -> None:
        """The FR-02 core claim, at the orchestrator level.

        The mining agent now runs real computations, so the goal-conditioning
        is observable in the planner's per-task rationale carried on the mining
        step (which names the task-specific computations) and in the inferred
        task_type — not in a stub echo of the directives.
        """
        anomaly = orchestrator.run(goal="Detect outliers in the readings")
        cluster = orchestrator.run(goal="Segment customers into cohorts")

        def mining_reasoning(result):
            step = next(s for s in result["steps"] if s["agent_name"] == MINING)
            return step["reasoning"]

        assert anomaly["task_type"] != cluster["task_type"]
        assert mining_reasoning(anomaly) != mining_reasoning(cluster)

    def test_never_exceeds_step_cap(self, orchestrator: OrchestratorAgent) -> None:
        result = orchestrator.run(goal="Profile customer data")
        assert len(result["steps"]) <= MAX_REACT_STEPS

    @pytest.mark.parametrize("expertise_level", ["beginner", "intermediate", "expert"])
    def test_run_accepts_all_expertise_levels(
        self, orchestrator: OrchestratorAgent, expertise_level: str
    ) -> None:
        result = orchestrator.run(goal="Analyse product metrics", expertise_level=expertise_level)
        assert result["expertise_level"] == expertise_level

    def test_failing_agent_is_recorded_and_pipeline_continues(
        self, orchestrator: OrchestratorAgent
    ) -> None:
        """A specialist raising must be caught, logged as 'error', not crash the run."""

        class FailingAgent:
            def run(self, context=None):
                raise RuntimeError("boom")

        orchestrator._agents[MINING] = FailingAgent()
        result = orchestrator.run(goal="Cluster the customers into segments")

        agents = [s["agent_name"] for s in result["steps"]]
        mining_step = next(s for s in result["steps"] if s["agent_name"] == MINING)
        assert mining_step["status"] == "error"
        assert "boom" in mining_step["observation"]
        # Pipeline kept going: a step after mining still ran.
        assert "RecommendationAgent" in agents

    def test_target_column_refined_from_ingested_schema(
        self, orchestrator: OrchestratorAgent, tmp_path
    ) -> None:
        """After ingestion reveals the schema, a goal-named column becomes the target."""
        csv = tmp_path / "customers.csv"
        csv.write_text("churn,tenure\n1,12\n0,4\n1,7\n", encoding="utf-8")

        result = orchestrator.run(
            goal="Predict churn for each customer",
            data={"source": str(csv)},
        )
        assert result["task_type"] == "classification"
        assert result["classification"]["target_column"] == "churn"
