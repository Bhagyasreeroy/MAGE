"""
tests/agents/test_orchestrator.py
──────────────────────────────────
Smoke tests for OrchestratorAgent.

These tests verify the skeleton runs end-to-end without exceptions.
Business-logic assertions will be added once each agent is implemented.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.orchestrator import OrchestratorAgent


class TestOrchestratorAgent:
    """Smoke tests for the OrchestratorAgent skeleton."""

    @pytest.fixture
    def orchestrator(self) -> OrchestratorAgent:
        return OrchestratorAgent()

    def test_instantiation(self, orchestrator: OrchestratorAgent) -> None:
        """OrchestratorAgent can be instantiated without errors."""
        assert orchestrator is not None

    def test_run_returns_dict(self, orchestrator: OrchestratorAgent) -> None:
        """run() returns a dictionary result."""
        result = orchestrator.run(goal="Find anomalies in sales data")
        assert isinstance(result, dict)

    def test_run_contains_required_keys(self, orchestrator: OrchestratorAgent) -> None:
        """run() result contains all expected top-level keys."""
        result = orchestrator.run(goal="Identify churn drivers")
        assert "goal" in result
        assert "steps" in result
        assert "recommendations" in result
        assert "rag_sources" in result
        assert "summary" in result

    def test_run_echoes_goal(self, orchestrator: OrchestratorAgent) -> None:
        """run() echoes the input goal in the result."""
        goal = "Find seasonal patterns in revenue"
        result = orchestrator.run(goal=goal)
        assert result["goal"] == goal

    def test_run_executes_all_agents(self, orchestrator: OrchestratorAgent) -> None:
        """run() produces a step entry for each specialist agent."""
        result = orchestrator.run(goal="Profile customer data")
        agent_names = [s["agent"] for s in result["steps"]]
        assert "IngestionAgent" in agent_names
        assert "MiningAgent" in agent_names
        assert "VisualizationAgent" in agent_names
        assert "RecommendationAgent" in agent_names

    @pytest.mark.parametrize("expertise_level", ["beginner", "intermediate", "expert"])
    def test_run_accepts_all_expertise_levels(
        self, orchestrator: OrchestratorAgent, expertise_level: str
    ) -> None:
        """run() completes without error for all expertise levels."""
        result = orchestrator.run(
            goal="Analyse product metrics",
            expertise_level=expertise_level,
        )
        assert result["expertise_level"] == expertise_level
