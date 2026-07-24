"""
tests/agents/test_planner.py
─────────────────────────────
Tests for the Module 2 PipelinePlanner — verifies that each task type yields a
distinct, conditioned pipeline (the FR-02 goal-conditioning claim).
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.planner import (
    INGESTION,
    MINING,
    RECOMMENDATION,
    VISUALIZATION,
    PipelinePlanner,
)
from backend.schemas.analysis import TaskType


@pytest.fixture
def planner() -> PipelinePlanner:
    return PipelinePlanner()


class TestPipelinePlanner:
    def test_pipeline_is_bracketed_by_ingestion_and_recommendation(self, planner) -> None:
        for task_type in TaskType:
            plan = planner.build_plan(task_type)
            assert plan[0].agent_name == INGESTION
            assert plan[-1].agent_name == RECOMMENDATION

    def test_mining_directives_differ_by_task_type(self, planner) -> None:
        """The core claim: different goals → different computations."""
        comps = {}
        for task_type in TaskType:
            plan = planner.build_plan(task_type)
            mining = next(s for s in plan if s.agent_name == MINING)
            comps[task_type] = tuple(mining.directives.get("computations", []))
        # Every task type must ask mining to run a distinct computation set.
        assert len(set(comps.values())) == len(TaskType)

    @pytest.mark.parametrize("task_type", list(TaskType))
    def test_every_task_type_includes_visualization(self, planner, task_type) -> None:
        """Every task type — including reporting — gets a conditioned chart
        step; only the chart set itself varies by goal (see _VIZ_DIRECTIVES)."""
        plan = planner.build_plan(task_type)
        assert VISUALIZATION in [s.agent_name for s in plan]

    def test_visualization_directives_differ_by_task_type(self, planner) -> None:
        charts = {}
        for task_type in TaskType:
            plan = planner.build_plan(task_type)
            viz = next(s for s in plan if s.agent_name == VISUALIZATION)
            charts[task_type] = tuple(viz.directives.get("charts", []))
        assert len(set(charts.values())) == len(TaskType)

    def test_target_column_threaded_into_mining(self, planner) -> None:
        plan = planner.build_plan(TaskType.classification, target_column="churn")
        mining = next(s for s in plan if s.agent_name == MINING)
        assert mining.directives.get("target_column") == "churn"

    def test_every_step_carries_task_type_for_specialists(self, planner) -> None:
        plan = planner.build_plan(TaskType.clustering)
        for step in plan:
            if step.agent_name in (MINING, VISUALIZATION, RECOMMENDATION):
                assert step.directives.get("task_type") == "clustering"
