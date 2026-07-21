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

    def test_reporting_skips_visualization(self, planner) -> None:
        plan = planner.build_plan(TaskType.reporting)
        assert VISUALIZATION not in [s.agent_name for s in plan]

    @pytest.mark.parametrize(
        "task_type",
        [t for t in TaskType if t != TaskType.reporting],
    )
    def test_non_reporting_includes_visualization(self, planner, task_type) -> None:
        plan = planner.build_plan(task_type)
        assert VISUALIZATION in [s.agent_name for s in plan]

    def test_target_column_threaded_into_mining(self, planner) -> None:
        plan = planner.build_plan(TaskType.classification, target_column="churn")
        mining = next(s for s in plan if s.agent_name == MINING)
        assert mining.directives.get("target_column") == "churn"

    def test_every_step_carries_task_type_for_specialists(self, planner) -> None:
        plan = planner.build_plan(TaskType.clustering)
        for step in plan:
            if step.agent_name in (MINING, VISUALIZATION, RECOMMENDATION):
                assert step.directives.get("task_type") == "clustering"
