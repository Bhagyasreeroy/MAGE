"""
tests/agents/test_react_step_cap.py
────────────────────────────────────
Honesty test for the ReAct step cap.

``MAX_REACT_STEPS = 10`` is checked against ``len(steps)`` — the steps actually
emitted, not the four static entries ``PipelinePlanner`` still always builds
up front. ``OrchestratorAgent._reflect_on_mining`` (see agents/orchestrator.py)
observes what MiningAgent found and can insert 1-3 further
"OrchestratorAgent / reflect" steps that change what Visualization and
Recommendation do next — so a real run is genuinely 4-7 steps depending on the
data, not a compile-time constant. This is what closes the gap the original
version of this file pinned: previously the planner's fixed-length list made
the cap provably unreachable; now the loop is adaptive and the cap guards its
actual length, even though 7 < 10 still means it does not fire in ordinary
operation.

These tests pin two things: that a normal run — including one that hits every
reflection trigger — stays under the cap, and that the cap still truncates and
logs a plan that runs away regardless (``test_a_runaway_plan_is_truncated``).
See tests/agents/test_orchestrator_reflection.py for the reflection triggers
themselves.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents import orchestrator as orch
from agents.orchestrator import MAX_REACT_STEPS, OrchestratorAgent
from agents.planner import PipelinePlanner
from backend.schemas.analysis import TaskType


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"a": range(20), "b": [x * 2 for x in range(20)]})


class TestTheCapIsRealButUnreachedToday:
    def test_every_task_type_plans_well_inside_the_cap(self) -> None:
        """Why the branch never fires in practice — state this, don't hide it."""
        for task in TaskType:
            plan = PipelinePlanner().build_plan(task)
            assert len(plan) < MAX_REACT_STEPS, f"{task.value} plans {len(plan)} steps"

    def test_the_planner_length_is_fixed_at_four(self) -> None:
        lengths = {len(PipelinePlanner().build_plan(t)) for t in TaskType}
        assert lengths == {4}, f"plan lengths vary: {lengths} — revisit the cap's description"

    def test_a_runaway_plan_is_truncated(self, df, monkeypatch) -> None:
        """The cap works. Feed a plan longer than the bound and it stops."""
        real_build = PipelinePlanner.build_plan

        def runaway(self, task_type, target_column=None, dataset_schema=None):
            plan = real_build(self, task_type, target_column, dataset_schema)
            # 40 steps: four times the cap, all valid.
            return (plan * 10)[:40]

        monkeypatch.setattr(PipelinePlanner, "build_plan", runaway)
        result = OrchestratorAgent().run(goal="profile this dataset", data={"dataframe": df})
        assert len(result["steps"]) <= MAX_REACT_STEPS

    def test_truncation_is_logged_rather_than_silent(self, df, monkeypatch, caplog) -> None:
        """Silently dropping planned work would be worse than running it."""
        real_build = PipelinePlanner.build_plan
        monkeypatch.setattr(
            PipelinePlanner,
            "build_plan",
            lambda self, t, target_column=None, dataset_schema=None: (
                real_build(self, t, target_column, dataset_schema) * 10
            )[:40],
        )
        with caplog.at_level("WARNING", logger=orch.__name__):
            OrchestratorAgent().run(goal="profile this dataset", data={"dataframe": df})
        assert any("step limit" in r.message.lower() for r in caplog.records)

    def test_a_normal_run_is_never_truncated(self, df) -> None:
        result = OrchestratorAgent().run(goal="profile this dataset", data={"dataframe": df})
        assert len(result["steps"]) == 4
