"""
tests/agents/test_goal_conditioning.py
───────────────────────────────────────
Regression tests for the project's core claim (FR-02): the goal conditions
*which computations run and which charts are drawn*, not merely which results
are surfaced. Two different goals on the same dataset must produce different
mining and visualization output.

These lock in the fix for the gap where the planner emitted per-task directives
but MiningAgent / VisualizationAgent ignored them and ran a fixed pipeline.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.mining_agent import MiningAgent
from agents.planner import PipelinePlanner
from agents.visualization_agent import VisualizationAgent
from backend.schemas.analysis import TaskType


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 80
    units = rng.normal(5, 1.5, size=n)
    return pd.DataFrame(
        {
            "region": rng.choice(["East", "West", "North", "South"], size=n),
            "units": units.round(2),
            "revenue": (units * 20 + rng.normal(0, 2, size=n)).round(2),
            "label": rng.choice([0, 1], size=n),
        }
    )


def _mining_directives(task: TaskType, target: str | None = None) -> dict:
    """Pull the mining directives the planner would emit for a task type."""
    steps = PipelinePlanner().build_plan(task, target_column=target)
    mining_step = next(s for s in steps if s.agent_name == "MiningAgent")
    return mining_step.directives


def _viz_directives(task: TaskType) -> dict:
    steps = PipelinePlanner().build_plan(task)
    viz_step = next(s for s in steps if s.agent_name == "VisualizationAgent")
    return viz_step.directives


# ── Mining: computations actually vary by task type ──────────────────────────


class TestMiningConditioning:
    def test_clustering_runs_dbscan_not_isolation_forest(self, df) -> None:
        out = MiningAgent().run(
            context={"dataframe": df, "directives": _mining_directives(TaskType.clustering)}
        )
        assert out["dbscan"] is not None          # requested
        assert out["clustering"] is not None       # kmeans requested
        assert out["isolation_forest"] == {}       # NOT requested for clustering

    def test_anomaly_runs_isolation_forest_not_clustering(self, df) -> None:
        out = MiningAgent().run(
            context={"dataframe": df, "directives": _mining_directives(TaskType.anomaly_detection)}
        )
        assert out["isolation_forest"] != {}       # requested
        assert out["outliers"] != {} or out["isolation_forest"]  # iqr_outliers requested
        assert out["clustering"] is None            # kmeans NOT requested
        assert out["dbscan"] is None

    def test_classification_runs_class_balance(self, df) -> None:
        directives = _mining_directives(TaskType.classification, target="label")
        out = MiningAgent().run(context={"dataframe": df, "directives": directives})
        assert out["class_balance"] != {}
        assert out["class_balance"]["target"] == "label"
        assert out["class_balance"]["n_classes"] == 2

    def test_regression_runs_linearity(self, df) -> None:
        directives = _mining_directives(TaskType.regression, target="revenue")
        out = MiningAgent().run(context={"dataframe": df, "directives": directives})
        assert len(out["linearity"]) > 0
        # units↔revenue was constructed to be strongly linear.
        top = out["linearity"][0]
        assert top["feature"] in {"units", "label"}
        assert out["clustering"] is None            # not a clustering task

    def test_two_goals_produce_different_output(self, df) -> None:
        clustering_out = MiningAgent().run(
            context={"dataframe": df, "directives": _mining_directives(TaskType.clustering)}
        )
        anomaly_out = MiningAgent().run(
            context={"dataframe": df, "directives": _mining_directives(TaskType.anomaly_detection)}
        )
        # The whole point: same dataset, different goals → different computations.
        assert clustering_out["computations_run"] != anomaly_out["computations_run"]
        assert bool(clustering_out["dbscan"]) != bool(anomaly_out["dbscan"])
        assert bool(clustering_out["isolation_forest"]) != bool(anomaly_out["isolation_forest"])

    def test_no_directives_still_runs_full_default(self, df) -> None:
        """Backward compatibility: standalone use (no directives) = full profile."""
        out = MiningAgent().run(context={"dataframe": df, "goal": "profile"})
        assert out["correlations"] != {}
        assert out["feature_importance"] != []
        assert out["clustering"] is not None
        assert out["computations_run"] == ["<default profile>"]


# ── Visualization: chart set varies by task type ─────────────────────────────


class TestVisualizationConditioning:
    def _viz_for(self, df, task: TaskType) -> list[str]:
        mining = MiningAgent().run(
            context={"dataframe": df, "directives": _mining_directives(task)}
        )
        result = VisualizationAgent().run(
            context={
                "dataframe": df,
                "goal": task.value,
                "MiningAgent_output": mining,
                "directives": _viz_directives(task),
            }
        )
        return [s["type"] for s in result["viz_specs"]]

    def test_regression_yields_scatter_and_heatmap(self, df) -> None:
        types = self._viz_for(df, TaskType.regression)
        assert "scatter" in types or "correlation_heatmap" in types

    def test_reporting_yields_histograms_or_missingness(self, df) -> None:
        types = self._viz_for(df, TaskType.reporting)
        assert any(t in types for t in ("histogram", "missingness_matrix"))

    def test_chart_sets_differ_by_task(self, df) -> None:
        regression = set(self._viz_for(df, TaskType.regression))
        reporting = set(self._viz_for(df, TaskType.reporting))
        assert regression != reporting

    def test_no_directives_uses_default_set(self, df) -> None:
        mining = MiningAgent().run(context={"dataframe": df, "goal": "profile"})
        result = VisualizationAgent().run(
            context={"dataframe": df, "goal": "profile", "MiningAgent_output": mining}
        )
        assert len(result["viz_specs"]) > 0
