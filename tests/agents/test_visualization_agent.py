"""Tests for agents/visualization_agent.py."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.mining_agent import MiningAgent
from agents.visualization_agent import VisualizationAgent


@pytest.fixture
def agent() -> VisualizationAgent:
    return VisualizationAgent()


def _context_for(df: pd.DataFrame, goal: str = "profile this dataset") -> dict:
    mining_output = MiningAgent().run(context={"dataframe": df, "goal": goal})
    return {"dataframe": df, "goal": goal, "MiningAgent_output": mining_output}


class TestVisualizationAgentEmptyInputs:
    def test_no_dataframe_returns_empty_specs(self, agent: VisualizationAgent) -> None:
        result = agent.run(context={"goal": "profile", "MiningAgent_output": {"statistics": {}}})
        assert result["viz_specs"] == []

    def test_no_mining_output_returns_empty_specs(self, agent: VisualizationAgent) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = agent.run(context={"dataframe": df, "goal": "profile"})
        assert result["viz_specs"] == []


class TestVisualizationAgentSpecs:
    def test_correlation_heatmap_included_for_multiple_numeric_columns(self, agent: VisualizationAgent) -> None:
        df = pd.DataFrame({"a": range(20), "b": [x * 2 for x in range(20)], "region": ["E", "W"] * 10})
        result = agent.run(context=_context_for(df))
        types = [s["type"] for s in result["viz_specs"]]
        assert "correlation_heatmap" in types

    def test_id_like_column_excluded_from_histograms(self, agent: VisualizationAgent) -> None:
        df = pd.DataFrame({"id": range(1, 21), "value": [1, 2, 3, 4, 5] * 4})
        result = agent.run(context=_context_for(df))
        histogram_cols = [s["column"] for s in result["viz_specs"] if s["type"] == "histogram"]
        assert "id" not in histogram_cols

    def test_categorical_bar_chart_included(self, agent: VisualizationAgent) -> None:
        df = pd.DataFrame({"value": range(20), "category": ["A", "B", "C"] * 6 + ["A", "B"]})
        result = agent.run(context=_context_for(df))
        bar_specs = [s for s in result["viz_specs"] if s["type"] == "bar" and s["title"].startswith("Top values")]
        assert len(bar_specs) == 1

    def test_outlier_column_gets_boxplot(self, agent: VisualizationAgent) -> None:
        values = [10, 11, 9, 10, 12, 11, 10, 9, 500]
        df = pd.DataFrame({"value": values})
        result = agent.run(context=_context_for(df))
        boxplot_specs = [s for s in result["viz_specs"] if s["type"] == "boxplot"]
        assert len(boxplot_specs) == 1
        assert boxplot_specs[0]["column"] == "value"

    def test_histogram_bin_labels_are_distinct(self, agent: VisualizationAgent) -> None:
        # Large-magnitude values with repeats (not an id-like column, so it
        # isn't excluded by the uniqueness filter) — regression check for
        # bin labels collapsing to e.g. "1e+03-1e+03".
        df = pd.DataFrame({"value": [1001, 1002, 1001, 1005, 1010, 1015, 1002, 1020, 1001, 1008] * 2})
        result = agent.run(context=_context_for(df))
        histograms = [s for s in result["viz_specs"] if s["type"] == "histogram"]
        assert len(histograms) == 1
        labels = [b["label"] for b in histograms[0]["bins"]]
        assert len(labels) == len(set(labels)), "bin labels must be distinct, not collapse to e.g. '1e+03-1e+03'"

    def test_cluster_scatter_included_when_clustering_present(self, agent: VisualizationAgent) -> None:
        import numpy as np

        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30)})
        result = agent.run(context=_context_for(df))
        types = [s["type"] for s in result["viz_specs"]]
        # Clustering may or may not find a good k depending on random data,
        # but if MiningAgent reports one, VisualizationAgent must include it.
        mining_output = MiningAgent().run(context={"dataframe": df, "goal": "profile"})
        if mining_output["clustering"] is not None:
            assert "cluster_scatter" in types
