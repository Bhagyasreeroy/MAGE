"""Tests for agents/mining_agent.py."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.mining_agent import MiningAgent


@pytest.fixture
def agent() -> MiningAgent:
    return MiningAgent()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "region": rng.choice(["East", "West", "North", "South"], size=n),
            "units": rng.normal(5, 1.5, size=n).round(1),
            # revenue correlated with units by construction
            "revenue": None,
        }
    ).assign(revenue=lambda d: (d["units"] * 20 + rng.normal(0, 2, size=n)).round(2))


class TestMiningAgentEmptyInputs:
    def test_no_dataframe_returns_graceful_empty_result(self, agent: MiningAgent) -> None:
        result = agent.run(context={"goal": "profile this"})
        assert result["statistics"] == {}
        assert result["clustering"] is None
        assert result["patterns"] == []

    def test_run_with_no_context(self, agent: MiningAgent) -> None:
        result = agent.run(context=None)
        assert result["statistics"] == {}


class TestMiningAgentStatistics:
    def test_numeric_and_categorical_columns_typed_correctly(self, agent: MiningAgent, sample_df) -> None:
        result = agent.run(context={"dataframe": sample_df, "goal": "profile"})
        assert result["statistics"]["units"]["type"] == "numeric"
        assert result["statistics"]["region"]["type"] == "categorical"
        assert result["statistics"]["region"]["cardinality"] == 4

    def test_numeric_stats_are_sane(self, agent: MiningAgent, sample_df) -> None:
        result = agent.run(context={"dataframe": sample_df, "goal": "profile"})
        units_stats = result["statistics"]["units"]
        assert units_stats["min"] <= units_stats["median"] <= units_stats["max"]
        assert units_stats["q1"] <= units_stats["median"] <= units_stats["q3"]

    def test_data_quality_reports_completeness_and_uniqueness(self, agent: MiningAgent) -> None:
        df = pd.DataFrame({"a": [1, 2, None, 4], "b": [1, 1, 1, 1]})
        result = agent.run(context={"dataframe": df, "goal": "profile"})
        assert result["data_quality"]["a"]["completeness_pct"] == 75.0
        assert result["data_quality"]["a"]["missing_count"] == 1
        assert result["data_quality"]["b"]["uniqueness_pct"] == 25.0


class TestMiningAgentCorrelation:
    def test_correlated_columns_detected(self, agent: MiningAgent, sample_df) -> None:
        result = agent.run(context={"dataframe": sample_df, "goal": "profile"})
        r = result["correlations"]["units"]["revenue"]
        assert r > 0.9, "units and revenue were constructed to be strongly correlated"

    def test_single_numeric_column_produces_no_correlations(self, agent: MiningAgent) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"]})
        result = agent.run(context={"dataframe": df, "goal": "profile"})
        assert result["correlations"] == {}

    def test_strong_correlation_surfaces_as_pattern(self, agent: MiningAgent, sample_df) -> None:
        result = agent.run(context={"dataframe": sample_df, "goal": "profile"})
        assert any("correlated" in p for p in result["patterns"])


class TestMiningAgentOutliers:
    def test_injected_outlier_is_detected(self, agent: MiningAgent) -> None:
        values = [10, 11, 9, 10, 12, 11, 10, 9, 500]  # 500 is a clear IQR outlier
        df = pd.DataFrame({"value": values})
        result = agent.run(context={"dataframe": df, "goal": "profile"})
        assert "value" in result["outliers"]
        assert result["outliers"]["value"]["count"] == 1

    def test_no_outliers_when_column_is_uniform(self, agent: MiningAgent) -> None:
        df = pd.DataFrame({"value": [5, 5, 5, 5, 5]})
        result = agent.run(context={"dataframe": df, "goal": "profile"})
        assert "value" not in result["outliers"]


class TestMiningAgentFeatureImportance:
    def test_returns_ranked_scores_summing_to_one(self, agent: MiningAgent, sample_df) -> None:
        result = agent.run(context={"dataframe": sample_df, "goal": "profile"})
        importance = result["feature_importance"]
        assert len(importance) > 0
        total = sum(f["score"] for f in importance)
        assert total == pytest.approx(1.0, abs=1e-3)
        scores = [f["score"] for f in importance]
        assert scores == sorted(scores, reverse=True)

    def test_empty_for_single_numeric_column(self, agent: MiningAgent) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"]})
        result = agent.run(context={"dataframe": df, "goal": "profile"})
        assert result["feature_importance"] == []


class TestMiningAgentClustering:
    def test_clustering_runs_on_sufficient_data(self, agent: MiningAgent, sample_df) -> None:
        result = agent.run(context={"dataframe": sample_df, "goal": "profile"})
        clustering = result["clustering"]
        assert clustering is not None
        assert clustering["k"] >= 2
        assert sum(clustering["cluster_sizes"]) == len(sample_df)
        assert len(clustering["points"]) == len(sample_df)

    def test_clustering_skipped_for_small_dataset(self, agent: MiningAgent) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = agent.run(context={"dataframe": df, "goal": "profile"})
        assert result["clustering"] is None

    def test_clustering_skipped_for_single_numeric_column(self, agent: MiningAgent) -> None:
        df = pd.DataFrame({"a": range(20)})
        result = agent.run(context={"dataframe": df, "goal": "profile"})
        assert result["clustering"] is None
