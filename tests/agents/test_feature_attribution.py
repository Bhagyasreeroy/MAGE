"""
tests/agents/test_feature_attribution.py
─────────────────────────────────────────
Tests for supervised feature attribution — SHAP over a small tree model
(Objective 5 / M6 explainability, FR).

The central assertions are the *conditioning* ones: attribution runs for
classification and regression goals and does **not** run for clustering,
anomaly, or reporting goals. That is what makes it part of the goal-conditioned
pipeline rather than a computation bolted onto every run.

The recovery tests matter too. An attribution ranking that looks plausible but
is wrong would pass any structural check, so these plant a known driver in the
data and assert SHAP actually finds it.
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
def agent() -> MiningAgent:
    return MiningAgent()


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """`driver` determines the label; `noise` is independent of it."""
    rng = np.random.default_rng(42)
    n = 200
    driver = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "driver": driver,
            "noise": rng.normal(0, 1, n),
            "label": (driver > 0).astype(int),
        }
    )


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """`driver` explains the target; `noise` contributes nothing."""
    rng = np.random.default_rng(7)
    n = 200
    driver = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "driver": driver,
            "noise": rng.normal(0, 1, n),
            "outcome": driver * 10 + rng.normal(0, 0.5, n),
        }
    )


def _run(agent: MiningAgent, df: pd.DataFrame, computations: list[str], target: str | None):
    context = {"dataframe": df, "directives": {"computations": computations, "target_column": target}}
    return agent.run(context=context)


class TestAttributionIsGoalConditioned:
    """Strictly opt-in, like every other conditioned computation."""

    def test_runs_when_the_directive_asks_for_it(
        self, agent: MiningAgent, classification_df: pd.DataFrame
    ) -> None:
        result = _run(agent, classification_df, ["shap_attribution"], "label")
        assert result["feature_attribution"]

    def test_absent_without_the_directive(
        self, agent: MiningAgent, classification_df: pd.DataFrame
    ) -> None:
        result = _run(agent, classification_df, ["correlation"], "label")
        assert result["feature_attribution"] == {}

    def test_absent_from_the_unconditioned_default_profile(
        self, agent: MiningAgent, classification_df: pd.DataFrame
    ) -> None:
        """A bare run must keep its original shape — attribution is opt-in only."""
        result = agent.run(context={"dataframe": classification_df})
        assert result["feature_attribution"] == {}

    def test_key_is_always_present_in_the_result_shape(self, agent: MiningAgent) -> None:
        result = agent.run(context={})
        assert "feature_attribution" in result


class TestPlannerIssuesTheDirective:
    """The conditioning has to be wired at the plan level, not just supported."""

    @pytest.fixture
    def planner(self) -> PipelinePlanner:
        return PipelinePlanner()

    @pytest.mark.parametrize("task", [TaskType.classification, TaskType.regression])
    def test_supervised_goals_request_attribution(
        self, planner: PipelinePlanner, task: TaskType
    ) -> None:
        plan = planner.build_plan(task, "target")
        mining = next(s for s in plan if s.agent_name == "MiningAgent")
        assert "shap_attribution" in mining.directives["computations"]

    @pytest.mark.parametrize(
        "task", [TaskType.clustering, TaskType.anomaly_detection, TaskType.reporting]
    )
    def test_unsupervised_goals_do_not(self, planner: PipelinePlanner, task: TaskType) -> None:
        plan = planner.build_plan(task, None)
        mining = next(s for s in plan if s.agent_name == "MiningAgent")
        assert "shap_attribution" not in mining.directives["computations"]

    @pytest.mark.parametrize("task", [TaskType.classification, TaskType.regression])
    def test_supervised_goals_request_the_chart(
        self, planner: PipelinePlanner, task: TaskType
    ) -> None:
        plan = planner.build_plan(task, "target")
        viz = next(s for s in plan if s.agent_name == "VisualizationAgent")
        assert "feature_attribution" in viz.directives["charts"]


class TestAttributionRecoversTheRealDriver:
    """A ranking that is merely well-formed is not good enough."""

    def test_classification_ranks_the_true_driver_first(
        self, agent: MiningAgent, classification_df: pd.DataFrame
    ) -> None:
        result = _run(agent, classification_df, ["shap_attribution"], "label")
        assert result["feature_attribution"]["attributions"][0]["feature"] == "driver"

    def test_regression_ranks_the_true_driver_first(
        self, agent: MiningAgent, regression_df: pd.DataFrame
    ) -> None:
        result = _run(agent, regression_df, ["shap_attribution"], "outcome")
        assert result["feature_attribution"]["attributions"][0]["feature"] == "driver"

    def test_the_driver_dominates_the_noise_feature(
        self, agent: MiningAgent, regression_df: pd.DataFrame
    ) -> None:
        attributions = _run(agent, regression_df, ["shap_attribution"], "outcome")[
            "feature_attribution"
        ]["attributions"]
        scores = {a["feature"]: a["score"] for a in attributions}
        assert scores["driver"] > scores["noise"] * 3


class TestAttributionOutputShape:
    @pytest.fixture
    def attribution(self, agent: MiningAgent, classification_df: pd.DataFrame) -> dict:
        return _run(agent, classification_df, ["shap_attribution"], "label")["feature_attribution"]

    def test_names_the_target(self, attribution: dict) -> None:
        assert attribution["target"] == "label"

    def test_names_the_method_actually_used(self, attribution: dict) -> None:
        """Honesty requirement: never present permutation numbers as SHAP's."""
        assert attribution["method"] in ("shap.TreeExplainer", "permutation_importance")

    def test_detects_the_task_type(self, attribution: dict) -> None:
        assert attribution["task"] == "classification"

    def test_reports_the_model_score_so_weak_fits_are_visible(self, attribution: dict) -> None:
        assert 0.0 <= attribution["model_score"] <= 1.0

    def test_scores_are_normalised_shares(self, attribution: dict) -> None:
        total = sum(a["score"] for a in attribution["attributions"])
        assert total == pytest.approx(1.0, abs=0.01)

    def test_scores_are_non_negative(self, attribution: dict) -> None:
        assert all(a["score"] >= 0 for a in attribution["attributions"])

    def test_ranked_descending(self, attribution: dict) -> None:
        scores = [a["score"] for a in attribution["attributions"]]
        assert scores == sorted(scores, reverse=True)

    def test_regression_target_is_detected_as_continuous(
        self, agent: MiningAgent, regression_df: pd.DataFrame
    ) -> None:
        result = _run(agent, regression_df, ["shap_attribution"], "outcome")
        assert result["feature_attribution"]["task"] == "regression"

    def test_target_is_not_ranked_as_its_own_predictor(
        self, attribution: dict
    ) -> None:
        assert all(a["feature"] != "label" for a in attribution["attributions"])


class TestAttributionDegradesSafely:
    """Attribution is additive — it must never take the pipeline down with it."""

    def test_missing_target_yields_no_attribution(
        self, agent: MiningAgent, classification_df: pd.DataFrame
    ) -> None:
        assert _run(agent, classification_df, ["shap_attribution"], None)["feature_attribution"] == {}

    def test_unknown_target_column_yields_no_attribution(
        self, agent: MiningAgent, classification_df: pd.DataFrame
    ) -> None:
        result = _run(agent, classification_df, ["shap_attribution"], "no_such_column")
        assert result["feature_attribution"] == {}

    def test_too_few_rows_yields_no_attribution(self, agent: MiningAgent) -> None:
        tiny = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "t": [0, 1, 0]})
        assert _run(agent, tiny, ["shap_attribution"], "t")["feature_attribution"] == {}

    def test_single_class_target_yields_no_attribution(self, agent: MiningAgent) -> None:
        rng = np.random.default_rng(1)
        constant = pd.DataFrame(
            {"a": rng.normal(size=50), "b": rng.normal(size=50), "t": [1] * 50}
        )
        assert _run(agent, constant, ["shap_attribution"], "t")["feature_attribution"] == {}

    def test_identifier_columns_are_excluded_from_attribution(self, agent: MiningAgent) -> None:
        """Row ids must not be credited with explaining the target."""
        rng = np.random.default_rng(3)
        n = 120
        driver = rng.normal(size=n)
        df = pd.DataFrame(
            {
                "order_id": range(1, n + 1),
                "driver": driver,
                "label": (driver > 0).astype(int),
            }
        )
        result = _run(agent, df, ["shap_attribution"], "label")
        features = [a["feature"] for a in result["feature_attribution"]["attributions"]]
        assert "order_id" not in features

    def test_a_failure_does_not_break_the_rest_of_the_profile(
        self, agent: MiningAgent, classification_df: pd.DataFrame
    ) -> None:
        result = _run(agent, classification_df, ["shap_attribution", "correlation"], "no_such_col")
        assert result["feature_attribution"] == {}
        assert result["statistics"]
        assert result["correlations"]


class TestAttributionReachesTheNarrativeAndCharts:
    def test_a_pattern_describes_the_top_contributor(
        self, agent: MiningAgent, regression_df: pd.DataFrame
    ) -> None:
        result = _run(agent, regression_df, ["shap_attribution"], "outcome")
        assert any("contributes most to predicting" in p for p in result["patterns"])

    def test_the_pattern_discloses_the_method_and_model_score(
        self, agent: MiningAgent, regression_df: pd.DataFrame
    ) -> None:
        """The reader must be able to weigh the claim, not just read it."""
        result = _run(agent, regression_df, ["shap_attribution"], "outcome")
        pattern = next(p for p in result["patterns"] if "contributes most" in p)
        assert "shap" in pattern.lower() or "permutation" in pattern.lower()
        assert "model score" in pattern.lower()

    def test_a_chart_spec_is_built_for_the_attribution(
        self, agent: MiningAgent, regression_df: pd.DataFrame
    ) -> None:
        mining = _run(agent, regression_df, ["shap_attribution"], "outcome")
        specs = VisualizationAgent().run(
            context={
                "dataframe": regression_df,
                "MiningAgent_output": mining,
                "directives": {"charts": ["feature_attribution"]},
            }
        )["viz_specs"]
        assert any("Feature Attribution" in s.get("title", "") for s in specs)

    def test_the_chart_title_distinguishes_it_from_pca_importance(
        self, agent: MiningAgent, regression_df: pd.DataFrame
    ) -> None:
        mining = _run(agent, regression_df, ["shap_attribution", "feature_importance"], "outcome")
        specs = VisualizationAgent().run(
            context={
                "dataframe": regression_df,
                "MiningAgent_output": mining,
                "directives": {"charts": ["feature_attribution", "feature_importance"]},
            }
        )["viz_specs"]
        titles = [s.get("title", "") for s in specs]
        assert any("Feature Attribution for 'outcome'" in t for t in titles)
        assert any("PCA loading" in t for t in titles)

    def test_chart_falls_back_to_pca_when_attribution_did_not_run(
        self, agent: MiningAgent, regression_df: pd.DataFrame
    ) -> None:
        mining = _run(agent, regression_df, ["feature_importance"], None)
        specs = VisualizationAgent().run(
            context={
                "dataframe": regression_df,
                "MiningAgent_output": mining,
                "directives": {"charts": ["feature_attribution"]},
            }
        )["viz_specs"]
        assert specs and "PCA loading" in specs[0]["title"]
