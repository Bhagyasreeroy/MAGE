"""
tests/agents/test_time_series_decomposition.py
───────────────────────────────────────────────
B1 — seasonal decomposition (M3).

Strictly opt-in: the computation runs only when the plan asks for
``time_series_decomposition`` by name, bypassing MiningAgent's `wants()`
default-on behaviour. That is not a style choice. `computations_run` records
what the plan *requested*, not what produced a result, so a token requested on
a dataset with no dates would be reported as run — on iris, wine and
breast_cancer, none of which has a time axis. The orchestrator therefore asks
for it only after ingestion reports one.

The output quotes the inferred period and how it was inferred, the way SHAP
attribution quotes `method` and `model_score`: a decomposition is only as
trustworthy as its assumed period, and a reader who cannot see the period
cannot judge the result.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.mining_agent import MiningAgent

TOKEN = "time_series_decomposition"


def _monthly_series(periods: int = 48, freq: str = "MS") -> pd.DataFrame:
    """A clean monthly series: rising trend plus a 12-month seasonal cycle."""
    idx = pd.date_range("2020-01-01", periods=periods, freq=freq)
    trend = np.linspace(100.0, 200.0, periods)
    seasonal = 10.0 * np.sin(2 * np.pi * np.arange(periods) / 12.0)
    return pd.DataFrame({
        "month": idx.strftime("%Y-%m-%d"),   # text dates, as a CSV upload delivers
        "passengers": trend + seasonal,
    })


def _run(df: pd.DataFrame, computations: list[str]) -> dict:
    return MiningAgent().run(context={
        "goal": "analyse the seasonality of passengers",
        "dataframe": df,
        "directives": {"task_type": "reporting", "computations": computations},
    })


class TestStrictlyOptIn:
    def test_it_does_not_run_unless_the_plan_asks_for_it(self) -> None:
        result = _run(_monthly_series(), ["descriptive_profile", "missingness"])

        assert result.get("time_series") in (None, {}), (
            "a decomposition must not appear merely because the data could support one"
        )
        assert TOKEN not in result["computations_run"]

    def test_it_does_not_run_on_the_unconditioned_default_profile(self) -> None:
        """
        MiningAgent's `wants()` returns True for everything when no computation
        tokens are given. This computation deliberately bypasses that: a
        standalone profile must not silently start fitting seasonal models.
        """
        result = MiningAgent().run(context={
            "goal": "profile this", "dataframe": _monthly_series(), "directives": {},
        })

        assert result.get("time_series") in (None, {})

    def test_it_runs_when_requested(self) -> None:
        result = _run(_monthly_series(), [TOKEN])

        assert result["time_series"], "the plan asked for it and the data supports it"
        assert TOKEN in result["computations_run"]


class TestDecompositionOutput:
    @pytest.fixture
    def decomposition(self) -> dict:
        return _run(_monthly_series(), [TOKEN])["time_series"]

    def test_it_names_the_columns_it_used(self, decomposition: dict) -> None:
        assert decomposition["time_column"] == "month"
        assert decomposition["column"] == "passengers"

    def test_it_quotes_the_inferred_period_and_its_basis(self, decomposition: dict) -> None:
        """
        A decomposition is only as good as its assumed period. Quoting it — and
        how it was arrived at — is what lets a reader disagree, exactly as
        SHAP's `method` and `model_score` do for attribution.
        """
        assert decomposition["period"] == 12
        assert decomposition["period_basis"], "say how the period was inferred"
        assert "month" in decomposition["period_basis"].lower()

    def test_it_returns_the_three_components(self, decomposition: dict) -> None:
        points = decomposition["points"]
        assert points
        for key in ("t", "observed", "trend", "seasonal", "residual"):
            assert key in points[0], f"missing component: {key}"

    def test_the_components_reconstruct_the_series(self, decomposition: dict) -> None:
        """
        An additive decomposition must satisfy observed = trend + seasonal +
        residual. Asserting the arithmetic rather than the shape is what makes
        this a test of the decomposition and not of the plumbing.
        """
        interior = [
            p for p in decomposition["points"]
            if p["trend"] is not None and p["residual"] is not None
        ]
        assert interior, "trend is undefined at the edges, but not everywhere"
        for p in interior[:20]:
            assert p["observed"] == pytest.approx(
                p["trend"] + p["seasonal"] + p["residual"], abs=1e-6
            )

    def test_it_recovers_the_seasonal_amplitude(self, decomposition: dict) -> None:
        """The fixture's seasonal component has amplitude 10."""
        seasonal = [p["seasonal"] for p in decomposition["points"]]
        assert max(seasonal) == pytest.approx(10.0, abs=1.5)
        assert min(seasonal) == pytest.approx(-10.0, abs=1.5)


class TestDecliningCleanly:
    """Real series have gaps and real datasets have no dates at all. Each
    refusal must say which, rather than raising or returning something empty
    that a reader cannot interpret."""

    def test_no_time_axis_declines_with_a_reason(self) -> None:
        df = pd.DataFrame({"a": range(40), "b": range(40)})
        result = _run(df, [TOKEN])

        assert not result["time_series"].get("points")
        assert "no time" in result["time_series"]["skipped"].lower()

    def test_too_few_observations_declines_with_a_reason(self) -> None:
        """`seasonal_decompose` needs two full periods; below that it raises."""
        result = _run(_monthly_series(periods=8), [TOKEN])

        skipped = result["time_series"]["skipped"].lower()
        assert "period" in skipped or "observation" in skipped

    def test_no_numeric_column_declines_with_a_reason(self) -> None:
        df = pd.DataFrame({
            "month": pd.date_range("2020-01-01", periods=30, freq="MS").strftime("%Y-%m-%d"),
            "label": [f"cat_{i % 3}" for i in range(30)],
        })
        result = _run(df, [TOKEN])

        assert "numeric" in result["time_series"]["skipped"].lower()

    def test_a_regular_series_is_not_reported_as_resampled(self) -> None:
        """
        The control the resampling test needs to mean anything.

        Calendar months are 28-31 days apart, so a *perfectly regular* monthly
        series has varying day-spacing. A naive "do all the gaps match?" check
        calls that irregular and tells the reader their series was altered when
        nothing was touched — which is worse than saying nothing, because it is
        a false statement about the data.
        """
        result = _run(_monthly_series(), [TOKEN])

        assert result["time_series"]["resampled"] == "", (
            "a regular monthly series was not resampled and must not claim it was"
        )

    def test_an_irregular_series_says_it_was_resampled(self) -> None:
        """
        Gaps are ordinary. Resampling is a defensible response and silently
        resampling is not — the reader is owed the fact that the series they
        are looking at is not quite the series they uploaded.
        """
        df = _monthly_series(periods=48)
        df = df.drop(index=[5, 11, 23]).reset_index(drop=True)   # punch holes
        result = _run(df, [TOKEN])
        ts = result["time_series"]

        assert ts.get("points"), "a gapped series should still decompose"
        assert ts["resampled"], "say that the series was regularised"


class TestItReachesTheNarrative:
    def test_a_pattern_describes_the_decomposition(self) -> None:
        result = _run(_monthly_series(), [TOKEN])

        assert any("season" in p.lower() for p in result["patterns"]), result["patterns"]


class TestTheOrchestratorGatesItOnTheData:
    """
    The decision that keeps the evaluation honest.

    `computations_run` reports what the plan *requested*. Adding
    `time_series_decomposition` to reporting's directives unconditionally would
    therefore report it as run on iris, wine, breast_cancer and diabetes — none
    of which contains a date — and either drop precision or force an edit to the
    hand-labelled relevance set. Editing ground truth in the direction that
    raises your own score is the move `evaluation/metrics.py` warns about in its
    own change log.

    So the token is requested only once IngestionAgent has confirmed a time axis
    exists, at the same point the orchestrator already refines the target column
    from the real schema.
    """

    @staticmethod
    def _mining_directives(result: dict) -> dict:
        step = next(s for s in result["steps"] if s["agent_name"] == "MiningAgent")
        return set(step["output"].get("computations_run") or [])

    def _run(self, tmp_path, df: pd.DataFrame, goal: str) -> dict:
        from agents.orchestrator import OrchestratorAgent

        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        return OrchestratorAgent().run(goal=goal, data={"source": str(csv)})

    def test_a_temporal_dataset_and_a_temporal_goal_requests_it(self, tmp_path) -> None:
        result = self._run(
            tmp_path, _monthly_series(), "analyse the seasonality of passengers over time",
        )

        assert TOKEN in self._mining_directives(result)

    def test_a_dataset_without_dates_does_not_request_it(self, tmp_path) -> None:
        """The harness case: reporting over data with no time axis."""
        df = pd.DataFrame({
            "sepal_length": np.linspace(4.0, 8.0, 60),
            "petal_width": np.linspace(0.1, 2.5, 60),
        })
        result = self._run(tmp_path, df, "give me a general overview of this dataset")

        assert TOKEN not in self._mining_directives(result)

    def test_a_clustering_goal_on_temporal_data_does_not_request_it(self, tmp_path) -> None:
        """
        Goal conditioning still governs. A time axis makes a decomposition
        *possible*; it is the goal that makes it *relevant*, and a clustering
        run has not asked to be told about seasonality.
        """
        result = self._run(
            tmp_path, _monthly_series(), "find natural clusters and segments in this data",
        )

        assert TOKEN not in self._mining_directives(result)


class TestItReachesTheChart:
    """A computation nobody can see is half-built. The decomposition gets its
    own chart rather than being folded into the existing `line` spec, because
    the three components have genuinely different scales — a residual sits
    around zero while the observed series may be in the hundreds — and one
    y-axis for both would either flatten the residual to nothing or be a
    dual-axis chart, which is worse."""

    def _viz(self, df: pd.DataFrame) -> list[dict]:
        from agents.visualization_agent import VisualizationAgent

        mining = _run(df, [TOKEN])
        out = VisualizationAgent().run(context={
            "goal": "analyse the seasonality of passengers",
            "dataframe": df,
            "directives": {"task_type": "reporting", "charts": ["seasonal_decomposition"]},
            "MiningAgent_output": mining,
        })
        return out["viz_specs"]

    def test_a_decomposition_spec_is_emitted(self) -> None:
        specs = self._viz(_monthly_series())
        spec = next((s for s in specs if s["type"] == "seasonal_decomposition"), None)

        assert spec is not None, [s["type"] for s in specs]
        assert spec["points"]
        assert spec["period"] == 12

    def test_the_title_carries_the_period_so_it_can_be_judged(self) -> None:
        spec = next(s for s in self._viz(_monthly_series()) if s["type"] == "seasonal_decomposition")

        assert "12" in spec["title"]

    def test_no_decomposition_no_spec(self) -> None:
        """The builder returns None rather than an empty chart when Mining
        declined — the router then falls back to a chart that can be drawn."""
        df = pd.DataFrame({"a": range(40), "b": range(40)})
        specs = self._viz(df)

        assert not any(s["type"] == "seasonal_decomposition" for s in specs)
