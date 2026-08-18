"""
tests/agents/test_chart_builders.py
────────────────────────────────────
Tests for the dedicated M4 chart builders.

Four chart types named by the planner — ``grouped_bar``, ``box_by_class``,
``pairplot`` and ``highlighted_scatter`` — were previously *routed to
approximations*: a grouped bar became a plain category-frequency bar, a
box-by-class became a plain boxplot that ignored the class entirely, a pairplot
became a single cluster scatter, and a highlighted scatter became an unmarked
one. The chart the planner asked for and the chart the user saw were different
charts, which is a caveat the project would otherwise have to disclose.

These tests pin the real builders. Each asserts the *structure the render layer
needs* and the *numbers against pandas*, so a builder that emits a
well-formed-but-wrong chart fails. Two further types, ``violin`` and ``line``,
were absent entirely and are covered here too.

The final class is the backward-compatibility guard: the unconditioned default
chart set must not change, because several older tests depend on it.
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


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def agent() -> VisualizationAgent:
    return VisualizationAgent()


@pytest.fixture
def labelled_df() -> pd.DataFrame:
    """A classification-shaped frame: categorical feature, numerics, a label."""
    rng = np.random.default_rng(11)
    n = 90
    tenure = rng.normal(30, 8, size=n)
    return pd.DataFrame(
        {
            "region": rng.choice(["East", "West", "North"], size=n),
            "tenure": tenure.round(2),
            "spend": (tenure * 3 + rng.normal(0, 5, size=n)).round(2),
            "visits": rng.integers(1, 20, size=n),
            "churned": rng.choice(["yes", "no"], size=n),
        }
    )


@pytest.fixture
def outlier_df() -> pd.DataFrame:
    """Two correlated numerics with a handful of planted extreme rows."""
    rng = np.random.default_rng(3)
    n = 60
    base = rng.normal(50, 5, size=n)
    df = pd.DataFrame({"amount": base.round(2), "score": (base * 1.4 + rng.normal(0, 2, size=n)).round(2)})
    # Plant unambiguous IQR outliers.
    df.loc[0, "amount"] = 900.0
    df.loc[1, "amount"] = 950.0
    df.loc[2, "amount"] = -400.0
    return df


@pytest.fixture
def temporal_df() -> pd.DataFrame:
    """A dated series — the only shape a line chart is meaningful for."""
    rng = np.random.default_rng(5)
    n = 40
    return pd.DataFrame(
        {
            "order_date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "revenue": (np.arange(n) * 3 + rng.normal(0, 4, size=n)).round(2),
            "units": rng.integers(1, 30, size=n),
        }
    )


def _mining(df: pd.DataFrame, task: TaskType, target: str | None = None) -> dict:
    steps = PipelinePlanner().build_plan(task, target_column=target)
    directives = next(s for s in steps if s.agent_name == "MiningAgent").directives
    return MiningAgent().run(context={"dataframe": df, "directives": directives})


def _viz(agent: VisualizationAgent, df: pd.DataFrame, mining: dict, charts: list[str]) -> list[dict]:
    result = agent.run(
        context={
            "dataframe": df,
            "goal": "test goal",
            "MiningAgent_output": mining,
            "directives": {"charts": charts},
        }
    )
    return result["viz_specs"]


def _of_type(specs: list[dict], ctype: str) -> list[dict]:
    return [s for s in specs if s["type"] == ctype]


# ── grouped_bar ──────────────────────────────────────────────────────────────


class TestGroupedBar:
    def test_emits_a_real_grouped_bar_not_a_plain_bar(self, agent, labelled_df) -> None:
        mining = _mining(labelled_df, TaskType.classification, target="churned")
        specs = _viz(agent, labelled_df, mining, ["grouped_bar"])
        grouped = _of_type(specs, "grouped_bar")
        assert grouped, "grouped_bar directive must produce a grouped_bar spec, not a stand-in"

    def test_one_series_per_target_class(self, agent, labelled_df) -> None:
        mining = _mining(labelled_df, TaskType.classification, target="churned")
        spec = _of_type(_viz(agent, labelled_df, mining, ["grouped_bar"]), "grouped_bar")[0]
        assert {s["name"] for s in spec["series"]} == {"yes", "no"}

    def test_counts_match_a_pandas_crosstab(self, agent, labelled_df) -> None:
        mining = _mining(labelled_df, TaskType.classification, target="churned")
        spec = _of_type(_viz(agent, labelled_df, mining, ["grouped_bar"]), "grouped_bar")[0]
        expected = pd.crosstab(labelled_df[spec["x_label"]], labelled_df["churned"])
        for series in spec["series"]:
            for category, value in zip(spec["categories"], series["values"]):
                assert value == int(expected.loc[category, series["name"]])

    def test_series_values_align_with_categories(self, agent, labelled_df) -> None:
        mining = _mining(labelled_df, TaskType.classification, target="churned")
        spec = _of_type(_viz(agent, labelled_df, mining, ["grouped_bar"]), "grouped_bar")[0]
        for series in spec["series"]:
            assert len(series["values"]) == len(spec["categories"])

    def test_prefers_a_low_cardinality_feature_over_a_timestamp(self, agent) -> None:
        """
        Found end-to-end: on an uploaded CSV the date column is text, so the
        profiler types it *categorical*, and it was being chosen as the grouped
        bar's feature. A grouped bar over 400 distinct timestamps — truncated
        to its 8 busiest — is noise. Prefer the feature a reader can actually
        compare across.
        """
        rng = np.random.default_rng(21)
        n = 200
        df = pd.DataFrame(
            {
                "order_date": pd.date_range("2026-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M"),
                "region": rng.choice(["East", "West", "North"], size=n),
                "spend": rng.normal(100, 15, size=n).round(2),
                "churned": rng.choice(["yes", "no"], size=n),
            }
        )
        mining = _mining(df, TaskType.classification, target="churned")
        spec = _of_type(_viz(agent, df, mining, ["grouped_bar"]), "grouped_bar")[0]
        assert spec["x_label"] == "region"

    def test_skips_a_feature_with_too_many_categories(self, agent) -> None:
        rng = np.random.default_rng(22)
        n = 120
        df = pd.DataFrame(
            {
                "ticket": [f"T-{i:04d}" for i in range(n)],  # unique per row
                "spend": rng.normal(50, 8, size=n).round(2),
                "churned": rng.choice(["yes", "no"], size=n),
            }
        )
        mining = _mining(df, TaskType.classification, target="churned")
        specs = _viz(agent, df, mining, ["grouped_bar"])
        assert not _of_type(specs, "grouped_bar"), "a per-row identifier is not a groupable feature"

    def test_falls_back_to_plain_bar_without_a_target(self, agent, labelled_df) -> None:
        """No target column → nothing to group by; a plain bar is the honest chart."""
        mining = MiningAgent().run(context={"dataframe": labelled_df, "goal": "profile"})
        specs = _viz(agent, labelled_df, mining, ["grouped_bar"])
        assert specs, "must still produce something rather than an empty slot"
        assert not _of_type(specs, "grouped_bar")
        assert _of_type(specs, "bar")


# ── box_by_class ─────────────────────────────────────────────────────────────


class TestBoxByClass:
    def test_emits_a_real_box_by_class(self, agent, labelled_df) -> None:
        mining = _mining(labelled_df, TaskType.classification, target="churned")
        specs = _viz(agent, labelled_df, mining, ["box_by_class"])
        assert _of_type(specs, "box_by_class"), "must not fall back to a class-blind boxplot"

    def test_one_group_per_class_with_five_number_summary(self, agent, labelled_df) -> None:
        mining = _mining(labelled_df, TaskType.classification, target="churned")
        spec = _of_type(_viz(agent, labelled_df, mining, ["box_by_class"]), "box_by_class")[0]
        assert {g["label"] for g in spec["groups"]} == {"yes", "no"}
        for group in spec["groups"]:
            for key in ("min", "q1", "median", "q3", "max", "count"):
                assert group[key] is not None
            assert group["min"] <= group["q1"] <= group["median"] <= group["q3"] <= group["max"]

    def test_quartiles_match_pandas_per_class(self, agent, labelled_df) -> None:
        mining = _mining(labelled_df, TaskType.classification, target="churned")
        spec = _of_type(_viz(agent, labelled_df, mining, ["box_by_class"]), "box_by_class")[0]
        column = spec["column"]
        for group in spec["groups"]:
            series = labelled_df.loc[labelled_df["churned"] == group["label"], column].dropna()
            assert group["count"] == len(series)
            assert group["median"] == pytest.approx(float(series.median()), abs=1e-6)
            assert group["q1"] == pytest.approx(float(series.quantile(0.25)), abs=1e-6)

    def test_boxes_a_numeric_column_not_the_target(self, agent, labelled_df) -> None:
        mining = _mining(labelled_df, TaskType.classification, target="churned")
        spec = _of_type(_viz(agent, labelled_df, mining, ["box_by_class"]), "box_by_class")[0]
        assert spec["column"] != "churned"
        assert pd.api.types.is_numeric_dtype(labelled_df[spec["column"]])

    def test_falls_back_to_plain_boxplot_without_a_target(self, agent, outlier_df) -> None:
        mining = MiningAgent().run(context={"dataframe": outlier_df, "goal": "profile"})
        specs = _viz(agent, outlier_df, mining, ["box_by_class"])
        assert not _of_type(specs, "box_by_class")
        assert _of_type(specs, "boxplot")


# ── pairplot ─────────────────────────────────────────────────────────────────


class TestPairplot:
    def test_emits_a_real_pairplot_not_a_cluster_scatter(self, agent, labelled_df) -> None:
        mining = _mining(labelled_df, TaskType.clustering)
        specs = _viz(agent, labelled_df, mining, ["pairplot"])
        assert _of_type(specs, "pairplot"), "pairplot must not be served by a single cluster scatter"

    def test_covers_every_pair_of_the_selected_columns(self, agent, labelled_df) -> None:
        mining = _mining(labelled_df, TaskType.clustering)
        spec = _of_type(_viz(agent, labelled_df, mining, ["pairplot"]), "pairplot")[0]
        n = len(spec["columns"])
        assert n >= 2
        assert len(spec["pairs"]) == n * (n - 1) // 2
        seen = {(p["x_label"], p["y_label"]) for p in spec["pairs"]}
        assert len(seen) == len(spec["pairs"]), "pairs must be distinct"

    def test_each_pair_carries_points_and_a_correlation(self, agent, labelled_df) -> None:
        mining = _mining(labelled_df, TaskType.clustering)
        spec = _of_type(_viz(agent, labelled_df, mining, ["pairplot"]), "pairplot")[0]
        for pair in spec["pairs"]:
            assert pair["points"], "an empty panel is not a pairplot panel"
            assert -1.0 <= pair["r"] <= 1.0
            assert {"x", "y"} <= set(pair["points"][0])

    def test_correlation_matches_pandas(self, agent, labelled_df) -> None:
        mining = _mining(labelled_df, TaskType.clustering)
        spec = _of_type(_viz(agent, labelled_df, mining, ["pairplot"]), "pairplot")[0]
        pair = spec["pairs"][0]
        expected = labelled_df[[pair["x_label"], pair["y_label"]]].dropna().corr().iloc[0, 1]
        assert pair["r"] == pytest.approx(float(expected), abs=1e-2)

    def test_falls_back_when_too_few_numeric_columns(self, agent) -> None:
        df = pd.DataFrame({"only": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "tag": list("abcabc")})
        mining = MiningAgent().run(context={"dataframe": df, "goal": "profile"})
        specs = _viz(agent, df, mining, ["pairplot"])
        assert not _of_type(specs, "pairplot"), "a pairplot needs at least two numeric columns"


# ── highlighted_scatter ──────────────────────────────────────────────────────


class TestHighlightedScatter:
    def test_emits_a_real_highlighted_scatter(self, agent, outlier_df) -> None:
        mining = _mining(outlier_df, TaskType.anomaly_detection)
        specs = _viz(agent, outlier_df, mining, ["highlighted_scatter"])
        assert _of_type(specs, "highlighted_scatter"), "must not degrade to an unmarked scatter"

    def test_every_point_carries_an_outlier_flag(self, agent, outlier_df) -> None:
        mining = _mining(outlier_df, TaskType.anomaly_detection)
        spec = _of_type(_viz(agent, outlier_df, mining, ["highlighted_scatter"]), "highlighted_scatter")[0]
        assert spec["points"]
        for point in spec["points"]:
            assert isinstance(point["outlier"], bool)

    def test_planted_outliers_are_the_flagged_ones(self, agent, outlier_df) -> None:
        mining = _mining(outlier_df, TaskType.anomaly_detection)
        spec = _of_type(_viz(agent, outlier_df, mining, ["highlighted_scatter"]), "highlighted_scatter")[0]
        flagged = [p for p in spec["points"] if p["outlier"]]
        assert flagged, "the planted extreme rows must be highlighted"
        assert spec["highlighted_count"] == len(flagged)
        # The planted rows are far outside the bulk; every flagged point must be
        # one of them rather than an ordinary row mislabelled.
        bounds = mining["outliers"][spec["x_label"]]["bounds"]
        for point in flagged:
            assert point["x"] < bounds[0] or point["x"] > bounds[1]

    def test_unflagged_points_are_inside_the_bounds(self, agent, outlier_df) -> None:
        mining = _mining(outlier_df, TaskType.anomaly_detection)
        spec = _of_type(_viz(agent, outlier_df, mining, ["highlighted_scatter"]), "highlighted_scatter")[0]
        bounds = mining["outliers"][spec["x_label"]]["bounds"]
        for point in spec["points"]:
            if not point["outlier"]:
                assert bounds[0] <= point["x"] <= bounds[1]

    def test_falls_back_to_plain_scatter_without_outliers(self, agent) -> None:
        # Evenly spaced, so there is genuinely nothing outside 1.5x IQR. A
        # normal sample is the wrong fixture here — 40 draws from N(10,1)
        # reliably contain a couple of IQR outliers, which is the very thing
        # this test needs absent.
        clean = np.linspace(0.0, 10.0, 40)
        df = pd.DataFrame({"a": clean.round(3), "b": (clean * 2).round(3)})
        mining = MiningAgent().run(context={"dataframe": df, "goal": "profile"})
        specs = _viz(agent, df, mining, ["highlighted_scatter"])
        assert not _of_type(specs, "highlighted_scatter")
        assert _of_type(specs, "scatter")


# ── violin ───────────────────────────────────────────────────────────────────


class TestViolin:
    def test_emits_a_violin_spec(self, agent, labelled_df) -> None:
        mining = MiningAgent().run(context={"dataframe": labelled_df, "goal": "profile"})
        specs = _viz(agent, labelled_df, mining, ["violin"])
        assert _of_type(specs, "violin")

    def test_bands_describe_a_density_profile(self, agent, labelled_df) -> None:
        mining = MiningAgent().run(context={"dataframe": labelled_df, "goal": "profile"})
        spec = _of_type(_viz(agent, labelled_df, mining, ["violin"]), "violin")[0]
        assert len(spec["bands"]) >= 3
        widths = [b["width"] for b in spec["bands"]]
        assert all(0.0 <= w <= 1.0 for w in widths), "widths are normalised for the renderer"
        assert max(widths) == pytest.approx(1.0), "the modal band anchors the scale at 1.0"

    def test_carries_the_five_number_summary_for_the_overlay(self, agent, labelled_df) -> None:
        mining = MiningAgent().run(context={"dataframe": labelled_df, "goal": "profile"})
        spec = _of_type(_viz(agent, labelled_df, mining, ["violin"]), "violin")[0]
        assert spec["min"] <= spec["q1"] <= spec["median"] <= spec["q3"] <= spec["max"]

    def test_band_counts_sum_to_the_observation_count(self, agent, labelled_df) -> None:
        mining = MiningAgent().run(context={"dataframe": labelled_df, "goal": "profile"})
        spec = _of_type(_viz(agent, labelled_df, mining, ["violin"]), "violin")[0]
        total = sum(b["count"] for b in spec["bands"])
        assert total == int(labelled_df[spec["column"]].notna().sum())

    def test_skipped_for_a_constant_column(self, agent) -> None:
        df = pd.DataFrame({"flat": [7.0] * 30, "tag": ["a", "b"] * 15})
        mining = MiningAgent().run(context={"dataframe": df, "goal": "profile"})
        specs = _viz(agent, df, mining, ["violin"])
        assert not _of_type(specs, "violin"), "a constant column has no distribution shape"


# ── line ─────────────────────────────────────────────────────────────────────


class TestLine:
    def test_emits_a_line_spec_for_a_dated_frame(self, agent, temporal_df) -> None:
        mining = MiningAgent().run(context={"dataframe": temporal_df, "goal": "profile"})
        specs = _viz(agent, temporal_df, mining, ["line"])
        assert _of_type(specs, "line")

    def test_points_are_ordered_by_time(self, agent, temporal_df) -> None:
        mining = MiningAgent().run(context={"dataframe": temporal_df, "goal": "profile"})
        spec = _of_type(_viz(agent, temporal_df, mining, ["line"]), "line")[0]
        xs = [p["x"] for p in spec["points"]]
        assert xs == sorted(xs), "a trend line drawn out of order is a scribble"

    def test_uses_the_datetime_column_as_the_axis(self, agent, temporal_df) -> None:
        mining = MiningAgent().run(context={"dataframe": temporal_df, "goal": "profile"})
        spec = _of_type(_viz(agent, temporal_df, mining, ["line"]), "line")[0]
        assert spec["x_label"] == "order_date"
        assert spec["y_label"] in {"revenue", "units"}

    def test_values_match_the_source_frame(self, agent, temporal_df) -> None:
        mining = MiningAgent().run(context={"dataframe": temporal_df, "goal": "profile"})
        spec = _of_type(_viz(agent, temporal_df, mining, ["line"]), "line")[0]
        expected = (
            temporal_df[["order_date", spec["y_label"]]]
            .dropna()
            .sort_values("order_date")
        )
        assert len(spec["points"]) == len(expected)
        assert spec["points"][0]["y"] == pytest.approx(float(expected.iloc[0][spec["y_label"]]))

    def test_absent_without_a_datetime_column(self, agent, labelled_df) -> None:
        mining = MiningAgent().run(context={"dataframe": labelled_df, "goal": "profile"})
        specs = _viz(agent, labelled_df, mining, ["line"])
        assert not _of_type(specs, "line"), "no time axis → no trend line"

    def test_built_for_string_dates_as_delivered_by_csv_ingestion(self, agent, temporal_df) -> None:
        """
        The shape that actually arrives from an upload.

        A CSV round-trip leaves the date column as strings, not ``datetime64``.
        Requiring the parsed dtype would mean this chart never fires on real
        user data — it would only ever appear in tests that build the frame by
        hand, which is the worst kind of passing feature.
        """
        as_uploaded = temporal_df.copy()
        as_uploaded["order_date"] = as_uploaded["order_date"].dt.strftime("%Y-%m-%d")
        # pandas 3 gives these StringDtype rather than object; what matters to
        # this test is only that they are no longer parsed datetimes.
        assert not pd.api.types.is_datetime64_any_dtype(as_uploaded["order_date"])

        mining = MiningAgent().run(context={"dataframe": as_uploaded, "goal": "profile"})
        specs = _viz(agent, as_uploaded, mining, ["line"])
        line = _of_type(specs, "line")
        assert line, "string dates must still yield a trend line"
        assert line[0]["x_label"] == "order_date"
        xs = [p["x"] for p in line[0]["points"]]
        assert xs == sorted(xs)

    def test_short_category_codes_are_not_mistaken_for_dates(self, agent) -> None:
        """Guard on the parse: '1'/'2'/'3' must not become a time axis."""
        df = pd.DataFrame(
            {"code": [str(i % 5) for i in range(30)], "value": np.linspace(1, 30, 30).round(2)}
        )
        mining = MiningAgent().run(context={"dataframe": df, "goal": "profile"})
        specs = _viz(agent, df, mining, ["line"])
        assert not _of_type(specs, "line")


# ── Planner wiring ───────────────────────────────────────────────────────────


class TestPlannerRequestsTheNewCharts:
    def _charts(self, task: TaskType) -> list[str]:
        steps = PipelinePlanner().build_plan(task)
        return next(s for s in steps if s.agent_name == "VisualizationAgent").directives["charts"]

    def test_reporting_requests_violin_and_line(self) -> None:
        """Both are distribution/trend profiling charts; reporting is their home."""
        charts = self._charts(TaskType.reporting)
        assert "violin" in charts
        assert "line" in charts

    def test_the_four_previously_approximated_charts_are_still_requested(self) -> None:
        assert "grouped_bar" in self._charts(TaskType.classification)
        assert "box_by_class" in self._charts(TaskType.classification)
        assert "pairplot" in self._charts(TaskType.clustering)
        assert "highlighted_scatter" in self._charts(TaskType.anomaly_detection)

    def test_violin_and_line_do_not_leak_into_supervised_goals(self) -> None:
        for task in (TaskType.classification, TaskType.regression):
            charts = self._charts(task)
            assert "violin" not in charts
            assert "line" not in charts


# ── Backward compatibility ───────────────────────────────────────────────────


class TestDefaultProfileUnchanged:
    """The unconditioned chart set is depended on by older tests; it must not move."""

    def test_default_set_contains_no_new_chart_types(self, agent, labelled_df) -> None:
        mining = MiningAgent().run(context={"dataframe": labelled_df, "goal": "profile"})
        result = agent.run(
            context={"dataframe": labelled_df, "goal": "profile", "MiningAgent_output": mining}
        )
        types = {s["type"] for s in result["viz_specs"]}
        assert types <= {
            "correlation_heatmap",
            "cluster_scatter",
            "feature_importance",
            "histogram",
            "boxplot",
            "bar",
        }

    def test_default_set_is_still_non_empty(self, agent, labelled_df) -> None:
        mining = MiningAgent().run(context={"dataframe": labelled_df, "goal": "profile"})
        result = agent.run(
            context={"dataframe": labelled_df, "goal": "profile", "MiningAgent_output": mining}
        )
        assert len(result["viz_specs"]) > 0

    def test_conditioned_run_never_returns_an_empty_slot(self, agent, labelled_df) -> None:
        """Every planner chart directive must yield at least one drawable spec."""
        for task in TaskType:
            steps = PipelinePlanner().build_plan(task)
            charts = next(s for s in steps if s.agent_name == "VisualizationAgent").directives["charts"]
            mining = _mining(labelled_df, task, target="churned")
            specs = _viz(agent, labelled_df, mining, charts)
            assert specs, f"{task.value} produced no charts at all"


# ── Default categorical bars (found by the browser walkthrough) ──────────────


class TestCategoricalBarsSkipNonGroupableColumns:
    """
    `_categorical_bar_specs` is the *default* bar builder, and it also backs the
    `grouped_bar` directive when no target exists — which is every reporting
    run. It picked columns purely in dataset order, so on an uploaded CSV it
    chose the text date column and drew "Top values in 'order_date'": five
    unique timestamps, each with a count of 1. A meaningless chart, in the most
    common goal's default view.

    Same root cause as the grouped_bar fix above (CSV dates are typed
    *categorical* by the profiler) but in the older builder, which had no guard.
    """

    def test_timestamp_column_is_not_barred(self, agent) -> None:
        rng = np.random.default_rng(31)
        n = 200
        df = pd.DataFrame(
            {
                "order_date": pd.date_range("2026-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M"),
                "region": rng.choice(["East", "West", "North"], size=n),
                "spend": rng.normal(100, 12, size=n).round(2),
            }
        )
        mining = MiningAgent().run(context={"dataframe": df, "goal": "profile"})
        result = agent.run(context={"dataframe": df, "goal": "profile", "MiningAgent_output": mining})
        barred = [s["title"] for s in result["viz_specs"] if s["type"] == "bar"]
        assert not any("order_date" in t for t in barred), f"charted the time axis: {barred}"
        assert any("region" in t for t in barred), "should still bar the real categorical"

    def test_per_row_identifier_is_not_barred(self, agent) -> None:
        n = 60
        df = pd.DataFrame(
            {
                "ticket": [f"T-{i:04d}" for i in range(n)],  # unique per row
                "status": ["open", "closed", "pending"] * 20,
                "amount": np.linspace(1, 60, n).round(2),
            }
        )
        mining = MiningAgent().run(context={"dataframe": df, "goal": "profile"})
        result = agent.run(context={"dataframe": df, "goal": "profile", "MiningAgent_output": mining})
        barred = [s["title"] for s in result["viz_specs"] if s["type"] == "bar"]
        assert not any("ticket" in t for t in barred), f"charted a per-row id: {barred}"
        assert any("status" in t for t in barred)

    def test_a_frame_of_only_ungroupable_columns_yields_no_bars(self, agent) -> None:
        """Better no bar chart than a meaningless one."""
        n = 40
        df = pd.DataFrame(
            {
                "uid": [f"u{i}" for i in range(n)],
                "value": np.linspace(0, 39, n).round(2),
            }
        )
        mining = MiningAgent().run(context={"dataframe": df, "goal": "profile"})
        result = agent.run(context={"dataframe": df, "goal": "profile", "MiningAgent_output": mining})
        assert not [s for s in result["viz_specs"] if s["type"] == "bar"]
