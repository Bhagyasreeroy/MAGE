"""
tests/evaluation/test_baseline_and_datasets.py
───────────────────────────────────────────────
Tests for the two inputs the experiment depends on being trustworthy.

The **baseline** tests pin down its defining property — goal-invariance. If the
baseline ever varied by goal, the headline divergence comparison would be
meaningless, so that invariance is asserted rather than assumed.

The **dataset** tests pin down determinism and the planted structure. The suite
is only a fair test if clustering, anomaly, and classification goals each have
something real to find; a dataset that silently lost its planted outliers would
quietly understate MAGE without failing anything.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from evaluation.baseline import (
    BASELINE_COMPUTATIONS,
    baseline_computations,
    ydata_profiling_available,
)
from evaluation.datasets import EvalDataset, load_eval_datasets


class TestBaselineIsGoalInvariant:
    """The property the whole comparison rests on."""

    def test_identical_set_for_every_goal(self) -> None:
        goals = [
            "Predict churn",
            "Find natural clusters",
            "Detect fraudulent transactions",
            "Summarise the data quality",
            "",
        ]
        results = [baseline_computations(goal=g) for g in goals]
        assert all(r == results[0] for r in results)

    def test_identical_set_for_different_dataframes(self) -> None:
        a = pd.DataFrame({"x": [1, 2, 3]})
        b = pd.DataFrame({"p": ["a", "b"], "q": [1.5, 2.5], "r": [True, False]})
        assert baseline_computations(a) == baseline_computations(b)

    def test_matches_the_pinned_constant(self) -> None:
        assert baseline_computations() == set(BASELINE_COMPUTATIONS)

    def test_returns_a_copy_so_callers_cannot_mutate_the_constant(self) -> None:
        got = baseline_computations()
        got.add("injected_computation")
        assert "injected_computation" not in BASELINE_COMPUTATIONS
        assert baseline_computations() == set(BASELINE_COMPUTATIONS)

    def test_excludes_computations_ydata_profiling_does_not_perform(self) -> None:
        """Guards against quietly crediting the baseline with MAGE's specialised work."""
        never_performed = {
            "kmeans", "dbscan", "silhouette", "standardize",
            "isolation_forest", "class_balance", "linearity_check", "feature_importance",
        }
        assert not (set(BASELINE_COMPUTATIONS) & never_performed)

    def test_includes_the_generic_work_it_genuinely_does(self) -> None:
        """Understating the baseline would unfairly flatter MAGE."""
        assert {"descriptive_profile", "missingness", "distribution", "correlation"} <= set(
            BASELINE_COMPUTATIONS
        )


class TestLiveBaselineDegradesGracefully:
    def test_availability_check_never_raises(self) -> None:
        assert isinstance(ydata_profiling_available(), bool)

    @pytest.mark.skipif(
        not ydata_profiling_available(), reason="ydata-profiling is not installed (optional)"
    )
    def test_live_derivation_matches_the_pinned_spec(self) -> None:
        """Recorded in evaluation/BASELINE_VALIDATION.md; re-checked when available."""
        from evaluation.baseline import profile_computations

        df = load_eval_datasets()[0].build()
        assert profile_computations(df) == set(BASELINE_COMPUTATIONS)


class TestDatasetSuite:
    def test_suite_is_not_empty(self) -> None:
        assert len(load_eval_datasets()) >= 3

    def test_names_are_unique(self) -> None:
        names = [d.name for d in load_eval_datasets()]
        assert len(names) == len(set(names))

    def test_ordering_is_stable_across_calls(self) -> None:
        assert [d.name for d in load_eval_datasets()] == [
            d.name for d in load_eval_datasets()
        ]

    @pytest.mark.parametrize("dataset", load_eval_datasets(), ids=lambda d: d.name)
    def test_every_dataset_builds_a_usable_frame(self, dataset: EvalDataset) -> None:
        df = dataset.build()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 1

    @pytest.mark.parametrize("dataset", load_eval_datasets(), ids=lambda d: d.name)
    def test_build_is_deterministic(self, dataset: EvalDataset) -> None:
        """Reproducibility is a headline claim — the data must not drift between runs."""
        pd.testing.assert_frame_equal(dataset.build(), dataset.build())

    @pytest.mark.parametrize("dataset", load_eval_datasets(), ids=lambda d: d.name)
    def test_declared_target_column_exists(self, dataset: EvalDataset) -> None:
        if dataset.target_column:
            assert dataset.target_column in dataset.build().columns

    def test_materialize_writes_a_readable_csv(self, tmp_path) -> None:
        dataset = load_eval_datasets()[0]
        path = dataset.materialize(data_dir=tmp_path)
        assert path.exists()
        assert len(pd.read_csv(path)) == len(dataset.build())


class TestSyntheticDatasetHasPlantedStructure:
    """Without this structure the specialised goals would be scored on noise."""

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        (dataset,) = [d for d in load_eval_datasets() if d.name == "customer_orders"]
        return dataset.build()

    def test_has_outliers_for_anomaly_goals(self, df: pd.DataFrame) -> None:
        revenue = df["revenue"].dropna()
        q1, q3 = revenue.quantile(0.25), revenue.quantile(0.75)
        iqr = q3 - q1
        outliers = revenue[(revenue > q3 + 1.5 * iqr) | (revenue < q1 - 1.5 * iqr)]
        assert len(outliers) > 0

    def test_has_missing_values_for_reporting_goals(self, df: pd.DataFrame) -> None:
        assert df.isnull().sum().sum() > 0

    def test_missingness_stays_below_the_drop_threshold(self, df: pd.DataFrame) -> None:
        """Above 40% the ingestion layer warns and the column stops being useful."""
        assert (df.isnull().mean() < 0.4).all()

    def test_has_an_imbalanced_label_for_classification_goals(self, df: pd.DataFrame) -> None:
        counts = df["churned"].value_counts()
        assert len(counts) == 2
        assert counts.max() / counts.min() > 1.2

    def test_has_a_strong_linear_driver_for_regression_goals(self, df: pd.DataFrame) -> None:
        subset = df[["units", "unit_price", "revenue"]].dropna()
        assert subset["revenue"].corr(subset["units"]) > 0.5

    def test_has_separable_segments_for_clustering_goals(self, df: pd.DataFrame) -> None:
        """Age is bimodal by construction (two customer segments)."""
        young = df[df["customer_age"] < 38]
        old = df[df["customer_age"] >= 38]
        assert len(young) > 20 and len(old) > 20
        assert old["unit_price"].mean() > young["unit_price"].mean() * 1.5
