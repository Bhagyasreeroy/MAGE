"""
tests/agents/test_clustering_performance.py
────────────────────────────────────────────
Performance contract for clustering (FR-05).

FR-05 requires an end-to-end analysis in under 60 seconds for datasets below
100k rows. The evaluation harness reported a comfortable 1.69s worst case —
but every dataset in that suite is between 150 and 569 rows, so the figure
said nothing about the scale the requirement actually names.

Driving the UI against a 40,000-row file exposed the gap: the run hung on the
Mining step. Measured, clustering alone took **74.5 seconds** at 40k rows,
breaching FR-05 well inside its stated bound.

The cost is `silhouette_score`, which is O(n²) in pairwise distances and was
being called once per candidate k (2 through 6) — five full quadratic passes
over the data. Isolating it: k-means plus silhouette took 76.6s at 40k rows
while DBSCAN over the same data took 0.8s.

These tests pin the fix so the cliff cannot come back silently. Written
test-first: the scaling test failed at roughly 75 seconds before the change.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.mining_agent import MiningAgent

CLUSTERING = ["standardize", "kmeans", "dbscan", "silhouette"]

# Budget for the mining step alone. FR-05 allows 60s end-to-end; holding the
# heaviest computation to a fraction of that leaves room for ingestion,
# visualization, retrieval, and persistence on the same request.
MINING_BUDGET_SECONDS = 20.0


def _frame(n: int) -> pd.DataFrame:
    """Two well-separated clusters, so silhouette has real structure to score."""
    rng = np.random.default_rng(42)
    seg = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "customer_age": np.where(seg == 0, rng.normal(28, 4, n), rng.normal(48, 6, n)),
            "units": np.where(seg == 0, rng.normal(3, 1, n), rng.normal(9, 2, n)),
            "unit_price": np.where(seg == 0, rng.normal(15, 3, n), rng.normal(35, 6, n)),
        }
    )


def _run(df: pd.DataFrame) -> tuple[dict, float]:
    agent = MiningAgent()
    started = time.perf_counter()
    result = agent.run(context={"dataframe": df, "directives": {"computations": CLUSTERING}})
    return result, time.perf_counter() - started


class TestClusteringScales:
    @pytest.mark.parametrize("rows", [20_000, 50_000])
    def test_stays_within_budget(self, rows: int) -> None:
        """The regression that broke FR-05: ~75s at 40k rows."""
        _, elapsed = _run(_frame(rows))
        assert elapsed < MINING_BUDGET_SECONDS, (
            f"clustering took {elapsed:.1f}s on {rows:,} rows — FR-05 allows 60s "
            f"end-to-end for <100k rows, and this is the mining step alone"
        )

    def test_cost_does_not_explode_with_size(self) -> None:
        """
        Guards the *shape* of the cost, not just one measurement.

        A quadratic silhouette made 4x the rows cost ~28x the time. Capping the
        sample makes the dominant term linear in the k-means fits, so a 4x
        increase should stay far below quadratic growth.
        """
        _, small = _run(_frame(10_000))
        _, large = _run(_frame(40_000))
        assert large < small * 12, (
            f"4x the rows cost {large / max(small, 1e-9):.1f}x the time "
            f"({small:.1f}s to {large:.1f}s) — that is the quadratic path again"
        )


@pytest.fixture(scope="module")
def result() -> dict:
    """One clustering run over 20k rows, shared by the correctness assertions."""
    computed, _ = _run(_frame(20_000))
    return computed


class TestClusteringStaysCorrect:
    """Speed must not be bought by breaking the result."""

    def test_a_clustering_result_is_still_produced(self, result: dict) -> None:
        assert result["clustering"] is not None

    def test_silhouette_score_is_still_reported(self, result: dict) -> None:
        score = result["clustering"]["silhouette_score"]
        assert -1.0 <= score <= 1.0

    def test_planted_structure_is_still_found(self, result: dict) -> None:
        """Two well-separated clusters should score clearly positive."""
        assert result["clustering"]["silhouette_score"] > 0.4

    def test_cluster_sizes_cover_every_row(self, result: dict) -> None:
        assert sum(result["clustering"]["cluster_sizes"]) == 20_000

    def test_k_is_within_the_searched_range(self, result: dict) -> None:
        from agents.mining_agent import MAX_CLUSTER_K

        assert 2 <= result["clustering"]["k"] <= MAX_CLUSTER_K

    def test_scatter_points_are_still_capped(self, result: dict) -> None:
        from agents.mining_agent import MAX_SCATTER_POINTS

        assert len(result["clustering"]["points"]) <= MAX_SCATTER_POINTS

    def test_result_is_deterministic(self) -> None:
        """Reproducibility is a headline claim; sampling must be seeded."""
        first, _ = _run(_frame(20_000))
        second, _ = _run(_frame(20_000))
        assert first["clustering"]["k"] == second["clustering"]["k"]
        assert (
            first["clustering"]["silhouette_score"]
            == second["clustering"]["silhouette_score"]
        )


class TestSmallDataIsUnaffected:
    """Below the sampling threshold the computation should be exactly as before."""

    def test_small_frames_still_cluster(self) -> None:
        result, _ = _run(_frame(200))
        assert result["clustering"] is not None
        assert result["clustering"]["silhouette_score"] > 0.4

    def test_sampling_does_not_distort_the_score(self) -> None:
        """A sampled estimate must track the exact value it approximates."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        from agents.mining_agent import SILHOUETTE_SAMPLE_SIZE

        df = _frame(12_000)
        scaled = StandardScaler().fit_transform(df)
        labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(scaled)

        exact = silhouette_score(scaled, labels)
        sampled = silhouette_score(
            scaled, labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=42
        )
        assert abs(exact - sampled) < 0.05, (
            f"sampled silhouette {sampled:.3f} strays from exact {exact:.3f}"
        )
