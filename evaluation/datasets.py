"""
evaluation/datasets.py
───────────────────────
The fixed dataset suite the evaluation harness runs against.

Two sources, deliberately mixed:

  • **scikit-learn built-ins** (iris, wine, breast_cancer, diabetes) — real,
    widely recognised data that ships with a dependency the project already
    has. No network access, no download step, byte-identical on every machine.
  • **One seeded synthetic set** (``customer_orders``) carrying *planted*
    structure — two separable segments, injected outliers, an imbalanced
    label, and deliberate missingness. Without it, clustering and anomaly
    goals would be scored on data that has nothing for them to find, which
    would understate MAGE and flatter the baseline.

Every dataset is deterministic and pinned. Datasets are materialised to CSV
under ``evaluation/_data/`` because the pipeline under test ingests *sources*,
not DataFrames — routing through a real file exercises IngestionAgent for
real rather than bypassing it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Materialised CSVs live here. Regenerated deterministically on every run.
DATA_DIR = Path(__file__).resolve().parent / "_data"

# Seed for the synthetic generator. Pinned — changing it changes the results.
SYNTHETIC_SEED = 42
SYNTHETIC_ROWS = 400


@dataclass(frozen=True)
class EvalDataset:
    """
    One dataset in the evaluation suite.

    Attributes
    ----------
    name : str
        Stable identifier used as the CSV filename and as a results key.
    description : str
        One-line provenance note, reproduced in the generated report.
    build : Callable[[], pd.DataFrame]
        Deterministic builder. Called once per harness run.
    target_column : str | None
        The natural supervised target, where one exists. Used only to phrase
        goals naturally — the pipeline still detects the target itself.
    notes : dict
        Free-form provenance metadata carried through into results.json.
    """

    name: str
    description: str
    build: Callable[[], pd.DataFrame]
    target_column: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def materialize(self, data_dir: Path = DATA_DIR) -> Path:
        """Write the dataset to CSV and return the path the harness ingests."""
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / f"{self.name}.csv"
        df = self.build()
        df.to_csv(path, index=False)
        logger.info(
            "Materialized %s → %s (%d rows × %d cols)", self.name, path, len(df), len(df.columns)
        )
        return path


# ── scikit-learn built-ins ───────────────────────────────────────────────────


def _sklearn_frame(loader_name: str, target_name: str, as_labels: bool) -> pd.DataFrame:
    """
    Load a bundled scikit-learn dataset into a flat DataFrame.

    Imported lazily so that merely importing this module does not pull in
    scikit-learn's dataset machinery.

    Parameters
    ----------
    loader_name : str
        Attribute name on ``sklearn.datasets`` (e.g. ``"load_iris"``).
    target_name : str
        Column name to give the target.
    as_labels : bool
        True for classification sets — maps the integer target onto its
        human-readable class names, so class-balance output is legible and the
        column is genuinely categorical rather than a bare integer code.
    """
    from sklearn import datasets as skds

    bunch = getattr(skds, loader_name)()
    df = pd.DataFrame(bunch.data, columns=[str(c) for c in bunch.feature_names])

    target = bunch.target
    if as_labels and getattr(bunch, "target_names", None) is not None:
        names = list(bunch.target_names)
        df[target_name] = [str(names[int(t)]) for t in target]
    else:
        df[target_name] = target

    # Feature names in some bundled sets contain spaces and parentheses
    # (e.g. "sepal length (cm)"). The ingestion layer normalizes headers
    # anyway; doing it here keeps the on-disk CSV readable in the report.
    df.columns = [str(c).replace(" (cm)", "_cm").replace(" ", "_") for c in df.columns]
    return df


# ── Seeded synthetic set ─────────────────────────────────────────────────────


def _build_customer_orders() -> pd.DataFrame:
    """
    Generate the synthetic commercial dataset with planted structure.

    Mirrors ``scripts/generate_sample_dataset.py`` (the demo dataset) but at
    evaluation scale. Every task type has something real to discover:

      • two separable customer segments   → clustering (KMeans / DBSCAN)
      • revenue ≈ units × unit_price      → correlation / linearity
      • injected extreme revenues + units → IQR / Isolation Forest
      • imbalanced ``churned`` label      → class balance
      • deliberate NaNs                   → missingness / data quality
    """
    rng = np.random.default_rng(SYNTHETIC_SEED)
    n = SYNTHETIC_ROWS

    segment = rng.integers(0, 2, size=n)

    age = np.where(segment == 0, rng.normal(28, 4, n), rng.normal(48, 6, n)).round().astype(int)
    units = np.where(segment == 0, rng.normal(3, 1.0, n), rng.normal(9, 2.0, n)).round().clip(1)
    unit_price = np.where(
        segment == 0, rng.normal(15, 3, n), rng.normal(35, 6, n)
    ).round(2).clip(1)

    # Strong linear driver plus mild noise, so regression/correlation goals
    # surface genuine signal rather than noise.
    revenue = (units * unit_price + rng.normal(0, 8, n)).round(2)

    region = rng.choice(["North", "South", "East", "West"], size=n)
    product = rng.choice(["Widget", "Gadget", "Gizmo", "Doohickey"], size=n)

    # Mildly imbalanced label, loosely tied to segment.
    churn_p = np.where(segment == 0, 0.45, 0.20)
    churned = (rng.random(n) < churn_p).astype(int)

    df = pd.DataFrame(
        {
            "order_id": range(1001, 1001 + n),
            "region": region,
            "product": product,
            "customer_age": age,
            "units": units.astype(int),
            "unit_price": unit_price,
            "revenue": revenue,
            "churned": churned,
        }
    )

    # Planted outliers — deterministic positions so results are reproducible.
    outlier_rows = rng.choice(n, size=8, replace=False)
    df.loc[outlier_rows, "revenue"] = (
        df.loc[outlier_rows, "revenue"].astype(float) * 6.0
    ).round(2)
    df.loc[outlier_rows[:3], "units"] = 60

    # Planted missingness, well under the 40% drop threshold.
    missing_price = rng.choice(n, size=int(n * 0.03), replace=False)
    missing_revenue = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[missing_price, "unit_price"] = np.nan
    df.loc[missing_revenue, "revenue"] = np.nan

    return df


# ── The suite ────────────────────────────────────────────────────────────────


def load_eval_datasets() -> list[EvalDataset]:
    """
    Return the pinned evaluation suite.

    Order is stable so results.json diffs cleanly between runs.
    """
    return [
        EvalDataset(
            name="customer_orders",
            description=(
                "Seeded synthetic commercial dataset with planted segments, "
                "outliers, class imbalance, and missingness."
            ),
            build=_build_customer_orders,
            target_column="churned",
            notes={"source": "synthetic", "seed": SYNTHETIC_SEED, "rows": SYNTHETIC_ROWS},
        ),
        EvalDataset(
            name="iris",
            description="Fisher's iris — 3-class flower measurements (scikit-learn bundled).",
            build=lambda: _sklearn_frame("load_iris", "species", as_labels=True),
            target_column="species",
            notes={"source": "sklearn.datasets.load_iris"},
        ),
        EvalDataset(
            name="wine",
            description="Wine cultivar chemistry — 3-class, 13 features (scikit-learn bundled).",
            build=lambda: _sklearn_frame("load_wine", "cultivar", as_labels=True),
            target_column="cultivar",
            notes={"source": "sklearn.datasets.load_wine"},
        ),
        EvalDataset(
            name="breast_cancer",
            description="Breast cancer Wisconsin — binary diagnostic (scikit-learn bundled).",
            build=lambda: _sklearn_frame("load_breast_cancer", "diagnosis", as_labels=True),
            target_column="diagnosis",
            notes={"source": "sklearn.datasets.load_breast_cancer"},
        ),
        EvalDataset(
            name="diabetes",
            description=(
                "Diabetes progression — continuous target, the suite's genuine "
                "regression case (scikit-learn bundled)."
            ),
            build=lambda: _sklearn_frame("load_diabetes", "progression", as_labels=False),
            target_column="progression",
            notes={"source": "sklearn.datasets.load_diabetes"},
        ),
    ]
