"""
scripts/generate_sample_dataset.py
───────────────────────────────────
Generates a realistic demo dataset that exercises every goal-conditioned
computation MAGE supports — so a user can see the pipeline actually branch:

  • correlation / linearity  → units & unit_price drive revenue
  • clustering (KMeans/DBSCAN) → two natural customer segments
  • anomaly (IQR + Isolation Forest) → a handful of injected outliers
  • classification (class balance) → a `churned` target label
  • missingness → a few deliberately missing values

Deterministic (fixed seed) so the file is reproducible. Run:
    python scripts/generate_sample_dataset.py
Writes: data/samples/customer_orders.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 60

OUT = Path(__file__).resolve().parents[1] / "data" / "samples" / "customer_orders.csv"


def main() -> None:
    # Two customer segments → gives clustering something real to find.
    #   segment 0: younger, small orders, low spend
    #   segment 1: older, large orders, high spend
    segment = RNG.integers(0, 2, size=N)

    age = np.where(segment == 0, RNG.normal(28, 4, N), RNG.normal(48, 6, N)).round().astype(int)
    units = np.where(segment == 0, RNG.normal(3, 1.0, N), RNG.normal(9, 2.0, N)).round().clip(1)
    unit_price = np.where(segment == 0, RNG.normal(15, 3, N), RNG.normal(35, 6, N)).round(2).clip(1)

    # revenue is driven by units * unit_price (strong linearity/correlation)
    # plus mild noise — so regression/correlation goals surface real signal.
    revenue = (units * unit_price + RNG.normal(0, 8, N)).round(2)

    region = RNG.choice(["North", "South", "East", "West"], size=N)
    product = RNG.choice(["Widget", "Gadget", "Gizmo", "Doohickey"], size=N)

    # Churn label — mildly imbalanced, loosely tied to segment, for
    # classification/class-balance goals.
    churn_p = np.where(segment == 0, 0.45, 0.20)
    churned = (RNG.random(N) < churn_p).astype(int)

    df = pd.DataFrame(
        {
            "order_id": range(1001, 1001 + N),
            "region": region,
            "product": product,
            "customer_age": age,
            "units": units.astype(int),
            "unit_price": unit_price,
            "revenue": revenue,
            "churned": churned,
        }
    )

    # Inject a few clear outliers so IQR + Isolation Forest have something to flag.
    for idx, mult in [(5, 6.0), (23, 5.0), (47, 7.0)]:
        df.loc[idx, "revenue"] = round(float(df.loc[idx, "revenue"]) * mult, 2)
    df.loc[11, "units"] = 60  # anomalously large order

    # A few missing values so data-quality / missingness surfaces.
    df.loc[[3, 19], "unit_price"] = np.nan
    df.loc[[7, 31, 44], "revenue"] = np.nan

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns)} cols → {OUT}")


if __name__ == "__main__":
    main()
