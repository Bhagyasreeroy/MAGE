"""
agents/ingestion_agent.py
──────────────────────────
IngestionAgent — data ingestion, validation, and quality profiling.

Responsibilities:
    • Accept heterogeneous data sources (CSV, TSV, JSON, Parquet, Excel).
    • Validate and normalise the raw payload into a canonical internal
      representation (a Pandas DataFrame + metadata).
    • Report ingestion statistics: row count, column types, and a set of
      data-quality warnings (missing values, empty/constant columns,
      duplicate rows, empty datasets).
    • Surface those warnings to the Orchestrator for downstream agents.

Wiring (M1):
    - Calls DataIngestionEngine from /data_pipeline/ingestion.py.
    - Reads the file path from context["data"]["path"] (or "source").
    - Emits schema, row_count, columns, and quality_warnings.
    - Never raises on bad input: ingestion errors are returned as warnings
      so a single bad upload cannot crash the whole analysis run.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from data_pipeline.ingestion import DataIngestionEngine

logger = logging.getLogger(__name__)


def _profile_quality(df: pd.DataFrame) -> list[str]:
    """
    Inspect a DataFrame and return a list of human-readable quality warnings.

    Checks: empty dataset, missing values (per column), fully-empty columns,
    duplicate rows, and constant (single-value) columns.
    """
    warnings: list[str] = []
    n_rows = len(df)

    if n_rows == 0:
        warnings.append("Dataset is empty (0 rows).")
        return warnings

    # Missing values, distinguishing "some missing" from "entirely empty".
    missing_counts = df.isna().sum()
    for column, n_missing in missing_counts.items():
        n_missing = int(n_missing)
        if n_missing == 0:
            continue
        if n_missing == n_rows:
            warnings.append(f"Column '{column}' is entirely empty ({n_rows} rows).")
        else:
            pct = n_missing / n_rows * 100
            warnings.append(
                f"Column '{column}' has {n_missing} missing value(s) ({pct:.1f}%)."
            )

    # Duplicate rows.
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        warnings.append(f"Found {n_dupes} duplicate row(s).")

    # Constant columns (a single unique non-null value carries no signal).
    for column in df.columns:
        non_null = df[column].dropna()
        if len(non_null) > 0 and non_null.nunique() == 1:
            warnings.append(f"Column '{column}' is constant (only one unique value).")

    return warnings


class IngestionAgent:
    """
    Parses, validates, and profiles heterogeneous tabular data uploads.

    Input context keys consumed:
        - ``data``            : dict; expects ``path`` (or ``source``) pointing
                                to a local CSV / TSV / JSON / Parquet / Excel file
        - ``goal``            : analytical goal (logged for traceability)
        - ``expertise_level`` : reserved for future verbosity adaptation

    Output keys produced:
        - ``schema``          : inferred column -> dtype mapping
        - ``row_count``       : number of rows ingested
        - ``columns``         : list of column names
        - ``format``          : detected file format (e.g. ".csv"), if loaded
        - ``quality_warnings``: list of data-quality issue strings
        - ``message``         : human-readable status line
    """

    def __init__(self) -> None:
        self._engine = DataIngestionEngine()

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute the ingestion and validation pipeline.

        Parameters
        ----------
        context : dict, optional
            Shared pipeline context forwarded by the Orchestrator. The data
            payload is read from ``context["data"]``.

        Returns
        -------
        dict
            Ingestion result payload (see class docstring). Always returns a
            valid dict — failures are reported via ``quality_warnings``.
        """
        context = context or {}
        data = context.get("data") or {}
        source = data.get("path") or data.get("source")

        logger.info(
            "IngestionAgent.run() | goal=%r source=%r",
            context.get("goal"),
            source,
        )

        # No data provided — return an empty-but-valid result rather than raising,
        # so the orchestrator's skeleton smoke run still succeeds.
        if not source:
            return {
                "schema": {},
                "row_count": 0,
                "columns": [],
                "quality_warnings": [],
                "message": "No data source provided in context['data']; nothing ingested.",
            }

        try:
            result = self._engine.load(source)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Ingestion failed for source=%r: %s", source, exc)
            return {
                "schema": {},
                "row_count": 0,
                "columns": [],
                "quality_warnings": [f"Ingestion error: {exc}"],
                "message": f"Failed to ingest {source}: {exc}",
            }

        return {
            "schema": result["dtypes"],
            "row_count": result["row_count"],
            "columns": result["columns"],
            "format": result["format"],
            "quality_warnings": _profile_quality(result["df"]),
            "message": result["message"],
        }
