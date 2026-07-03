"""
agents/ingestion_agent.py
──────────────────────────
IngestionAgent — multimodal data ingestion and validation.

Responsibilities:
    • Accept heterogeneous data sources (CSV, JSON, Parquet — more later).
    • Validate schema, detect file type, and normalise the raw payload into
      a canonical internal representation (a Pandas DataFrame + metadata).
    • Report ingestion statistics: row count, column types, missing-value
      rates, and any parse errors.
    • Surface data quality warnings to the Orchestrator for downstream
      agents to act on.

Wiring (M1):
    - Calls DataIngestionEngine from /data_pipeline/ingestion.py.
    - Reads the file path from context["data"]["path"] (or "source").
    - Emits schema, row_count, and quality_warnings.
"""

from __future__ import annotations

import logging
from typing import Any

from data_pipeline.ingestion import DataIngestionEngine

logger = logging.getLogger(__name__)


class IngestionAgent:
    """
    Parses, validates, and normalises heterogeneous data uploads.

    Input context keys consumed:
        - ``data``            : dict; expects ``path`` (or ``source``) pointing
                                to a local CSV / JSON / Parquet file
        - ``goal``            : analytical goal (used to infer required schema)
        - ``expertise_level`` : adapts verbosity of quality warnings

    Output keys produced:
        - ``schema``          : inferred column → dtype mapping
        - ``row_count``       : number of rows ingested
        - ``columns``         : list of column names
        - ``quality_warnings``: list of data-quality issue strings
    """

    def __init__(self) -> None:
        self._engine = DataIngestionEngine()

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute the ingestion and validation pipeline.

        Parameters
        ----------
        context : dict, optional
            Shared pipeline context forwarded by the Orchestrator.
            The data payload is read from ``context["data"]``.

        Returns
        -------
        dict
            Ingestion result payload.
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

        df = result["df"]

        # ── Data-quality checks: flag columns with missing values ──────────────
        quality_warnings: list[str] = []
        missing_counts = df.isna().sum()
        for column, n_missing in missing_counts.items():
            if n_missing > 0:
                pct = n_missing / len(df) * 100 if len(df) else 0.0
                quality_warnings.append(
                    f"Column '{column}' has {int(n_missing)} missing value(s) "
                    f"({pct:.1f}%)."
                )

        return {
            "schema": result["dtypes"],
            "row_count": result["row_count"],
            "columns": result["columns"],
            "quality_warnings": quality_warnings,
            "message": result["message"],
        }
