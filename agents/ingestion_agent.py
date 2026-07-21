"""
agents/ingestion_agent.py
──────────────────────────
IngestionAgent — tabular data ingestion, validation, and profiling.

Accepts CSV, TSV, JSON, Parquet, and Excel (XLSX/XLS) sources, normalises
them into a canonical Pandas DataFrame via DataIngestionEngine, and profiles
the result into a structured IngestionResult (row/column counts, per-column
stats, and data-quality warnings).
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import UploadFile

from backend.schemas.analysis import ColumnStats, ColumnSummary, IngestionResult
from data_pipeline.ingestion import DataIngestionEngine, IngestionError

logger = logging.getLogger(__name__)


class IngestionAgent:
    """
    Ingests tabular data, validates structural constraints, and profiles columns.

    Input context keys consumed (via run):
        - ``source`` : str, Path, or UploadFile
    """

    def __init__(self) -> None:
        self._engine = DataIngestionEngine()

    def run(
        self,
        source: str | Path | UploadFile | None = None,
        context: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """
        Execute ingestion, validation, and profiling on the provided source.

        Parameters
        ----------
        source : str | Path | UploadFile, optional
            The data source to ingest.
        context : dict, optional
            Shared pipeline context. If source is not provided directly, it is
            extracted from context["data"].get("source") or context.get("source").

        Returns
        -------
        IngestionResult
            Structured ingestion profile for the tabular dataset.

        Raises
        ------
        IngestionError
            If validation or ingestion fails critically (empty, duplicate headers, etc.).
        """
        context = context or {}

        target_source = source
        if target_source is None:
            target_source = context.get("source")
            if target_source is None:
                target_source = context.get("data", {}).get("source")

        if target_source is None:
            raise IngestionError("No data source provided to IngestionAgent.")

        logger.info("IngestionAgent running on source: %s", target_source)

        self._validate_raw_headers(target_source)

        df = self._engine.load(target_source)

        # Downstream agents (MiningAgent, VisualizationAgent) consume the
        # canonical DataFrame directly rather than re-parsing the source.
        context["dataframe"] = df

        row_count = len(df)
        column_count = len(df.columns)
        if row_count < 1:
            raise IngestionError("Invalid dataset: the loaded dataset has 0 rows.")
        if column_count < 1:
            raise IngestionError("Invalid dataset: the loaded dataset has 0 columns.")

        column_summary: list[ColumnSummary] = []
        warnings: list[str] = []

        for col in df.columns:
            col_series = df[col]
            dtype_str = str(col_series.dtype)
            missing_count = int(col_series.isnull().sum())
            missing_pct = (missing_count / row_count) * 100 if row_count > 0 else 0.0

            if missing_pct >= 40.0:
                warnings.append(
                    f"Column '{col}' has {missing_pct:.1f}% missing values ({missing_count}/{row_count})."
                )

            if pd.api.types.is_numeric_dtype(col_series) and not pd.api.types.is_bool_dtype(col_series):
                min_val = col_series.min()
                max_val = col_series.max()
                mean_val = col_series.mean()
                stats = ColumnStats(
                    min=float(min_val) if pd.notnull(min_val) else None,
                    max=float(max_val) if pd.notnull(max_val) else None,
                    mean=float(mean_val) if pd.notnull(mean_val) else None,
                )
            else:
                unique_count = int(col_series.nunique())
                stats = ColumnStats(unique_count=unique_count)

            column_summary.append(
                ColumnSummary(
                    name=str(col),
                    dtype=dtype_str,
                    missing_count=missing_count,
                    stats=stats,
                )
            )

        n_dupes = int(df.duplicated().sum())
        if n_dupes > 0:
            warnings.append(f"Found {n_dupes} duplicate row(s).")

        for col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) > 0 and non_null.nunique() == 1:
                warnings.append(f"Column '{col}' is constant (only one unique value).")

        return IngestionResult(
            row_count=row_count,
            column_count=column_count,
            column_summary=column_summary,
            warnings=warnings,
        )

    def _validate_raw_headers(self, source: str | Path | UploadFile) -> None:
        """Inspect raw headers before Pandas normalizes duplicate column names."""
        filename = ""
        content = b""

        if self._is_upload_file(source):
            filename = source.filename or ""
            try:
                source.file.seek(0)
                content = source.file.read()
                source.file.seek(0)
            except Exception as exc:
                raise IngestionError(f"Failed to read raw file header: {exc}") from exc
        else:
            file_path = Path(source)
            filename = file_path.name
            try:
                content = file_path.read_bytes()
            except Exception as exc:
                raise IngestionError(f"Failed to read file for header validation: {exc}") from exc

        if not content:
            return

        ext = Path(filename).suffix.lower()
        headers = []

        if ext in (".csv", ".tsv"):
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise IngestionError(f"Encoding check failed: {exc}") from exc

            lines = [line for line in text.splitlines() if line.strip()]
            if not lines:
                return
            first_line = lines[0]

            if ext == ".tsv":
                sep = "\t"
            else:
                try:
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(first_line, delimiters=[",", ";", "\t", "|"])
                    sep = dialect.delimiter
                except csv.Error:
                    sep = ","

            reader = csv.reader([first_line], delimiter=sep)
            headers = next(reader, [])

        elif ext in (".xlsx", ".xls"):
            try:
                import openpyxl

                wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True)
                sheet = wb.active
                if sheet:
                    for row in sheet.iter_rows(max_row=1, values_only=True):
                        headers = ["" if val is None else str(val).strip() for val in row]
                        break
            except Exception as exc:
                raise IngestionError(f"Failed to read Excel header: {exc}") from exc

        seen = set()
        duplicates = []
        for h in headers:
            if h in seen:
                duplicates.append(h)
            seen.add(h)

        if duplicates:
            raise IngestionError(
                f"Duplicate column names detected in header: {sorted(set(duplicates))}"
            )

    def _is_upload_file(self, source: str | Path | UploadFile) -> bool:
        """Return True for FastAPI/Starlette upload objects without relying on class identity."""
        return hasattr(source, "filename") and hasattr(source, "file")

    # TODO: Add image/text/multimodal ingestion once modality-specific loaders are ready.
    # TODO: Add Spark/Dask hooks for large datasets and out-of-core processing.
