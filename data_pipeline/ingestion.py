"""
data_pipeline/ingestion.py
───────────────────────────
DataIngestionEngine — multi-source data loading and normalisation.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Custom exception raised for all non-recoverable data ingestion errors."""


class DataIngestionEngine:
    """
    Multi-source data loader that normalises CSV, TSV, JSON, Parquet, and
    Excel (XLSX/XLS) files to Pandas DataFrames.
    """

    #: Formats this engine can currently load.
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (".csv", ".tsv", ".json", ".parquet", ".xlsx", ".xls")

    def load(self, source: str | Path | UploadFile, **kwargs: Any) -> pd.DataFrame:
        """
        Load tabular data from a supported source and return a Pandas DataFrame.

        Parameters
        ----------
        source : str | Path | UploadFile
            The file source to load. Can be a file path string, a Path object,
            or a FastAPI UploadFile.
        **kwargs
            Additional arguments passed to the underlying Pandas loader.

        Returns
        -------
        pd.DataFrame
            The parsed and loaded dataset as a Pandas DataFrame.

        Raises
        ------
        IngestionError
            If the file is not found, empty, has an unsupported extension,
            fails encoding checks, or fails delimiter/parse detection.
        """
        filename = ""
        content = b""

        if self._is_upload_file(source):
            filename = source.filename or ""
            try:
                source.file.seek(0)
                content = source.file.read()
                source.file.seek(0)
            except Exception as exc:
                raise IngestionError(f"Failed to read uploaded file content: {exc}") from exc
        else:
            file_path = Path(source)
            filename = file_path.name
            if not file_path.exists():
                raise IngestionError(f"File not found: {file_path}")
            if not file_path.is_file():
                raise IngestionError(f"Path is not a file: {file_path}")
            try:
                content = file_path.read_bytes()
            except Exception as exc:
                raise IngestionError(f"Failed to read file from path: {exc}") from exc

        if not content or len(content.strip()) == 0:
            raise IngestionError("Empty file: the provided dataset has no content.")

        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(self.SUPPORTED_EXTENSIONS)
            raise IngestionError(
                f"Unsupported extension '{ext}': supported types are {supported}."
            )

        if ext == ".csv":
            return self._load_csv(content, **kwargs)
        if ext == ".tsv":
            return self._load_csv(content, sep="\t", **kwargs)
        if ext == ".json":
            return self._load_json(content, **kwargs)
        if ext == ".parquet":
            return self._load_parquet(content, **kwargs)
        if ext in (".xlsx", ".xls"):
            return self._load_xlsx(content, **kwargs)
        raise IngestionError(f"Unsupported file type: {ext}")

    def _load_csv(self, content: bytes, sep: str | None = None, **kwargs: Any) -> pd.DataFrame:
        """Decode, detect delimiter (unless forced), and parse CSV/TSV contents."""
        try:
            sample_text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestionError(f"Encoding error: failed to decode file as UTF-8: {exc}") from exc

        if not sample_text.strip():
            raise IngestionError("Empty file.")

        if sep is None:
            try:
                sample_lines = "\n".join(sample_text.splitlines()[:20])
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample_lines, delimiters=[",", ";", "\t", "|"])
                sep = dialect.delimiter
            except csv.Error as exc:
                raise IngestionError(f"Delimiter detection failure: {exc}") from exc

        try:
            return pd.read_csv(io.StringIO(sample_text), sep=sep, **kwargs)
        except pd.errors.EmptyDataError as exc:
            raise IngestionError("Empty file.") from exc
        except Exception as exc:
            raise IngestionError(f"Failed to parse delimited file: {exc}") from exc

    def _load_json(self, content: bytes, **kwargs: Any) -> pd.DataFrame:
        """Parse JSON contents into a DataFrame."""
        try:
            return pd.read_json(io.BytesIO(content), **kwargs)
        except ValueError as exc:
            raise IngestionError(f"Failed to parse JSON: {exc}") from exc
        except Exception as exc:
            raise IngestionError(f"Failed to parse JSON: {exc}") from exc

    def _load_parquet(self, content: bytes, **kwargs: Any) -> pd.DataFrame:
        """Parse Parquet contents into a DataFrame."""
        try:
            return pd.read_parquet(io.BytesIO(content), **kwargs)
        except Exception as exc:
            raise IngestionError(f"Failed to parse Parquet: {exc}") from exc

    def _load_xlsx(self, content: bytes, **kwargs: Any) -> pd.DataFrame:
        """Parse Excel (XLSX/XLS) contents."""
        try:
            return pd.read_excel(io.BytesIO(content), engine="openpyxl", **kwargs)
        except ImportError as exc:
            raise IngestionError(
                f"Cannot read Excel file: a required library is missing ({exc}). "
                "Install the 'openpyxl' extra."
            ) from exc
        except Exception as exc:
            raise IngestionError(f"Failed to parse Excel file: {exc}") from exc

    def _is_upload_file(self, source: str | Path | UploadFile) -> bool:
        """Return True for FastAPI/Starlette upload objects without relying on class identity."""
        return hasattr(source, "filename") and hasattr(source, "file")

    # TODO: Add image/text/multimodal ingestion once modality-specific loaders are ready.
    # TODO: Add Spark/Dask hooks for large datasets and out-of-core processing.
