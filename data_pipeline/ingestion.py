"""
data_pipeline/ingestion.py
───────────────────────────
DataIngestionEngine — multi-source data loading and normalisation.

Responsibilities:
    • Load data from heterogeneous sources:
        - Local files   : CSV, TSV, JSON, Parquet, Excel
        - Remote URLs   : HTTP/S files, S3 objects            (future)
        - Databases     : SQLite, PostgreSQL, MySQL           (future)
        - Streaming     : Kafka topics, event queues          (future)
    • Normalise all sources into a canonical Pandas DataFrame.
    • Provide a standard result shape so upstream agents always receive
      the same keys regardless of source type.
    • Expose hooks for Spark and Dask loaders for large-scale data.

Wiring (M1):
    - IngestionAgent calls DataIngestionEngine.load().
    - Supports local CSV / TSV / JSON / Parquet / Excel files.
    - Parse failures are surfaced as ValueError with a readable message
      instead of leaking a raw pandas traceback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Map file extension -> (pandas reader name, default kwargs for that format).
# Adding a new tabular format is a one-line change here.
_READERS: dict[str, tuple[str, dict[str, Any]]] = {
    ".csv": ("read_csv", {}),
    ".tsv": ("read_csv", {"sep": "\t"}),
    ".json": ("read_json", {}),
    ".parquet": ("read_parquet", {}),
    ".xlsx": ("read_excel", {}),
    ".xls": ("read_excel", {}),
}


class DataIngestionEngine:
    """
    Multi-source data loader that normalises all inputs to Pandas DataFrames.

    Usage::

        engine = DataIngestionEngine()
        result = engine.load("data/samples/sales.csv")
        df = result["df"]
    """

    #: Formats this engine can currently load.
    SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(sorted(_READERS))

    def load(self, source: str | Path, **kwargs: Any) -> dict[str, Any]:
        """
        Load data from a source and return a normalised result.

        Parameters
        ----------
        source : str | Path
            Path to a .csv, .tsv, .json, .parquet, .xlsx, or .xls file.
        **kwargs
            Extra options forwarded to the underlying pandas reader
            (e.g. ``sep``, ``encoding``, ``orient``). These override the
            per-format defaults.

        Returns
        -------
        dict
            Keys:
                ``df``       : loaded Pandas DataFrame
                ``source``   : echo of the original source identifier
                ``format``   : detected file extension (e.g. ".csv")
                ``row_count``: number of rows
                ``columns``  : list of column names
                ``dtypes``   : {column: dtype_string} mapping
                ``message``  : human-readable status line

        Raises
        ------
        FileNotFoundError
            If the source file does not exist.
        ValueError
            If the file type is unsupported, a required engine (e.g.
            openpyxl for .xlsx) is missing, or the file cannot be parsed.
        """
        path = Path(source)
        logger.info("DataIngestionEngine.load() | source=%s", path)

        if not path.exists():
            raise FileNotFoundError(f"Data source not found: {path}")

        suffix = path.suffix.lower()
        entry = _READERS.get(suffix)
        if entry is None:
            supported = ", ".join(self.SUPPORTED_EXTENSIONS)
            raise ValueError(
                f"Unsupported file type '{suffix}'. Supported types: {supported}."
            )

        reader_name, default_kwargs = entry
        reader = getattr(pd, reader_name)
        read_kwargs = {**default_kwargs, **kwargs}

        try:
            df: pd.DataFrame = reader(path, **read_kwargs)
        except ImportError as exc:
            # e.g. reading .xlsx without openpyxl installed.
            raise ValueError(
                f"Cannot read '{path.name}': a required library is missing "
                f"({exc}). Install the appropriate extra (e.g. 'openpyxl' for Excel)."
            ) from exc
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
            raise ValueError(f"Failed to parse '{path.name}': {exc}") from exc

        return {
            "df": df,
            "source": str(path),
            "format": suffix,
            "row_count": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "message": f"Loaded {len(df)} rows from {path.name}.",
        }
