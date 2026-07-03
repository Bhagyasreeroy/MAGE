"""
data_pipeline/ingestion.py
───────────────────────────
DataIngestionEngine — multi-source data loading and normalisation.

Responsibilities:
    • Load data from heterogeneous sources:
        - Local files   : CSV, JSON, Parquet, Excel, Feather
        - Remote URLs   : HTTP/S files, S3 objects
        - Databases     : SQLite, PostgreSQL, MySQL (via SQLAlchemy)
        - Streaming     : Kafka topics, event queues (future)
    • Normalise all sources into a canonical Pandas DataFrame.
    • Provide a standard DataSource protocol / dataclass so upstream
      agents always receive the same shape regardless of source type.
    • Expose hooks for Spark and Dask loaders for large-scale data.

Wiring (M1):
    - IngestionAgent calls DataIngestionEngine.load().
    - Currently supports local CSV / JSON / Parquet files.
    - For large files (> MAX_UPLOAD_SIZE_MB), automatically fall back to
      Dask read_csv / Spark read.csv via the processing engine (future).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Map file extension → the pandas reader used to load it.
_READERS: dict[str, str] = {
    ".csv": "read_csv",
    ".json": "read_json",
    ".parquet": "read_parquet",
}


class DataIngestionEngine:
    """
    Multi-source data loader that normalises all inputs to Pandas DataFrames.

    Usage::

        engine = DataIngestionEngine()
        result = engine.load("data/samples/sales.csv")
        df = result["df"]
    """

    def load(self, source: str | Path, **kwargs: Any) -> dict[str, Any]:
        """
        Load data from a source and return a normalised result.

        Parameters
        ----------
        source : str | Path
            File path to a .csv, .json, or .parquet file.
        **kwargs
            Additional options forwarded to the underlying pandas reader
            (e.g. ``sep``, ``orient``).

        Returns
        -------
        dict
            Keys:
                ``df``       : loaded Pandas DataFrame
                ``source``   : echo of the original source identifier
                ``row_count``: number of rows
                ``columns``  : list of column names
                ``dtypes``   : {column: dtype_string} mapping

        Raises
        ------
        FileNotFoundError
            If the source file does not exist.
        ValueError
            If the file extension is not supported.
        """
        path = Path(source)
        logger.info("DataIngestionEngine.load() | source=%s", path)

        if not path.exists():
            raise FileNotFoundError(f"Data source not found: {path}")

        suffix = path.suffix.lower()
        reader_name = _READERS.get(suffix)
        if reader_name is None:
            supported = ", ".join(sorted(_READERS))
            raise ValueError(
                f"Unsupported file type '{suffix}'. Supported types: {supported}."
            )

        reader = getattr(pd, reader_name)
        df: pd.DataFrame = reader(path, **kwargs)

        return {
            "df": df,
            "source": str(path),
            "row_count": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "message": f"Loaded {len(df)} rows from {path.name}.",
        }
