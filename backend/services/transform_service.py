"""
services/transform_service.py
───────────────────────────────
Dataset manipulation: cleaning-op / cell-edit transforms (via
DataProcessingEngine) and a sandboxed SQL query console (via DuckDB).

Every function here is ownership-scoped the same way dataset_service.py's
are (ownership checked through get_dataset), and every transform produces
a *new* Dataset row (dataset_service.save_dataset with parent_id set) —
nothing here ever mutates a Dataset in place.

Transformed versions are always re-serialized as Parquet, not the
original upload format: cast_dtype results (real datetime64/category
dtypes) would silently degrade back to strings on a CSV round-trip,
drifting from what the user just set.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.dataset import Dataset
from backend.services import dataset_service
from data_pipeline.ingestion import DataIngestionEngine
from data_pipeline.processing import DataProcessingEngine

logger = logging.getLogger(__name__)

_ingestion_engine = DataIngestionEngine()
_processing_engine = DataProcessingEngine()

# Query console safety limits — see run_query() for why each exists.
QUERY_TIMEOUT_SECONDS = 10.0
QUERY_ROW_CAP = 5000


class TransformNotFoundError(Exception):
    """Raised when the target dataset doesn't exist or isn't owned by the
    requesting user — the router turns this into a 404."""


class QueryValidationError(Exception):
    """Raised for rejected SQL (not a single SELECT/WITH statement, or a
    result that exceeds the row cap) — the router turns this into a 400."""


async def load_dataframe(db: AsyncSession, user_id: str, dataset_id: str) -> pd.DataFrame:
    """Fetch a Dataset's bytes and parse them into a DataFrame, reusing the
    same format-detection DataIngestionEngine uses for uploads — so a
    transformed (Parquet) version round-trips through the same parser as
    an original CSV/JSON/etc. upload."""
    dataset = await dataset_service.get_dataset(db, user_id, dataset_id)
    if dataset is None:
        raise TransformNotFoundError(f"Dataset {dataset_id!r} not found.")
    stored = dataset_service.as_stored_file(dataset)
    return _ingestion_engine.load(stored)


async def apply_transform(
    db: AsyncSession, user_id: str, dataset_id: str, ops: list[dict[str, Any]]
) -> tuple[Dataset, list[str]]:
    """Load the parent version, apply `ops` in order, persist the result as
    exactly one new child version. Returns (new Dataset, report lines)."""
    dataset = await dataset_service.get_dataset(db, user_id, dataset_id)
    if dataset is None:
        raise TransformNotFoundError(f"Dataset {dataset_id!r} not found.")

    df = await load_dataframe(db, user_id, dataset_id)
    result = _processing_engine.process(df, {"ops": ops})
    transformed_df: pd.DataFrame = result["df"]
    report: list[str] = result["report"]

    content = _to_parquet_bytes(transformed_df)
    filename = _stem(dataset.filename) + ".parquet"

    new_dataset = await dataset_service.save_dataset(
        db,
        user_id,
        filename,
        content,
        row_count=len(transformed_df),
        column_count=len(transformed_df.columns),
        parent_id=dataset.id,
        transform_type="clean",
        transform_params={"ops": ops},
    )
    logger.info(
        "apply_transform | parent=%s -> new=%s (v%d) | %d op(s)",
        dataset.id, new_dataset.id, new_dataset.version, len(ops),
    )
    return new_dataset, report


async def run_query(db: AsyncSession, user_id: str, dataset_id: str, sql: str) -> pd.DataFrame:
    """Execute a read-only SQL query against a dataset's DataFrame via a
    sandboxed, in-memory DuckDB connection. Does not touch the datasets
    table — callers decide whether to persist the result (save_query_result)."""
    _validate_single_select(sql)
    df = await load_dataframe(db, user_id, dataset_id)
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_execute_duckdb, df, sql), timeout=QUERY_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError as exc:
        raise QueryValidationError(
            f"Query exceeded the {QUERY_TIMEOUT_SECONDS:.0f}s time limit."
        ) from exc
    if len(result) > QUERY_ROW_CAP:
        raise QueryValidationError(
            f"Query result has {len(result)} rows, exceeding the {QUERY_ROW_CAP}-row cap. "
            "Add a LIMIT or narrow the query."
        )
    return result


async def save_query_result(
    db: AsyncSession, user_id: str, dataset_id: str, sql: str
) -> tuple[Dataset, int]:
    """Re-run the query server-side (never trusts a client-cached preview,
    so transform_params.sql stays genuinely reproducible) and persist the
    result as a new version. Returns (new Dataset, row_count)."""
    dataset = await dataset_service.get_dataset(db, user_id, dataset_id)
    if dataset is None:
        raise TransformNotFoundError(f"Dataset {dataset_id!r} not found.")

    result_df = await run_query(db, user_id, dataset_id, sql)
    content = _to_parquet_bytes(result_df)
    filename = _stem(dataset.filename) + ".parquet"

    new_dataset = await dataset_service.save_dataset(
        db,
        user_id,
        filename,
        content,
        row_count=len(result_df),
        column_count=len(result_df.columns),
        parent_id=dataset.id,
        transform_type="query_save",
        transform_params={"sql": sql},
    )
    logger.info(
        "save_query_result | parent=%s -> new=%s (v%d)",
        dataset.id, new_dataset.id, new_dataset.version,
    )
    return new_dataset, len(result_df)


def build_preview(df: pd.DataFrame, offset: int = 0, limit: int = 50) -> dict[str, Any]:
    """A JSON-safe page of rows — backs both the spreadsheet grid and the
    query console's result table. Handles NaN/NaT/numpy scalar types,
    none of which are directly JSON-serializable."""
    total_rows = len(df)
    page = df.iloc[offset : offset + limit]
    columns = [str(c) for c in page.columns]
    dtypes = [str(dt) for dt in page.dtypes]
    rows = [[_json_safe(v) for v in row] for row in page.itertuples(index=False, name=None)]
    return {
        "columns": columns,
        "dtypes": dtypes,
        "rows": rows,
        "total_rows": total_rows,
        "offset": offset,
        "limit": limit,
    }


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        as_float = float(value)
        return None if pd.isna(as_float) else as_float
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


# ── Internal helpers ─────────────────────────────────────────────────────


def _stem(filename: str) -> str:
    return filename.rsplit(".", 1)[0] if "." in filename else filename


def _to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _validate_single_select(sql: str) -> None:
    """Cheap first filter against stacked statements — the real backstop
    is enable_external_access=False in _execute_duckdb, which blocks
    filesystem/network access regardless of what SQL gets through here."""
    stripped = sql.strip()
    if not stripped:
        raise QueryValidationError("Query cannot be empty.")
    upper = stripped.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise QueryValidationError("Only a single SELECT (or WITH ... SELECT) statement is allowed.")
    # A ';' followed by more non-whitespace content means stacked statements.
    body = stripped.rstrip(";").strip()
    if ";" in body:
        raise QueryValidationError("Multiple statements are not allowed — submit one query at a time.")


def _execute_duckdb(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    """Runs in a worker thread (see run_query). Opens a fresh sandboxed
    connection per call — cheap for an in-memory DataFrame, and avoids any
    state leaking between queries."""
    import duckdb

    con = duckdb.connect(":memory:", config={"enable_external_access": False})
    try:
        con.register("df", df)
        return con.sql(sql).df()
    except duckdb.Error as exc:
        raise QueryValidationError(f"Query failed: {exc}") from exc
    finally:
        con.close()
