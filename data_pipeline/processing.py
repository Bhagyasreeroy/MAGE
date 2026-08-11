"""
data_pipeline/processing.py
────────────────────────────
DataProcessingEngine — pandas transform layer for the dataset manipulation
platform (cleaning ops, batched spreadsheet cell edits).

process(df, config) applies every op in config["ops"] to df, in order, and
returns the resulting DataFrame plus a human-readable report line per op.
The caller (backend/services/transform_service.py) persists the final
result as exactly one new dataset version — never one version per op, and
never one per spreadsheet keystroke (cell edits are batched client-side
into a single "edit_cells" op before being sent here).

Supported op types: drop_columns, rename_columns, filter_rows,
fill_missing, drop_missing, dedupe, cast_dtype, edit_cells.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_FILTER_OPS = {"eq", "neq", "gt", "gte", "lt", "lte", "contains", "is_null", "not_null"}
_FILL_STRATEGIES = {"mean", "median", "mode", "constant", "ffill", "bfill"}
_CAST_DTYPES = {"int64", "float64", "string", "bool", "datetime64[ns]", "category"}


class ProcessingError(Exception):
    """Raised for invalid transform ops — unknown op type, missing column,
    bad params. Mirrors IngestionError; the router turns this into a 400."""


class DataProcessingEngine:
    """Applies an ordered list of transform ops to a DataFrame, collecting
    a human-readable report line per op (mirrors IngestionResult.warnings)."""

    def __init__(self, backend: str = "pandas") -> None:
        self.backend = backend
        logger.info("DataProcessingEngine created with backend=%s", backend)

    def process(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """config = {"ops": [{"type": "drop_columns", ...}, ...]}."""
        config = config or {}
        ops = config.get("ops") or []
        working = df.copy()
        report: list[str] = []
        for op in ops:
            working, message = self._apply_op(working, op)
            report.append(message)
        logger.info("DataProcessingEngine.process() | %d op(s) applied", len(ops))
        return {"df": working, "report": report, "backend": self.backend}

    # ── Dispatch ──────────────────────────────────────────────────────────

    def _apply_op(self, df: pd.DataFrame, op: dict[str, Any]) -> tuple[pd.DataFrame, str]:
        op_type = op.get("type")
        handler = self._HANDLERS.get(op_type)
        if handler is None:
            raise ProcessingError(f"Unknown transform op type: {op_type!r}")
        return handler(self, df, op)

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ProcessingError(f"Column(s) not found: {', '.join(missing)}")

    # ── Individual ops ────────────────────────────────────────────────────

    def _drop_columns(self, df: pd.DataFrame, op: dict[str, Any]) -> tuple[pd.DataFrame, str]:
        columns = op.get("columns") or []
        if not columns:
            raise ProcessingError("drop_columns requires a non-empty 'columns' list.")
        self._require_columns(df, columns)
        return df.drop(columns=columns), f"Dropped {len(columns)} column(s): {', '.join(columns)}."

    def _rename_columns(self, df: pd.DataFrame, op: dict[str, Any]) -> tuple[pd.DataFrame, str]:
        mapping = op.get("mapping") or {}
        if not mapping:
            raise ProcessingError("rename_columns requires a non-empty 'mapping'.")
        self._require_columns(df, list(mapping.keys()))
        collisions = [new for new in mapping.values() if new in df.columns and new not in mapping]
        if collisions:
            raise ProcessingError(f"Rename target(s) already exist: {', '.join(collisions)}")
        return df.rename(columns=mapping), f"Renamed {len(mapping)} column(s)."

    def _filter_rows(self, df: pd.DataFrame, op: dict[str, Any]) -> tuple[pd.DataFrame, str]:
        column = op.get("column")
        operator = op.get("op")
        value = op.get("value")
        if column is None or operator not in _FILTER_OPS:
            raise ProcessingError(
                f"filter_rows requires a valid 'column' and 'op' (one of {sorted(_FILTER_OPS)})."
            )
        self._require_columns(df, [column])
        series = df[column]
        if operator == "eq":
            mask = series == value
        elif operator == "neq":
            mask = series != value
        elif operator == "gt":
            mask = series > value
        elif operator == "gte":
            mask = series >= value
        elif operator == "lt":
            mask = series < value
        elif operator == "lte":
            mask = series <= value
        elif operator == "contains":
            mask = series.astype(str).str.contains(str(value), na=False)
        elif operator == "is_null":
            mask = series.isna()
        else:  # not_null
            mask = series.notna()
        before = len(df)
        result = df[mask]
        return result, f"Filtered '{column}' {operator} {value!r}: {before} → {len(result)} row(s)."

    def _fill_missing(self, df: pd.DataFrame, op: dict[str, Any]) -> tuple[pd.DataFrame, str]:
        column = op.get("column")
        strategy = op.get("strategy")
        if column is None or strategy not in _FILL_STRATEGIES:
            raise ProcessingError(
                f"fill_missing requires a valid 'column' and 'strategy' (one of {sorted(_FILL_STRATEGIES)})."
            )
        self._require_columns(df, [column])
        missing_before = int(df[column].isna().sum())
        result = df.copy()
        if strategy == "constant":
            if "value" not in op:
                raise ProcessingError("fill_missing with strategy='constant' requires a 'value'.")
            result[column] = result[column].fillna(op["value"])
            desc = f"constant ({op['value']!r})"
        elif strategy == "mean":
            result[column] = result[column].fillna(result[column].mean())
            desc = "mean"
        elif strategy == "median":
            result[column] = result[column].fillna(result[column].median())
            desc = "median"
        elif strategy == "mode":
            mode = result[column].mode()
            result[column] = result[column].fillna(mode.iloc[0] if not mode.empty else None)
            desc = "mode"
        elif strategy == "ffill":
            result[column] = result[column].ffill()
            desc = "forward-fill"
        else:  # bfill
            result[column] = result[column].bfill()
            desc = "backward-fill"
        return result, f"Filled {missing_before} missing value(s) in '{column}' with {desc}."

    def _drop_missing(self, df: pd.DataFrame, op: dict[str, Any]) -> tuple[pd.DataFrame, str]:
        columns = op.get("columns")
        how = op.get("how", "any")
        if how not in ("any", "all"):
            raise ProcessingError("drop_missing 'how' must be 'any' or 'all'.")
        if columns:
            self._require_columns(df, columns)
        before = len(df)
        result = df.dropna(subset=columns, how=how)
        return result, f"Dropped rows with missing values ({how}): {before} → {len(result)} row(s)."

    def _dedupe(self, df: pd.DataFrame, op: dict[str, Any]) -> tuple[pd.DataFrame, str]:
        subset = op.get("subset")
        keep = op.get("keep", "first")
        if keep not in ("first", "last"):
            raise ProcessingError("dedupe 'keep' must be 'first' or 'last'.")
        if subset:
            self._require_columns(df, subset)
        before = len(df)
        result = df.drop_duplicates(subset=subset, keep=keep)
        return result, f"Removed {before - len(result)} duplicate row(s)."

    def _cast_dtype(self, df: pd.DataFrame, op: dict[str, Any]) -> tuple[pd.DataFrame, str]:
        column = op.get("column")
        dtype = op.get("dtype")
        if column is None or dtype not in _CAST_DTYPES:
            raise ProcessingError(
                f"cast_dtype requires a valid 'column' and 'dtype' (one of {sorted(_CAST_DTYPES)})."
            )
        self._require_columns(df, [column])
        result = df.copy()
        note = ""
        try:
            if dtype in ("int64", "float64"):
                coerced = pd.to_numeric(result[column], errors="coerce")
                failed = int((coerced.isna() & result[column].notna()).sum())
                if dtype == "int64" and coerced.isna().any():
                    # int64 can't represent NaN — keep the coerced float64
                    # rather than let astype() raise.
                    result[column] = coerced
                    note = " (kept as float64 — column has missing/unconvertible values, which int64 can't represent)"
                else:
                    result[column] = coerced.astype(dtype)
            elif dtype == "datetime64[ns]":
                coerced = pd.to_datetime(result[column], errors="coerce")
                failed = int((coerced.isna() & result[column].notna()).sum())
                result[column] = coerced
            elif dtype == "bool":
                result[column] = result[column].astype(bool)
                failed = 0
            elif dtype == "string":
                result[column] = result[column].astype(str)
                failed = 0
            else:  # category
                result[column] = result[column].astype("category")
                failed = 0
        except (ValueError, TypeError) as exc:
            raise ProcessingError(f"Could not cast '{column}' to {dtype}: {exc}") from exc
        if failed:
            note = f" ({failed} value(s) failed to convert, left as null)" + note
        return result, f"Cast '{column}' to {dtype}.{note}"

    def _edit_cells(self, df: pd.DataFrame, op: dict[str, Any]) -> tuple[pd.DataFrame, str]:
        edits = op.get("edits") or []
        if not edits:
            raise ProcessingError("edit_cells requires a non-empty 'edits' list.")
        result = df.copy()
        for edit in edits:
            row_index = edit.get("row_index")
            column = edit.get("column")
            if row_index is None or column is None:
                raise ProcessingError("Each edit requires 'row_index' and 'column'.")
            if column not in result.columns:
                raise ProcessingError(f"Column not found: {column}")
            if not (0 <= row_index < len(result)):
                raise ProcessingError(f"Row index out of range: {row_index}")
            result.iat[row_index, result.columns.get_loc(column)] = edit.get("value")
        return result, f"Edited {len(edits)} cell(s)."

    _HANDLERS = {
        "drop_columns": _drop_columns,
        "rename_columns": _rename_columns,
        "filter_rows": _filter_rows,
        "fill_missing": _fill_missing,
        "drop_missing": _drop_missing,
        "dedupe": _dedupe,
        "cast_dtype": _cast_dtype,
        "edit_cells": _edit_cells,
    }
