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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FILTER_OPS = {"eq", "neq", "gt", "gte", "lt", "lte", "contains", "is_null", "not_null"}
_FILL_STRATEGIES = {"mean", "median", "mode", "constant", "ffill", "bfill"}
_CAST_DTYPES = {"int64", "float64", "string", "bool", "datetime64[ns]", "category"}

# What the spreadsheet grid may send for a boolean cell. Every cell arrives as
# a string — the grid is made of text inputs — so "False" has to be recognised
# as false rather than as the non-empty (and therefore truthy) string it is.
_TRUE_STRINGS = {"true", "t", "yes", "y", "1"}
_FALSE_STRINGS = {"false", "f", "no", "n", "0"}


class ProcessingError(Exception):
    """Raised for invalid transform ops — unknown op type, missing column,
    bad params. Mirrors IngestionError; the router turns this into a 400."""


def _is_blank(value: Any) -> bool:
    """
    True for a cleared cell.

    The grid renders a null as an empty text input, so an empty string is how
    "no value" comes back — whether the user emptied the cell or never touched
    a cell that was already null. Both mean the same thing here.
    """
    if value is None or value is pd.NaT:
        return True
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, float) and pd.isna(value)


def _coerce_cell_value(series: pd.Series, value: Any, column: str) -> tuple[Any, str | None]:
    """
    Convert one incoming cell value into something ``series``'s dtype can hold.

    Returns ``(value, widen_to)``, where ``widen_to`` is a dtype the column has
    to be converted to before the assignment will work, or None when it fits as
    it is. Widening is a last resort — it is only returned where the column
    genuinely cannot represent the value, never merely to make the assignment
    easier.

    Raises ProcessingError (→ 400) when the value is not something the column
    could hold under any reasonable reading. "abc" in a numeric column is a
    typo, and saying so beats storing it as text and quietly turning a numeric
    column into an object one.
    """
    dtype = series.dtype
    blank = _is_blank(value)

    # Before the numeric branch: pandas counts bool as numeric, and a bool
    # column reaching `pd.to_numeric` would accept "1"/"0" while rejecting the
    # "true"/"false" the grid actually sends.
    if pd.api.types.is_bool_dtype(dtype):
        if blank:
            # A bool column has no null to hold; object is the narrowest dtype
            # that can carry True, False and missing together.
            return None, "object"
        text = str(value).strip().lower()
        if text in _TRUE_STRINGS:
            return True, None
        if text in _FALSE_STRINGS:
            return False, None
        raise ProcessingError(
            f"'{value}' is not a true/false value, and '{column}' is a boolean column."
        )

    if pd.api.types.is_numeric_dtype(dtype):
        # int64 cannot hold a missing value either, but unlike bool, pandas
        # promotes it to float64 on assignment by itself — so no widening.
        if blank:
            return np.nan, None
        try:
            number = pd.to_numeric(str(value).strip())
        except (ValueError, TypeError) as exc:
            raise ProcessingError(
                f"'{value}' is not a number, and '{column}' is a numeric column."
            ) from exc
        # 9.5 into an integer column: keeping the value and widening the column
        # is right, because the alternative is silently storing 9.
        if pd.api.types.is_integer_dtype(dtype) and not float(number).is_integer():
            return number, "float64"
        return number, None

    if pd.api.types.is_datetime64_any_dtype(dtype):
        if blank:
            return pd.NaT, None
        try:
            return pd.Timestamp(value), None
        except (ValueError, TypeError) as exc:
            raise ProcessingError(
                f"'{value}' is not a date, and '{column}' is a datetime column."
            ) from exc

    # A category rejects any value outside its existing categories. Editing a
    # cell to something new is a legitimate thing to want, so the column stops
    # being categorical rather than the edit being refused.
    if isinstance(dtype, pd.CategoricalDtype):
        if blank:
            return None, None
        if value in dtype.categories:
            return value, None
        return str(value), "object"

    return (None if blank else str(value)), None


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
        """
        Apply individual cell edits from the spreadsheet grid.

        Every value arrives as a **string**, because the grid is made of text
        inputs. Writing one straight into a typed column is what this used to
        do, and pandas refuses it — ``TypeError: Invalid value '42' for dtype
        'int64'`` — which is not a ProcessingError, so it escaped the router's
        400 handler and surfaced as a 500. Editing any numeric cell failed;
        only text columns worked, which is exactly what the tests covered.

        So each value is converted to the column's own type first, and a value
        the column genuinely cannot represent is the user's mistake (a 400 with
        a message naming the column), not a server fault.
        """
        edits = op.get("edits") or []
        if not edits:
            raise ProcessingError("edit_cells requires a non-empty 'edits' list.")
        result = df.copy()
        widened: list[str] = []
        for edit in edits:
            row_index = edit.get("row_index")
            column = edit.get("column")
            if row_index is None or column is None:
                raise ProcessingError("Each edit requires 'row_index' and 'column'.")
            if column not in result.columns:
                raise ProcessingError(f"Column not found: {column}")
            if not (0 <= row_index < len(result)):
                raise ProcessingError(f"Row index out of range: {row_index}")

            value, widen_to = _coerce_cell_value(result[column], edit.get("value"), column)
            if widen_to is not None:
                # The column cannot hold the new value as it stands. Widening
                # it is the same accommodation `_cast_dtype` already makes when
                # int64 meets a missing value, and it is reported the same way
                # — silently changing a column's type is not something a user
                # should have to discover for themselves.
                result[column] = result[column].astype(widen_to)
                widened.append(f"'{column}' widened to {widen_to}")
            result.iat[row_index, result.columns.get_loc(column)] = value

        note = f" ({'; '.join(dict.fromkeys(widened))})" if widened else ""
        return result, f"Edited {len(edits)} cell(s).{note}"

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
