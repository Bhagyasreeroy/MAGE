"""Tests for data_pipeline/processing.py."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_pipeline.processing import DataProcessingEngine, ProcessingError


@pytest.fixture
def engine() -> DataProcessingEngine:
    return DataProcessingEngine()


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "region": ["East", "West", "East", None, "West"],
            "revenue": [100.0, np.nan, 300.0, 400.0, 400.0],
            "units": [3, 5, 2, 7, 7],
        }
    )


class TestProcessNoOps:
    def test_empty_ops_returns_unchanged_df(self, engine, df) -> None:
        result = engine.process(df, {"ops": []})
        assert result["report"] == []
        pd.testing.assert_frame_equal(result["df"], df)

    def test_no_config_returns_unchanged_df(self, engine, df) -> None:
        result = engine.process(df, None)
        pd.testing.assert_frame_equal(result["df"], df)


class TestDropColumns:
    def test_drops_named_columns(self, engine, df) -> None:
        result = engine.process(df, {"ops": [{"type": "drop_columns", "columns": ["units"]}]})
        assert "units" not in result["df"].columns
        assert "Dropped 1 column(s)" in result["report"][0]

    def test_unknown_column_raises(self, engine, df) -> None:
        with pytest.raises(ProcessingError, match="not found"):
            engine.process(df, {"ops": [{"type": "drop_columns", "columns": ["nope"]}]})

    def test_empty_list_raises(self, engine, df) -> None:
        with pytest.raises(ProcessingError):
            engine.process(df, {"ops": [{"type": "drop_columns", "columns": []}]})


class TestRenameColumns:
    def test_renames_column(self, engine, df) -> None:
        result = engine.process(df, {"ops": [{"type": "rename_columns", "mapping": {"units": "qty"}}]})
        assert "qty" in result["df"].columns
        assert "units" not in result["df"].columns

    def test_collision_raises(self, engine, df) -> None:
        with pytest.raises(ProcessingError, match="already exist"):
            engine.process(df, {"ops": [{"type": "rename_columns", "mapping": {"units": "revenue"}}]})


class TestFilterRows:
    def test_gt_filters_rows(self, engine, df) -> None:
        result = engine.process(df, {"ops": [{"type": "filter_rows", "column": "units", "op": "gt", "value": 4}]})
        assert len(result["df"]) == 3
        assert "5 → 3" in result["report"][0]

    def test_is_null_filters_missing(self, engine, df) -> None:
        result = engine.process(df, {"ops": [{"type": "filter_rows", "column": "region", "op": "is_null"}]})
        assert len(result["df"]) == 1

    def test_contains_string_match(self, engine, df) -> None:
        result = engine.process(df, {"ops": [{"type": "filter_rows", "column": "region", "op": "contains", "value": "East"}]})
        assert len(result["df"]) == 2

    def test_invalid_op_raises(self, engine, df) -> None:
        with pytest.raises(ProcessingError):
            engine.process(df, {"ops": [{"type": "filter_rows", "column": "units", "op": "bogus", "value": 1}]})


class TestFillMissing:
    def test_mean_strategy(self, engine, df) -> None:
        result = engine.process(df, {"ops": [{"type": "fill_missing", "column": "revenue", "strategy": "mean"}]})
        assert result["df"]["revenue"].isna().sum() == 0
        assert "Filled 1 missing value(s)" in result["report"][0]

    def test_constant_strategy_requires_value(self, engine, df) -> None:
        with pytest.raises(ProcessingError, match="requires a 'value'"):
            engine.process(df, {"ops": [{"type": "fill_missing", "column": "region", "strategy": "constant"}]})

    def test_constant_strategy_fills_value(self, engine, df) -> None:
        result = engine.process(
            df, {"ops": [{"type": "fill_missing", "column": "region", "strategy": "constant", "value": "Unknown"}]}
        )
        assert result["df"]["region"].isna().sum() == 0
        assert "Unknown" in result["df"]["region"].values

    def test_mode_strategy(self, engine, df) -> None:
        result = engine.process(df, {"ops": [{"type": "fill_missing", "column": "region", "strategy": "mode"}]})
        assert result["df"]["region"].isna().sum() == 0


class TestDropMissing:
    def test_drops_rows_with_any_missing(self, engine, df) -> None:
        result = engine.process(df, {"ops": [{"type": "drop_missing", "columns": None, "how": "any"}]})
        assert len(result["df"]) == 3  # rows with null region or revenue removed


class TestDedupe:
    def test_removes_duplicate_subset(self, engine, df) -> None:
        result = engine.process(df, {"ops": [{"type": "dedupe", "subset": ["units"]}]})
        assert len(result["df"]) == 4  # units=7 appears twice (rows 4,5)
        assert "Removed 1 duplicate row(s)" in result["report"][0]


class TestCastDtype:
    def test_cast_to_float(self, engine, df) -> None:
        result = engine.process(df, {"ops": [{"type": "cast_dtype", "column": "units", "dtype": "float64"}]})
        assert result["df"]["units"].dtype == np.float64

    def test_cast_to_int_with_nan_keeps_float(self, engine, df) -> None:
        result = engine.process(df, {"ops": [{"type": "cast_dtype", "column": "revenue", "dtype": "int64"}]})
        assert result["df"]["revenue"].dtype == np.float64
        assert "kept as float64" in result["report"][0]

    def test_cast_unparseable_string_column_reports_failures(self, engine) -> None:
        d = pd.DataFrame({"x": ["1", "2", "not_a_number"]})
        engine2 = DataProcessingEngine()
        result = engine2.process(d, {"ops": [{"type": "cast_dtype", "column": "x", "dtype": "float64"}]})
        assert "1 value(s) failed to convert" in result["report"][0]

    def test_invalid_dtype_raises(self, engine, df) -> None:
        with pytest.raises(ProcessingError):
            engine.process(df, {"ops": [{"type": "cast_dtype", "column": "units", "dtype": "not_a_type"}]})


class TestEditCells:
    def test_edits_named_cell(self, engine, df) -> None:
        result = engine.process(df, {"ops": [{"type": "edit_cells", "edits": [{"row_index": 0, "column": "region", "value": "North"}]}]})
        assert result["df"].iloc[0]["region"] == "North"

    def test_out_of_range_row_raises(self, engine, df) -> None:
        with pytest.raises(ProcessingError, match="out of range"):
            engine.process(df, {"ops": [{"type": "edit_cells", "edits": [{"row_index": 999, "column": "region", "value": "x"}]}]})

    def test_unknown_column_raises(self, engine, df) -> None:
        with pytest.raises(ProcessingError, match="not found"):
            engine.process(df, {"ops": [{"type": "edit_cells", "edits": [{"row_index": 0, "column": "nope", "value": "x"}]}]})

    def test_batches_multiple_edits_into_one_op(self, engine, df) -> None:
        result = engine.process(
            df,
            {
                "ops": [
                    {
                        "type": "edit_cells",
                        "edits": [
                            {"row_index": 0, "column": "region", "value": "North"},
                            {"row_index": 1, "column": "region", "value": "South"},
                        ],
                    }
                ]
            },
        )
        assert len(result["report"]) == 1
        assert result["df"].iloc[0]["region"] == "North"
        assert result["df"].iloc[1]["region"] == "South"


class TestChainedOps:
    def test_multiple_ops_applied_in_order_produce_one_result(self, engine, df) -> None:
        result = engine.process(
            df,
            {
                "ops": [
                    {"type": "drop_columns", "columns": ["id"]},
                    {"type": "fill_missing", "column": "revenue", "strategy": "mean"},
                    {"type": "dedupe", "subset": ["units"]},
                ]
            },
        )
        assert len(result["report"]) == 3
        assert "id" not in result["df"].columns
        assert result["df"]["revenue"].isna().sum() == 0


class TestUnknownOp:
    def test_unknown_op_type_raises(self, engine, df) -> None:
        with pytest.raises(ProcessingError, match="Unknown transform op type"):
            engine.process(df, {"ops": [{"type": "levitate"}]})
