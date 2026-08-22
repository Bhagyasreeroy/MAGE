"""
tests/data_pipeline/test_cell_edit_types.py
────────────────────────────────────────────
Tests that a spreadsheet cell edit works on columns that are not text.

Every value from the grid arrives as a **string** — the grid is made of text
inputs. Writing one straight into a typed column is what `_edit_cells` used to
do, and pandas refuses it: ``TypeError: Invalid value '42' for dtype 'int64'``.
TypeError is not ProcessingError, so it slipped past the router's 400 handler
and reached the client as a 500. Editing any numeric cell failed.

The existing coverage in test_processing.py::TestEditCells edits ``region``, a
text column, in every case — which is why an entire broken path looked tested.
That is the thing to keep in mind here: each dtype the app can produce gets its
own case, because passing on one says nothing about the others.

The other half is *which* failure it is. A value the column cannot hold is the
user's typo, and has to arrive as a ProcessingError the router turns into a 400
naming the column — not as a server fault with nothing to act on.

Written test-first.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_pipeline.processing import DataProcessingEngine, ProcessingError


@pytest.fixture
def engine() -> DataProcessingEngine:
    return DataProcessingEngine()


@pytest.fixture
def df() -> pd.DataFrame:
    """One column per dtype the ingestion and cleaning paths can produce."""
    return pd.DataFrame(
        {
            "units": [1, 2, 3],
            "price": [1.5, 2.5, 3.5],
            "region": ["East", "West", "East"],
            "active": [True, False, True],
            "day": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "tier": pd.Categorical(["a", "b", "a"]),
        }
    )


def _edit(engine, df, column, value, row_index=0):
    return engine.process(
        df,
        {"ops": [{"type": "edit_cells", "edits": [{"row_index": row_index, "column": column, "value": value}]}]},
    )


class TestEveryColumnTypeAcceptsAnEdit:
    """The regression, one case per dtype."""

    def test_an_integer_cell(self, engine, df) -> None:
        result = _edit(engine, df, "units", "42")
        assert result["df"].iloc[0]["units"] == 42
        assert str(result["df"]["units"].dtype) == "int64"

    def test_a_float_cell(self, engine, df) -> None:
        result = _edit(engine, df, "price", "9.5")
        assert result["df"].iloc[0]["price"] == 9.5

    def test_a_text_cell(self, engine, df) -> None:
        assert _edit(engine, df, "region", "North")["df"].iloc[0]["region"] == "North"

    def test_a_boolean_cell(self, engine, df) -> None:
        """'false' is a non-empty string, so a truthiness check would store it
        as True — the one wrong answer that produces no error at all."""
        assert bool(_edit(engine, df, "active", "false")["df"].iloc[0]["active"]) is False
        assert bool(_edit(engine, df, "active", "True")["df"].iloc[0]["active"]) is True

    def test_a_datetime_cell(self, engine, df) -> None:
        result = _edit(engine, df, "day", "2025-05-05")
        assert result["df"].iloc[0]["day"] == pd.Timestamp("2025-05-05")

    def test_a_categorical_cell_with_an_existing_category(self, engine, df) -> None:
        result = _edit(engine, df, "tier", "b")
        assert result["df"].iloc[0]["tier"] == "b"
        assert isinstance(result["df"]["tier"].dtype, pd.CategoricalDtype)


class TestClearingACell:
    """The grid renders a null as an empty input, so an empty string is how
    "no value" comes back."""

    def test_clearing_a_float_cell_makes_it_null(self, engine, df) -> None:
        assert pd.isna(_edit(engine, df, "price", "")["df"].iloc[0]["price"])

    def test_clearing_a_text_cell_makes_it_null(self, engine, df) -> None:
        assert pd.isna(_edit(engine, df, "region", "")["df"].iloc[0]["region"])

    def test_clearing_a_datetime_cell_makes_it_null(self, engine, df) -> None:
        assert pd.isna(_edit(engine, df, "day", "")["df"].iloc[0]["day"])

    def test_clearing_an_integer_cell_makes_it_null(self, engine, df) -> None:
        """int64 has no null to hold, so the column has to give way."""
        result = _edit(engine, df, "units", "")
        assert pd.isna(result["df"].iloc[0]["units"])
        assert result["df"].iloc[1]["units"] == 2

    def test_clearing_a_boolean_cell_makes_it_null(self, engine, df) -> None:
        result = _edit(engine, df, "active", "")
        assert pd.isna(result["df"].iloc[0]["active"])
        assert bool(result["df"].iloc[1]["active"]) is False

    def test_whitespace_counts_as_cleared(self, engine, df) -> None:
        assert pd.isna(_edit(engine, df, "price", "   ")["df"].iloc[0]["price"])


class TestColumnsWidenOnlyWhenTheyMustAndSaySo:
    def test_a_fractional_value_widens_an_integer_column(self, engine, df) -> None:
        """Keeping 9.5 and widening the column beats silently storing 9."""
        result = _edit(engine, df, "units", "9.5")
        assert result["df"].iloc[0]["units"] == 9.5
        assert str(result["df"]["units"].dtype) == "float64"

    def test_a_whole_number_does_not_widen_an_integer_column(self, engine, df) -> None:
        assert str(_edit(engine, df, "units", "42")["df"]["units"].dtype) == "int64"

    def test_a_new_category_widens_a_categorical_column(self, engine, df) -> None:
        """A category rejects anything outside its own values, and editing a
        cell to something new is a legitimate thing to want."""
        result = _edit(engine, df, "tier", "z")
        assert result["df"].iloc[0]["tier"] == "z"
        assert not isinstance(result["df"]["tier"].dtype, pd.CategoricalDtype)

    def test_widening_is_reported(self, engine, df) -> None:
        """A column quietly changing type is not something a user should have
        to discover for themselves."""
        assert "widened to float64" in _edit(engine, df, "units", "9.5")["report"][0]

    def test_nothing_is_reported_when_nothing_widened(self, engine, df) -> None:
        assert _edit(engine, df, "units", "42")["report"][0] == "Edited 1 cell(s)."


class TestAnImpossibleValueIsTheUsersMistake:
    """ProcessingError → 400. Anything else → 500, which is the bug."""

    def test_text_in_a_numeric_column(self, engine, df) -> None:
        with pytest.raises(ProcessingError, match="not a number"):
            _edit(engine, df, "units", "abc")

    def test_the_message_names_the_column(self, engine, df) -> None:
        with pytest.raises(ProcessingError, match="units"):
            _edit(engine, df, "units", "abc")

    def test_a_non_boolean_in_a_boolean_column(self, engine, df) -> None:
        with pytest.raises(ProcessingError, match="true/false"):
            _edit(engine, df, "active", "maybe")

    def test_a_non_date_in_a_datetime_column(self, engine, df) -> None:
        with pytest.raises(ProcessingError, match="not a date"):
            _edit(engine, df, "day", "not a date")

    def test_a_rejected_edit_leaves_the_frame_alone(self, engine, df) -> None:
        before = df.copy()
        with pytest.raises(ProcessingError):
            _edit(engine, df, "units", "abc")
        pd.testing.assert_frame_equal(df, before)


class TestBatchedEdits:
    def test_mixed_types_in_one_batch(self, engine, df) -> None:
        """One save carries every pending edit across every page and column."""
        result = engine.process(
            df,
            {
                "ops": [
                    {
                        "type": "edit_cells",
                        "edits": [
                            {"row_index": 0, "column": "units", "value": "42"},
                            {"row_index": 1, "column": "price", "value": "7.25"},
                            {"row_index": 2, "column": "region", "value": "North"},
                            {"row_index": 0, "column": "active", "value": "no"},
                        ],
                    }
                ]
            },
        )
        out = result["df"]
        assert out.iloc[0]["units"] == 42
        assert out.iloc[1]["price"] == 7.25
        assert out.iloc[2]["region"] == "North"
        assert bool(out.iloc[0]["active"]) is False

    def test_one_bad_edit_rejects_the_whole_batch(self, engine, df) -> None:
        """A save is one op producing one version. Applying the good half of a
        batch would leave the user with a version they never asked for."""
        with pytest.raises(ProcessingError):
            engine.process(
                df,
                {
                    "ops": [
                        {
                            "type": "edit_cells",
                            "edits": [
                                {"row_index": 0, "column": "units", "value": "42"},
                                {"row_index": 1, "column": "units", "value": "abc"},
                            ],
                        }
                    ]
                },
            )

    def test_a_column_widened_once_is_reported_once(self, engine, df) -> None:
        result = engine.process(
            df,
            {
                "ops": [
                    {
                        "type": "edit_cells",
                        "edits": [
                            {"row_index": 0, "column": "units", "value": "9.5"},
                            {"row_index": 1, "column": "units", "value": "8.25"},
                        ],
                    }
                ]
            },
        )
        assert result["report"][0].count("widened") == 1
        assert result["df"].iloc[0]["units"] == 9.5
        assert result["df"].iloc[1]["units"] == 8.25
