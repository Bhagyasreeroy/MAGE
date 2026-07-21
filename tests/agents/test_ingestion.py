"""
tests/agents/test_ingestion.py
───────────────────────────────
M1 tests: DataIngestionEngine multi-format loading (CSV/TSV/JSON/Parquet/Excel).

Complements test_ingestion_agent.py, which covers the IngestionAgent's
Pydantic-based profiling contract in detail.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_pipeline.ingestion import DataIngestionEngine, IngestionError

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "data", "samples")
SAMPLE_CSV = os.path.join(SAMPLES, "sales.csv")
SAMPLE_JSON = os.path.join(SAMPLES, "sales.json")
SAMPLE_PARQUET = os.path.join(SAMPLES, "sales.parquet")
SAMPLE_XLSX = os.path.join(SAMPLES, "sales.xlsx")

EXPECTED_COLUMNS = ["order_id", "region", "product", "units", "revenue"]


class TestDataIngestionEngineFormats:
    """The engine loads every supported tabular format to a real DataFrame."""

    @pytest.mark.parametrize(
        "path",
        [SAMPLE_CSV, SAMPLE_JSON, SAMPLE_PARQUET, SAMPLE_XLSX],
        ids=["csv", "json", "parquet", "xlsx"],
    )
    def test_load_row_count(self, path: str) -> None:
        assert len(DataIngestionEngine().load(path)) == 6

    @pytest.mark.parametrize(
        "path",
        [SAMPLE_CSV, SAMPLE_JSON, SAMPLE_PARQUET, SAMPLE_XLSX],
        ids=["csv", "json", "parquet", "xlsx"],
    )
    def test_load_columns(self, path: str) -> None:
        assert list(DataIngestionEngine().load(path).columns) == EXPECTED_COLUMNS

    def test_load_tsv(self, tmp_path) -> None:
        p = tmp_path / "data.tsv"
        p.write_text("a\tb\n1\t2\n3\t4\n")
        df = DataIngestionEngine().load(str(p))
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]


class TestDataIngestionEngineErrors:
    """The engine fails cleanly with informative exceptions."""

    def test_missing_file_raises(self) -> None:
        with pytest.raises(IngestionError, match="File not found"):
            DataIngestionEngine().load(os.path.join(SAMPLES, "does_not_exist.csv"))

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(IngestionError, match="Unsupported extension"):
            DataIngestionEngine().load(os.path.join(SAMPLES, "notes.txt"))

    def test_empty_file_raises(self, tmp_path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(IngestionError, match="Empty file"):
            DataIngestionEngine().load(str(p))
