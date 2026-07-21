"""
tests/agents/test_ingestion.py
───────────────────────────────
M1 tests: DataIngestionEngine + IngestionAgent.

Covers multi-format loading (CSV/TSV/JSON/Parquet/Excel), error handling,
and the agent's data-quality profiling.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.ingestion_agent import IngestionAgent
from data_pipeline.ingestion import DataIngestionEngine

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
        assert DataIngestionEngine().load(path)["row_count"] == 6

    @pytest.mark.parametrize(
        "path",
        [SAMPLE_CSV, SAMPLE_JSON, SAMPLE_PARQUET, SAMPLE_XLSX],
        ids=["csv", "json", "parquet", "xlsx"],
    )
    def test_load_columns(self, path: str) -> None:
        assert DataIngestionEngine().load(path)["columns"] == EXPECTED_COLUMNS

    def test_load_reports_format(self) -> None:
        assert DataIngestionEngine().load(SAMPLE_CSV)["format"] == ".csv"

    def test_load_tsv(self, tmp_path) -> None:
        p = tmp_path / "data.tsv"
        p.write_text("a\tb\n1\t2\n3\t4\n")
        result = DataIngestionEngine().load(str(p))
        assert result["row_count"] == 2
        assert result["columns"] == ["a", "b"]


class TestDataIngestionEngineErrors:
    """The engine fails cleanly with informative exceptions."""

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            DataIngestionEngine().load(os.path.join(SAMPLES, "does_not_exist.csv"))

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(ValueError):
            DataIngestionEngine().load(os.path.join(SAMPLES, "notes.txt"))

    def test_empty_file_raises_valueerror(self, tmp_path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(ValueError):
            DataIngestionEngine().load(str(p))


class TestIngestionAgent:
    """The agent wraps the engine and produces schema + quality warnings."""

    @pytest.fixture
    def agent(self) -> IngestionAgent:
        return IngestionAgent()

    def test_run_returns_real_row_count(self, agent: IngestionAgent) -> None:
        result = agent.run(context={"data": {"path": SAMPLE_CSV}})
        assert result["row_count"] == 6

    def test_run_returns_schema(self, agent: IngestionAgent) -> None:
        result = agent.run(context={"data": {"path": SAMPLE_CSV}})
        assert set(result["schema"].keys()) == set(EXPECTED_COLUMNS)

    def test_run_flags_missing_values(self, agent: IngestionAgent) -> None:
        result = agent.run(context={"data": {"path": SAMPLE_CSV}})
        warned = " ".join(result["quality_warnings"])
        assert "units" in warned and "revenue" in warned

    def test_run_without_data_is_graceful(self, agent: IngestionAgent) -> None:
        result = agent.run(context={"goal": "profile", "data": {}})
        assert result["row_count"] == 0
        assert result["quality_warnings"] == []

    def test_run_bad_file_returns_warning_not_raise(self, agent: IngestionAgent) -> None:
        result = agent.run(context={"data": {"path": "/no/such/file.csv"}})
        assert result["row_count"] == 0
        assert any("error" in w.lower() for w in result["quality_warnings"])


class TestQualityProfiling:
    """The agent detects duplicates, constant columns, and empty datasets."""

    @pytest.fixture
    def agent(self) -> IngestionAgent:
        return IngestionAgent()

    def test_detects_duplicate_rows(self, agent: IngestionAgent, tmp_path) -> None:
        p = tmp_path / "dupes.csv"
        pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]}).to_csv(p, index=False)
        warned = " ".join(agent.run(context={"data": {"path": str(p)}})["quality_warnings"])
        assert "duplicate" in warned.lower()

    def test_detects_constant_column(self, agent: IngestionAgent, tmp_path) -> None:
        p = tmp_path / "const.csv"
        pd.DataFrame({"a": [1, 2, 3], "flag": ["Y", "Y", "Y"]}).to_csv(p, index=False)
        warned = " ".join(agent.run(context={"data": {"path": str(p)}})["quality_warnings"])
        assert "constant" in warned.lower()

    def test_detects_empty_dataset(self, agent: IngestionAgent, tmp_path) -> None:
        p = tmp_path / "headers_only.csv"
        p.write_text("a,b,c\n")
        warned = " ".join(agent.run(context={"data": {"path": str(p)}})["quality_warnings"])
        assert "empty" in warned.lower()
