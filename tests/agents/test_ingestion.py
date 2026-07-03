"""
tests/agents/test_ingestion.py
───────────────────────────────
Tests for M1: IngestionAgent + DataIngestionEngine loading real files.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.ingestion_agent import IngestionAgent
from data_pipeline.ingestion import DataIngestionEngine

# Path to the sample CSV shipped in the repo.
SAMPLE_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "samples", "sales.csv"
)


class TestDataIngestionEngine:
    """Unit tests for the low-level loader."""

    def test_load_csv_row_count(self) -> None:
        result = DataIngestionEngine().load(SAMPLE_CSV)
        assert result["row_count"] == 6

    def test_load_csv_columns(self) -> None:
        result = DataIngestionEngine().load(SAMPLE_CSV)
        assert result["columns"] == [
            "order_id",
            "region",
            "product",
            "units",
            "revenue",
        ]

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            DataIngestionEngine().load("data/samples/does_not_exist.csv")

    def test_load_unsupported_type_raises(self) -> None:
        with pytest.raises(ValueError):
            DataIngestionEngine().load("data/samples/notes.txt")


class TestIngestionAgent:
    """Tests for the agent wrapper consumed by the Orchestrator."""

    @pytest.fixture
    def agent(self) -> IngestionAgent:
        return IngestionAgent()

    def test_run_returns_real_row_count(self, agent: IngestionAgent) -> None:
        result = agent.run(context={"data": {"path": SAMPLE_CSV}})
        assert result["row_count"] == 6

    def test_run_returns_schema(self, agent: IngestionAgent) -> None:
        result = agent.run(context={"data": {"path": SAMPLE_CSV}})
        assert set(result["schema"].keys()) == {
            "order_id",
            "region",
            "product",
            "units",
            "revenue",
        }

    def test_run_flags_missing_values(self, agent: IngestionAgent) -> None:
        result = agent.run(context={"data": {"path": SAMPLE_CSV}})
        # 'units' and 'revenue' each have one blank cell in the sample.
        warned_columns = " ".join(result["quality_warnings"])
        assert "units" in warned_columns
        assert "revenue" in warned_columns

    def test_run_without_data_is_graceful(self, agent: IngestionAgent) -> None:
        """No path in context should not raise — keeps the skeleton smoke run green."""
        result = agent.run(context={"goal": "profile data", "data": {}})
        assert result["row_count"] == 0
        assert result["quality_warnings"] == []
