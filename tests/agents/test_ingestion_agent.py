"""Tests for the ingestion agent."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi import UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.ingestion_agent import IngestionAgent
from data_pipeline.ingestion import IngestionError


class TestIngestionAgent:
    @pytest.fixture
    def agent(self) -> IngestionAgent:
        return IngestionAgent()

    def test_run_happy_path(self, agent: IngestionAgent, tmp_path: Path) -> None:
        sample = tmp_path / "sample.csv"
        sample.write_text(
            "id,score,category\n"
            "1,10.5,a\n"
            "2,20.0,b\n"
            "3,30.5,a\n",
            encoding="utf-8",
        )

        result = agent.run(source=sample)

        assert result.row_count == 3
        assert result.column_count == 3
        assert [column.name for column in result.column_summary] == ["id", "score", "category"]
        assert [column.dtype for column in result.column_summary] == ["int64", "float64", "str"]
        assert result.warnings == []

        score_summary = result.column_summary[1]
        assert score_summary.stats is not None
        assert score_summary.stats.min == 10.5
        assert score_summary.stats.max == 30.5
        assert score_summary.stats.mean == pytest.approx(20.3333333333)

        category_summary = result.column_summary[2]
        assert category_summary.stats is not None
        assert category_summary.stats.unique_count == 2

    def test_run_missing_values_populates_warnings(self, agent: IngestionAgent, tmp_path: Path) -> None:
        sample = tmp_path / "missing.csv"
        sample.write_text(
            "id,value,category\n"
            "1,10,a\n"
            "2,,b\n"
            "3,,c\n"
            "4,40,d\n",
            encoding="utf-8",
        )

        result = agent.run(source=sample)

        assert result.row_count == 4
        assert result.column_count == 3
        assert any("value" in warning and "50.0% missing" in warning for warning in result.warnings)
        value_summary = next(column for column in result.column_summary if column.name == "value")
        assert value_summary.missing_count == 2

    def test_run_unsupported_file_type_raises(self, agent: IngestionAgent, tmp_path: Path) -> None:
        sample = tmp_path / "sample.txt"
        sample.write_text("not supported", encoding="utf-8")

        with pytest.raises(IngestionError, match="Unsupported extension"):
            agent.run(source=sample)

    def test_run_empty_file_raises(self, agent: IngestionAgent, tmp_path: Path) -> None:
        sample = tmp_path / "empty.csv"
        sample.write_text("", encoding="utf-8")

        with pytest.raises(IngestionError, match="Empty file"):
            agent.run(source=sample)

    def test_run_upload_file_object(self, agent: IngestionAgent) -> None:
        upload = UploadFile(
            filename="demo.csv",
            file=io.BytesIO(
                b"id,value,category\n"
                b"1,10,a\n"
                b"2,20,b\n",
            ),
        )

        result = agent.run(source=upload)

        assert result.row_count == 2
        assert result.column_count == 3
        assert [column.name for column in result.column_summary] == ["id", "value", "category"]
