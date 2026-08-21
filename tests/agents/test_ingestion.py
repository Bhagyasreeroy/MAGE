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


class TestSingleColumnAndRaggedFiles:
    """
    F3 — `csv.Sniffer` failing meant "Delimiter detection failure", whatever the
    real problem was.

    Two quite different files landed on that one message. A legitimate
    single-column upload has no delimiter to find, which is not an error at all.
    A file with ragged rows *does* have a delimiter — the sniffer just cannot
    settle on one when the field counts disagree — so blaming the delimiter
    points the reader at the wrong thing entirely.
    """

    @pytest.fixture
    def engine(self) -> DataIngestionEngine:
        return DataIngestionEngine()

    @pytest.mark.parametrize(
        ("content", "column", "rows"),
        [
            (b"only\n1\n2\n3\n", "only", 3),
            (b"note\nalpha\nbravo\n", "note", 2),
            # Trailing blank lines are ordinary in hand-made files.
            (b"value\n10\n20\n\n", "value", 2),
        ],
    )
    def test_a_single_column_file_loads(
        self, engine: DataIngestionEngine, content: bytes, column: str, rows: int,
    ) -> None:
        df = engine._load_csv(content)

        assert list(df.columns) == [column]
        assert len(df) == rows

    def test_ragged_rows_are_named_as_such(self, engine: DataIngestionEngine) -> None:
        """
        The file has commas and a clear header. The problem is that line 3 has
        four fields where the header declares three — say that, and say where.
        """
        content = b"a,b,c\n1,2,3\n4,5,6,7\n8,9,10\n"

        with pytest.raises(IngestionError) as exc:
            engine._load_csv(content)

        message = str(exc.value).lower()
        assert "delimiter" not in message, (
            "the delimiter is not the problem here — the message must not say it is"
        )
        assert "3" in message and "4" in message, "name the expected and actual field counts"
        assert "line" in message or "row" in message, "name where it happened"

    def test_a_single_column_file_is_not_confused_with_a_broken_one(
        self, engine: DataIngestionEngine,
    ) -> None:
        """
        The guard that makes the single-column fallback safe: fall back only
        when *no* candidate delimiter occurs anywhere. A file that does contain
        one is not single-column, whatever the sniffer decided, and silently
        parsing it into one column would turn a broken file into a wrong one.
        """
        content = b"a;b\n1;2\n3;4;5\n"

        with pytest.raises(IngestionError) as exc:
            engine._load_csv(content)

        assert "delimiter" not in str(exc.value).lower()

    @pytest.mark.parametrize(
        ("content", "shape"),
        [
            (b"a,b\n1,2\n3,4\n", (2, 2)),
            (b"a;b\n1;2\n3;4\n", (2, 2)),
            (b"a\tb\n1\t2\n3\t4\n", (2, 2)),
            (b"a|b\n1|2\n3|4\n", (2, 2)),
        ],
    )
    def test_normal_delimited_files_are_untouched(
        self, engine: DataIngestionEngine, content: bytes, shape: tuple[int, int],
    ) -> None:
        """Control: the fallback must not change how a real CSV parses."""
        assert engine._load_csv(content).shape == shape
