"""
tests/agents/test_ingestion_sources.py
───────────────────────────────────────
M1 tests for the non-file ingestion sources added to DataIngestionEngine:
PDF-table extraction, SQL database URIs, and REST/HTTP endpoints — plus the
``classify_source`` router that decides which path a source takes.

Complements test_ingestion.py (file formats) and test_ingestion_agent.py
(profiling contract).
"""

import io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.ingestion_agent import IngestionAgent
from data_pipeline.ingestion import DataIngestionEngine, IngestionError

EXPECTED_COLUMNS = ["order_id", "region", "product", "units", "revenue"]


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "region": ["North", "South", "East"],
            "product": ["A", "B", "C"],
            "units": [10, 20, 30],
            "revenue": [100.5, 200.0, 300.25],
        }
    )


# ── Source classification ────────────────────────────────────────────────────


class TestClassifySource:
    def test_plain_path_is_file(self) -> None:
        assert DataIngestionEngine.classify_source("data/sales.csv") == "file"

    def test_http_url_is_url(self) -> None:
        assert DataIngestionEngine.classify_source("https://api.example.com/data") == "url"

    @pytest.mark.parametrize(
        "uri",
        [
            "sqlite:///local.db",
            "postgresql://u:p@h/db",
            "postgresql+psycopg2://u:p@h/db",
            "mysql://u:p@h/db",
        ],
    )
    def test_db_uri_is_database(self, uri: str) -> None:
        assert DataIngestionEngine.classify_source(uri) == "database"

    def test_upload_like_object_is_file(self) -> None:
        class FakeUpload:
            filename = "x.csv"
            file = io.BytesIO(b"a,b\n1,2\n")

        assert DataIngestionEngine.classify_source(FakeUpload()) == "file"


# ── PDF table extraction ─────────────────────────────────────────────────────


def _make_pdf_with_table(rows: list[list[str]]) -> bytes:
    """Render a bordered table to a PDF so pdfplumber can extract it."""
    reportlab = pytest.importorskip("reportlab")  # noqa: F841
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf)
    table = Table(rows)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    doc.build([table])
    return buf.getvalue()


class TestPdfIngestion:
    def test_extracts_table(self, tmp_path) -> None:
        pytest.importorskip("pdfplumber")
        rows = [EXPECTED_COLUMNS, ["1", "North", "A", "10", "100.5"], ["2", "South", "B", "20", "200"]]
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(_make_pdf_with_table(rows))

        df = DataIngestionEngine().load(str(pdf_path))
        assert list(df.columns) == EXPECTED_COLUMNS
        assert len(df) == 2

    def test_numeric_columns_recovered(self, tmp_path) -> None:
        pytest.importorskip("pdfplumber")
        rows = [["id", "score"], ["1", "10"], ["2", "20"]]
        pdf_path = tmp_path / "nums.pdf"
        pdf_path.write_bytes(_make_pdf_with_table(rows))

        df = DataIngestionEngine().load(str(pdf_path))
        assert pd.api.types.is_numeric_dtype(df["score"])

    def test_pdf_without_tables_raises(self, tmp_path) -> None:
        pytest.importorskip("pdfplumber")
        from reportlab.pdfgen import canvas

        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        c.drawString(100, 750, "Just some prose, no tables here.")
        c.save()
        pdf_path = tmp_path / "prose.pdf"
        pdf_path.write_bytes(buf.getvalue())

        with pytest.raises(IngestionError, match="No tables found"):
            DataIngestionEngine().load(str(pdf_path))


# ── Database URI ingestion ───────────────────────────────────────────────────


class TestDatabaseIngestion:
    def _make_db(self, tmp_path, tables: dict[str, pd.DataFrame]) -> str:
        db_path = tmp_path / "test.db"
        uri = f"sqlite:///{db_path}"
        from sqlalchemy import create_engine

        engine = create_engine(uri)
        for name, df in tables.items():
            df.to_sql(name, engine, index=False, if_exists="replace")
        engine.dispose()
        return uri

    def test_single_table_auto_loaded(self, tmp_path) -> None:
        uri = self._make_db(tmp_path, {"orders": _sample_frame()})
        df = DataIngestionEngine().load_from_database(uri)
        assert list(df.columns) == EXPECTED_COLUMNS
        assert len(df) == 3

    def test_explicit_table(self, tmp_path) -> None:
        uri = self._make_db(tmp_path, {"orders": _sample_frame(), "extra": _sample_frame().head(1)})
        df = DataIngestionEngine().load_from_database(uri, table="extra")
        assert len(df) == 1

    def test_uri_fragment_selects_table(self, tmp_path) -> None:
        uri = self._make_db(tmp_path, {"orders": _sample_frame(), "extra": _sample_frame().head(2)})
        df = DataIngestionEngine().load_from_database(f"{uri}#extra")
        assert len(df) == 2

    def test_custom_query(self, tmp_path) -> None:
        uri = self._make_db(tmp_path, {"orders": _sample_frame()})
        df = DataIngestionEngine().load_from_database(uri, query="SELECT * FROM orders WHERE units > 15")
        assert len(df) == 2

    def test_multiple_tables_requires_choice(self, tmp_path) -> None:
        uri = self._make_db(tmp_path, {"a": _sample_frame(), "b": _sample_frame()})
        with pytest.raises(IngestionError, match="Multiple tables found"):
            DataIngestionEngine().load_from_database(uri)

    def test_table_and_query_conflict(self, tmp_path) -> None:
        uri = self._make_db(tmp_path, {"orders": _sample_frame()})
        with pytest.raises(IngestionError, match="not both"):
            DataIngestionEngine().load_from_database(uri, table="orders", query="SELECT 1")


# ── REST / URL ingestion (httpx mocked) ──────────────────────────────────────


class _FakeResponse:
    def __init__(self, content: bytes, content_type: str) -> None:
        self.content = content
        self.headers = {"content-type": content_type}

    def raise_for_status(self) -> None:
        pass


class TestUrlIngestion:
    def test_json_endpoint(self, monkeypatch) -> None:
        import httpx

        payload = _sample_frame().to_json(orient="records").encode()
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(payload, "application/json"))

        df = DataIngestionEngine().load_from_url("https://api.example.com/orders")
        assert list(df.columns) == EXPECTED_COLUMNS
        assert len(df) == 3

    def test_csv_endpoint_by_content_type(self, monkeypatch) -> None:
        import httpx

        payload = _sample_frame().to_csv(index=False).encode()
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(payload, "text/csv"))

        df = DataIngestionEngine().load_from_url("https://api.example.com/orders")
        assert len(df) == 3

    def test_format_hint_overrides(self, monkeypatch) -> None:
        import httpx

        payload = _sample_frame().to_csv(index=False).encode()
        # Server lies about content-type; explicit hint wins.
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(payload, "application/octet-stream"))

        df = DataIngestionEngine().load_from_url("https://x/y", format_hint="csv")
        assert len(df) == 3

    def test_empty_body_raises(self, monkeypatch) -> None:
        import httpx

        monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(b"", "application/json"))
        with pytest.raises(IngestionError, match="empty body"):
            DataIngestionEngine().load_from_url("https://x/y")


# ── Agent-level routing ──────────────────────────────────────────────────────


class TestAgentRouting:
    def test_agent_routes_database_source(self, tmp_path) -> None:
        db_path = tmp_path / "agent.db"
        uri = f"sqlite:///{db_path}"
        from sqlalchemy import create_engine

        engine = create_engine(uri)
        _sample_frame().to_sql("orders", engine, index=False, if_exists="replace")
        engine.dispose()

        context: dict = {"source": uri}
        result = IngestionAgent().run(context=context)
        assert result.row_count == 3
        assert result.column_count == 5
        # The canonical DataFrame is placed into context for downstream agents.
        assert "dataframe" in context

    def test_agent_routes_url_source(self, monkeypatch) -> None:
        import httpx

        payload = _sample_frame().to_json(orient="records").encode()
        monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(payload, "application/json"))

        context: dict = {"source": "https://api.example.com/orders"}
        result = IngestionAgent().run(context=context)
        assert result.row_count == 3
