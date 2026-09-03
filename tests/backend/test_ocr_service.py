"""
tests/backend/test_ocr_service.py
───────────────────────────────────
Unit and integration tests for OCR.space integration service and API endpoints.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app
from backend.services.ocr_service import OCRServiceError, OCRSpaceService, ocr_service
from backend.core.deps import get_current_user
from backend.models.user import User

client = TestClient(app)

MOCK_USER = User(
    id="test-user-id",
    email="test@example.com",
    full_name="Test User",
    auth_provider="local",
)


def mock_get_current_user():
    return MOCK_USER


@pytest.fixture
def ocr_test_payload():
    return {
        "ParsedResults": [
            {
                "TextOverlay": {},
                "TextOrientation": "0",
                "FileParseExitCode": 1,
                "ParsedText": "Header1  Header2\nVal1     Val2",
                "ErrorMessage": "",
                "ErrorDetails": "",
            }
        ],
        "OCRExitCode": 1,
        "IsErroredOnProcessing": False,
        "OCREngineExecuted": "2",
    }


class TestOCRSpaceService:

    @pytest.mark.asyncio
    async def test_parse_image_bytes_success(self, ocr_test_payload):
        service = OCRSpaceService(api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=ocr_test_payload)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await service.parse_image_bytes(b"dummy image bytes", "test.png")

        assert res["status"] == "success"
        assert res["exit_code"] == 1
        assert "Header1" in res["full_text"]
        assert len(res["pages"]) == 1
        assert res["stats"]["word_count"] > 0

    @pytest.mark.asyncio
    async def test_parse_image_url_success(self, ocr_test_payload):
        service = OCRSpaceService(api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=ocr_test_payload)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await service.parse_image_url("https://example.com/sample.png")

        assert res["status"] == "success"
        assert len(res["table_rows"]) >= 1

    @pytest.mark.asyncio
    async def test_ocr_api_error_raises_exception(self):
        service = OCRSpaceService(api_key="test_key")

        error_payload = {
            "IsErroredOnProcessing": True,
            "ErrorMessage": ["Invalid API Key"],
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=error_payload)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(OCRServiceError, match="Invalid API Key"):
                await service.parse_image_bytes(b"bytes", "image.png")

    def test_convert_ocr_to_csv(self):
        service = OCRSpaceService()
        ocr_result = {
            "table_rows": [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]],
            "full_text": "Name Age\nAlice 30\nBob 25",
        }

        csv_out = service.convert_ocr_to_csv(ocr_result)
        assert "Name,Age" in csv_out
        assert "Alice,30" in csv_out


class TestOCREndpoints:

    def test_process_file_endpoint(self, ocr_test_payload):
        app.dependency_overrides[get_current_user] = mock_get_current_user

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=ocr_test_payload)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            response = client.post(
                "/api/v1/ocr/process-file",
                files={"file": ("invoice.png", b"fake file data", "image/png")},
                data={"language": "eng", "is_table": "true", "ocr_engine": "2"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["stats"]["page_count"] == 1

    def test_process_url_endpoint(self, ocr_test_payload):
        app.dependency_overrides[get_current_user] = mock_get_current_user

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=ocr_test_payload)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            response = client.post(
                "/api/v1/ocr/process-url",
                json={
                    "url": "https://example.com/invoice.pdf",
                    "language": "eng",
                    "is_table": True,
                    "ocr_engine": 2,
                },
            )

        app.dependency_overrides.clear()
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"


class TestNormalizeTableRows:
    """OCR returns the whole page; only the grid part should become a dataset."""

    def test_page_title_is_not_used_as_the_header(self):
        from backend.services.ocr_service import normalize_table_rows

        rows = normalize_table_rows([
            ["Sample Invoice for OCR Testing"],
            ["Item Name", "Quantity", "Unit Price", "Total"],
            ["Cloud Server", "2", "150.00", "300.00"],
        ])

        assert rows[0] == ["Item Name", "Quantity", "Unit Price", "Total"]
        assert rows[1:] == [["Cloud Server", "2", "150.00", "300.00"]]

    def test_short_trailing_row_is_padded_not_dropped(self):
        from backend.services.ocr_service import normalize_table_rows

        rows = normalize_table_rows([
            ["Item", "Qty", "Total"],
            ["Cloud Server", "2", "300.00"],
            ["Total", "395.00"],
        ])

        assert rows[-1] == ["Total", "395.00", ""]

    def test_duplicate_header_cells_are_made_unique(self):
        from backend.services.ocr_service import normalize_table_rows

        rows = normalize_table_rows([["Total", "Total", ""], ["1", "2", "3"]])

        assert rows[0] == ["Total", "Total_2", "column_3"]

    def test_text_only_document_keeps_its_lines(self):
        from backend.services.ocr_service import normalize_table_rows

        assert normalize_table_rows([["a line"], ["another"]]) == [["a line"], ["another"]]
        assert normalize_table_rows([]) == []

    def test_csv_from_a_titled_invoice_is_rectangular(self):
        import csv
        import io

        from backend.services.ocr_service import OCRSpaceService

        csv_text = OCRSpaceService().convert_ocr_to_csv({
            "table_rows": [
                ["Sample Invoice for OCR Testing"],
                ["Item Name", "Quantity", "Total"],
                ["Cloud Server", "2", "300.00"],
            ],
            "full_text": "irrelevant",
        })

        parsed = list(csv.reader(io.StringIO(csv_text)))
        assert parsed[0] == ["Item Name", "Quantity", "Total"]
        assert all(len(row) == 3 for row in parsed)


class TestTlsFailureMessage:
    """A TLS-intercepting proxy is the common cause and has a concrete fix."""

    def test_certificate_failure_names_the_ca_bundle_setting(self):
        import httpx

        from backend.services.ocr_service import _connection_detail

        exc = httpx.ConnectError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
            "self-signed certificate in certificate chain (_ssl.c:1016)"
        )

        detail = _connection_detail(exc)
        assert "OCR_CA_BUNDLE" in detail

    def test_other_network_errors_are_reported_plainly(self):
        import httpx

        from backend.services.ocr_service import _connection_detail

        assert "OCR_CA_BUNDLE" not in _connection_detail(httpx.ConnectTimeout("timed out"))
