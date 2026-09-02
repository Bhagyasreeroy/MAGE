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
