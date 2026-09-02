"""
tests/backend/test_ocr_validation.py
────────────────────────────────────
Upload validation for the OCR endpoints: which file types are allowed, how
oversized uploads are refused, and how an upstream OCR.space failure is
reported back to the caller.

These never reach the network — every accepted upload is intercepted by a
stub service, so a test that gets as far as "the stub ran" has proved the
file passed validation.
"""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.core.deps import get_current_user
from backend.main import app
from backend.models.user import User
from backend.routers import ocr as ocr_router
from backend.services.ocr_service import OCRServiceError

client = TestClient(app)

# Real magic bytes, negligible size. Content does not matter: validation is
# what these exercise, and the OCR call itself is stubbed.
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PDF = b"%PDF-1.4\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64



async def _no_sleep(_seconds: float) -> None:
    """Collapse retry backoff so the tests stay fast."""
    return None

@pytest.fixture
def authed():
    app.dependency_overrides[get_current_user] = lambda: User(
        id="ocr-validation-user", email="ocr@example.com", is_active=True
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def stub_ocr(monkeypatch):
    """Replace the network call; record that validation let the file through."""
    calls: dict = {}

    async def fake_parse(content, filename, **kwargs):
        calls["filename"] = filename
        calls["size"] = len(content)
        return {
            "status": "success",
            "ocr_engine": "2",
            "exit_code": 1,
            "full_text": "Product\tUnits\nWidget A\t120",
            "pages": [{"page_number": 1, "parsed_text": "Product\tUnits\nWidget A\t120",
                       "lines": ["Product\tUnits", "Widget A\t120"], "exit_code": 1, "error_message": ""}],
            "table_rows": [["Product", "Units"], ["Widget A", "120"]],
            "stats": {"page_count": 1, "word_count": 4, "character_count": 26},
        }

    monkeypatch.setattr(ocr_router.ocr_service, "parse_image_bytes", fake_parse)
    return calls


class TestAcceptedFileTypes:
    """Images and PDF are the supported inputs; both must pass validation."""

    @pytest.mark.parametrize(
        "filename,content_type,body",
        [
            ("scan.png", "image/png", PNG),
            ("photo.jpg", "image/jpeg", JPEG),
            ("photo.jpeg", "image/jpeg", JPEG),
            ("receipt.pdf", "application/pdf", PDF),
        ],
    )
    def test_supported_types_reach_the_ocr_service(
        self, authed, stub_ocr, filename, content_type, body
    ) -> None:
        res = client.post(
            "/api/v1/ocr/process-file",
            files={"file": (filename, body, content_type)},
        )
        assert res.status_code == 200, res.text
        assert stub_ocr["filename"] == filename


class TestRejectedFileTypes:
    """Everything else is refused before a request is ever sent upstream."""

    @pytest.mark.parametrize(
        "filename,content_type",
        [
            ("data.csv", "text/csv"),
            ("book.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ("notes.txt", "text/plain"),
            ("archive.zip", "application/zip"),
            ("script.py", "text/x-python"),
        ],
    )
    def test_unsupported_types_are_refused(self, authed, stub_ocr, filename, content_type) -> None:
        res = client.post(
            "/api/v1/ocr/process-file",
            files={"file": (filename, b"\x00" * 32, content_type)},
        )
        # 415, not the 502 a rejected upstream call would produce: this is the
        # caller's mistake and must never cost an API request.
        assert res.status_code == 415, res.text
        assert "filename" not in stub_ocr, "unsupported file reached the OCR service"

    def test_the_refusal_names_the_supported_types(self, authed, stub_ocr) -> None:
        res = client.post(
            "/api/v1/ocr/process-file", files={"file": ("data.csv", b"a,b\n1,2", "text/csv")}
        )
        detail = res.json()["detail"].lower()
        assert "pdf" in detail and ("image" in detail or "png" in detail)

    def test_convert_to_dataset_applies_the_same_rule(self, authed, stub_ocr) -> None:
        res = client.post(
            "/api/v1/ocr/convert-to-dataset",
            files={"file": ("data.csv", b"a,b\n1,2", "text/csv")},
        )
        assert res.status_code == 415, res.text


class TestSizeLimit:
    """OCR.space's free plan hard-rejects over 1.5 MB, so refuse locally first."""

    def test_oversized_upload_is_refused_before_the_api_call(self, authed, stub_ocr) -> None:
        oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * (2 * 1024 * 1024)
        res = client.post(
            "/api/v1/ocr/process-file",
            files={"file": ("huge.png", oversized, "image/png")},
        )
        assert res.status_code == 413, res.text
        assert "filename" not in stub_ocr, "oversized file was sent upstream anyway"

    def test_the_size_refusal_states_the_limit(self, authed, stub_ocr) -> None:
        oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * (2 * 1024 * 1024)
        res = client.post(
            "/api/v1/ocr/process-file", files={"file": ("huge.png", oversized, "image/png")}
        )
        assert "1.5" in res.json()["detail"]

    def test_a_file_just_under_the_limit_is_accepted(self, authed, stub_ocr) -> None:
        ok = b"\x89PNG\r\n\x1a\n" + b"\x00" * (1_400_000)
        res = client.post(
            "/api/v1/ocr/process-file", files={"file": ("big.png", ok, "image/png")}
        )
        assert res.status_code == 200, res.text


class TestUpstreamErrorReporting:
    """An OCR.space failure should say what went wrong, not just echo a code."""

    def test_upstream_message_reaches_the_caller(self, authed, monkeypatch) -> None:
        async def failing(content, filename, **kwargs):
            raise OCRServiceError("OCR API error: E216: Unable to recognise the file type")

        monkeypatch.setattr(ocr_router.ocr_service, "parse_image_bytes", failing)
        res = client.post(
            "/api/v1/ocr/process-file", files={"file": ("scan.png", PNG, "image/png")}
        )
        assert res.status_code == 502
        assert "E216" in res.json()["detail"]


class TestUrlEndpointAppliesTheSameRule:
    """A URL is just another way in — it must not bypass the type rule."""

    @pytest.fixture
    def stub_url_ocr(self, monkeypatch):
        calls: dict = {}

        async def fake_parse_url(url, **kwargs):
            calls["url"] = url
            return {
                "status": "success", "ocr_engine": "2", "exit_code": 1,
                "full_text": "hello", "pages": [], "table_rows": [],
                "stats": {"page_count": 1, "word_count": 1, "character_count": 5},
            }

        monkeypatch.setattr(ocr_router.ocr_service, "parse_image_url", fake_parse_url)
        return calls

    def test_an_image_url_is_accepted(self, authed, stub_url_ocr) -> None:
        res = client.post(
            "/api/v1/ocr/process-url", json={"url": "https://example.com/scan.png"}
        )
        assert res.status_code == 200, res.text
        assert stub_url_ocr["url"].endswith("scan.png")

    def test_a_pdf_url_is_accepted(self, authed, stub_url_ocr) -> None:
        res = client.post(
            "/api/v1/ocr/process-url", json={"url": "https://example.com/report.pdf"}
        )
        assert res.status_code == 200, res.text

    def test_a_csv_url_is_refused(self, authed, stub_url_ocr) -> None:
        res = client.post(
            "/api/v1/ocr/process-url", json={"url": "https://example.com/data.csv"}
        )
        assert res.status_code == 415, res.text
        assert "url" not in stub_url_ocr, "a CSV URL was sent to the OCR service"

    def test_a_url_with_a_query_string_still_reads_the_extension(
        self, authed, stub_url_ocr
    ) -> None:
        # Signed URLs (S3, CDN) carry query parameters; the extension sits
        # before them and must not be mistaken for part of the path.
        res = client.post(
            "/api/v1/ocr/process-url",
            json={"url": "https://example.com/scan.png?signature=abc123&expires=999"},
        )
        assert res.status_code == 200, res.text

    def test_an_extensionless_url_is_allowed_through(self, authed, stub_url_ocr) -> None:
        # Plenty of legitimate image URLs carry no extension at all. Guessing
        # would block real images, so the upstream API gets to decide.
        res = client.post("/api/v1/ocr/process-url", json={"url": "https://example.com/image"})
        assert res.status_code == 200, res.text


class TestUpstreamErrorDetail:
    """A failure the user can act on must say what actually went wrong."""

    @pytest.mark.asyncio
    async def test_service_surfaces_the_upstream_message_not_just_the_code(self) -> None:
        """OCR.space explains its refusals ("E556: File too large. Max 1.5 MB");
        reporting only "status code 413" throws that away and leaves the user
        with nothing to act on."""
        import httpx

        from backend.services.ocr_service import OCRSpaceService

        service = OCRSpaceService(api_key="test_key")

        request = httpx.Request("POST", "https://api.ocr.space/parse/image")
        response = httpx.Response(
            413,
            request=request,
            text='{"error":"E556: File too large. Max 1.5 MB for Free Plan"}',
        )

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return response

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(httpx, "AsyncClient", lambda **k: _Client())
            with pytest.raises(OCRServiceError) as excinfo:
                await service.parse_image_bytes(content=b"\x89PNG data", filename="big.png")

        assert "E556" in str(excinfo.value)


class TestNoTextFound:
    """OCR that reads nothing must not manufacture an empty dataset."""

    @pytest.fixture
    def stub_empty_ocr(self, monkeypatch):
        async def fake_parse(content, filename, **kwargs):
            return {
                "status": "success", "ocr_engine": "2", "exit_code": 1,
                "full_text": "", "pages": [], "table_rows": [],
                "stats": {"page_count": 1, "word_count": 0, "character_count": 0},
            }

        monkeypatch.setattr(ocr_router.ocr_service, "parse_image_bytes", fake_parse)

    def test_convert_to_dataset_refuses_when_no_text_was_found(
        self, authed, stub_empty_ocr
    ) -> None:
        """A photo of a wall produces a 0-row, 1-column dataset that breaks
        every downstream step. Refusing is both honest and more useful than
        "processed successfully"."""
        res = client.post(
            "/api/v1/ocr/convert-to-dataset",
            files={"file": ("wall.png", PNG, "image/png")},
        )
        assert res.status_code == 422, res.text
        detail = res.json()["detail"].lower()
        assert "no text" in detail or "no readable" in detail

    def test_process_file_still_returns_the_empty_result(self, authed, stub_empty_ocr) -> None:
        """The raw endpoint is a different contract: "I read nothing" is a
        legitimate answer there, since nothing is being persisted."""
        res = client.post(
            "/api/v1/ocr/process-file", files={"file": ("wall.png", PNG, "image/png")}
        )
        assert res.status_code == 200, res.text
        assert res.json()["full_text"] == ""


class TestTransientThrottling:
    """OCR.space throttles free keys under load with a 503 (E571) and asks the
    caller to retry. A single attempt turns a momentary wobble into a failed
    upload, so transient statuses get another try — the same pattern
    GeminiClient already uses for the LLM calls."""

    @pytest.mark.asyncio
    async def test_a_503_is_retried_and_can_succeed(self, monkeypatch) -> None:
        import httpx

        from backend.services.ocr_service import OCRSpaceService

        service = OCRSpaceService(api_key="test_key")
        attempts = {"n": 0}
        request = httpx.Request("POST", "https://api.ocr.space/parse/image")

        ok_body = {
            "ParsedResults": [
                {"ParsedText": "Product\tUnits", "FileParseExitCode": 1, "ErrorMessage": ""}
            ],
            "OCRExitCode": 1,
            "IsErroredOnProcessing": False,
            "OCREngineExecuted": "2",
        }

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    return httpx.Response(
                        503, request=request,
                        text='{"error":"E571: Free OCR API overloaded currently"}',
                    )
                return httpx.Response(200, request=request, json=ok_body)

        monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _Client())
        monkeypatch.setattr("backend.services.ocr_service.asyncio.sleep", _no_sleep)

        result = await service.parse_image_bytes(content=b"png", filename="a.png")
        assert attempts["n"] == 2, "the 503 should have been retried"
        assert "Product" in result["full_text"]

    @pytest.mark.asyncio
    async def test_a_persistent_503_reports_the_upstream_reason(self, monkeypatch) -> None:
        import httpx

        from backend.services.ocr_service import OCRSpaceService

        service = OCRSpaceService(api_key="test_key")
        request = httpx.Request("POST", "https://api.ocr.space/parse/image")

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                return httpx.Response(
                    503, request=request,
                    text='{"error":"E571: Free OCR API overloaded currently, so your free ocr api key is throttled."}',
                )

        monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _Client())
        monkeypatch.setattr("backend.services.ocr_service.asyncio.sleep", _no_sleep)

        with pytest.raises(OCRServiceError) as excinfo:
            await service.parse_image_bytes(content=b"png", filename="a.png")
        assert "throttled" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_a_400_is_not_retried(self, monkeypatch) -> None:
        """A rejected file fails identically every time — retrying only spends
        quota and makes the user wait."""
        import httpx

        from backend.services.ocr_service import OCRSpaceService

        service = OCRSpaceService(api_key="test_key")
        attempts = {"n": 0}
        request = httpx.Request("POST", "https://api.ocr.space/parse/image")

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                attempts["n"] += 1
                return httpx.Response(400, request=request, text='{"error":"E216: bad file"}')

        monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _Client())
        monkeypatch.setattr("backend.services.ocr_service.asyncio.sleep", _no_sleep)

        with pytest.raises(OCRServiceError):
            await service.parse_image_bytes(content=b"png", filename="a.png")
        assert attempts["n"] == 1, "a 400 must not be retried"
