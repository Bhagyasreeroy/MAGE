"""
services/ocr_service.py
─────────────────────────
Service layer for Optical Character Recognition (OCR) via OCR.space API (https://ocr.space/).

Supports parsing images (PNG, JPG, WEBP, BMP, TIFF) and PDF documents to extract
plain text, structured table data, and line-by-line output.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import Any

import httpx

from backend.core.config import settings

logger = logging.getLogger(__name__)


class OCRServiceError(Exception):
    """Custom exception raised when OCR processing fails."""
    pass



# OCR.space throttles free API keys when the shared service is busy, answering
# 503 with "E571: Free OCR API overloaded currently ... Please retry in a few
# minutes". It clears on its own, so a couple of quick retries turn a common
# transient failure into a successful upload. A PRO key is never throttled.
OCR_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
OCR_MAX_ATTEMPTS = 3
OCR_RETRY_BACKOFF_SECONDS = 1.5


def _default_ca_bundle() -> str:
    """The CA bundle to verify OCR.space against, or "" for certifi's default.

    httpx does not consult the OS trust store, so on a network that re-signs
    TLS (a corporate proxy, a VPN) every call dies with "self-signed
    certificate in certificate chain" even though a browser on the same
    machine is fine. OCR_CA_BUNDLE names the proxy root CA; the two standard
    env vars are honoured too so a machine already configured for requests or
    the OpenSSL tools needs no extra setup. A path that does not exist is
    ignored rather than raising, since a stale value should not take OCR down.
    """
    for candidate in (
        settings.ocr_ca_bundle,
        os.environ.get("REQUESTS_CA_BUNDLE", ""),
        os.environ.get("SSL_CERT_FILE", ""),
    ):
        if candidate and os.path.exists(candidate):
            return candidate
        if candidate:
            logger.warning("Ignoring CA bundle %r: no such file.", candidate)
    return ""


def _connection_detail(exc: httpx.RequestError) -> str:
    """Explain a failed connection, naming the fix for the TLS-proxy case."""
    message = str(exc)
    if "CERTIFICATE_VERIFY_FAILED" in message or isinstance(exc, httpx.ConnectError) and "certificate" in message:
        return (
            "Could not verify the OCR service's TLS certificate. This usually means a "
            "corporate proxy or VPN is re-signing HTTPS traffic. Export its root "
            "certificate as a PEM file and set OCR_CA_BUNDLE to that path (or "
            f"REQUESTS_CA_BUNDLE), then restart the backend. Details: {message}"
        )
    return f"Network error connecting to OCR API: {message}"



def _dedupe_header(cells: list[str]) -> list[str]:
    """Make a header row usable as CSV column names: non-empty and unique."""
    header: list[str] = []
    seen: dict[str, int] = {}
    for idx, cell in enumerate(cells, start=1):
        name = cell.strip() or f"column_{idx}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        header.append(name if count == 0 else f"{name}_{count + 1}")
    return header


def normalize_table_rows(rows: list[list[str]]) -> list[list[str]]:
    """Square up OCR output so pandas reads it as a real table.

    OCR of a document returns everything on the page, not just the grid: a
    title, a caption, a page number all arrive as one-cell rows alongside the
    table proper. Written out as-is, the *title* becomes the CSV header, so the
    dataset lands with a single column named after the document and every real
    column is lost — which is why such an analysis shows no column stats and no
    charts worth drawing.

    So the header is the first row as wide as the table's most common width,
    anything above it is dropped as preamble, and later short rows (a totals
    line, a footnote) are padded to that width instead of breaking the parse.
    """
    widths: dict[int, int] = {}
    for row in rows:
        if len(row) > 1:
            widths[len(row)] = widths.get(len(row), 0) + 1
    if not widths:
        return [_dedupe_header(row) if i == 0 else row for i, row in enumerate(rows)] if rows else []

    # Most frequent width wins; the wider one breaks a tie, since a split cell
    # is a likelier accident than an invented extra column.
    width = max(widths, key=lambda w: (widths[w], w))

    start = next(i for i, row in enumerate(rows) if len(row) == width)
    body = rows[start:]
    squared = [(row + [""] * width)[:width] for row in body]
    return [_dedupe_header(squared[0])] + squared[1:]


def _upstream_detail(exc: httpx.HTTPStatusError) -> str:
    """Turn an OCR.space error response into something worth showing a user.

    The API explains its refusals in the body ("E556: File too large. Max
    1.5 MB for Free Plan"), and that sentence is the only actionable part of
    the failure. Reporting the status code alone discards it. Falls back to
    the code when the body is empty or not the shape we expect.
    """
    status_code = exc.response.status_code
    try:
        body = exc.response.json()
    except ValueError:
        body = None

    message = ""
    if isinstance(body, dict):
        raw = body.get("error") or body.get("ErrorMessage") or ""
        if isinstance(raw, list):
            raw = "; ".join(str(item) for item in raw)
        message = str(raw).strip()

    if not message:
        text = (exc.response.text or "").strip()
        # A short plain-text body is usually the explanation itself; a long
        # one is an HTML error page, which helps nobody.
        message = text if 0 < len(text) <= 200 else ""

    return f"OCR API error ({status_code}): {message}" if message else (
        f"OCR API returned status code {status_code}"
    )


class OCRSpaceService:
    """Async service interacting with the OCR.space REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        timeout: float = 30.0,
        ca_bundle: str | None = None,
    ):
        self.api_key = api_key or settings.ocr_space_api_key or "helloworld"
        self.api_url = api_url or settings.ocr_space_api_url or "https://api.ocr.space/parse/image"
        self.timeout = timeout
        self.ca_bundle = ca_bundle if ca_bundle is not None else _default_ca_bundle()

    def _verify(self) -> Any:
        """What httpx should verify the OCR.space certificate against."""
        return self.ca_bundle or True

    async def parse_image_bytes(
        self,
        content: bytes,
        filename: str,
        language: str = "eng",
        is_table: bool = True,
        ocr_engine: int = 2,
        scale: bool = True,
    ) -> dict[str, Any]:
        """
        Extract text and tabular data from image/PDF bytes using OCR.space API.

        :param content: Raw file bytes (JPEG, PNG, WEBP, PDF, etc.)
        :param filename: Filename with extension (e.g. sample.png, invoice.pdf)
        :param language: OCR language code (e.g., 'eng', 'fre', 'ger', 'spa')
        :param is_table: If True, enables table recognition (parses tabular layout)
        :param ocr_engine: Engine version: 1 (fast/default) or 2 (better for numbers & tables)
        :param scale: Enables automatic image scaling for better low-res recognition
        :return: Standardized result dict with extracted text, table rows, and page details.
        """
        if not content:
            raise OCRServiceError("Empty file content provided for OCR processing.")

        data = {
            "apikey": self.api_key,
            "language": language,
            "isTable": str(is_table).lower(),
            "scale": str(scale).lower(),
            "OCREngine": str(ocr_engine),
            "detectOrientation": "true",
        }

        files = {
            "file": (filename, content),
        }

        try:
            payload = await self._post_with_retry(data=data, files=files)
        except httpx.HTTPStatusError as exc:
            logger.error("OCR.space API HTTP error %s: %s", exc.response.status_code, exc.response.text)
            raise OCRServiceError(_upstream_detail(exc)) from exc
        except httpx.RequestError as exc:
            logger.error("OCR.space API connection error: %s", exc)
            raise OCRServiceError(_connection_detail(exc)) from exc
        except Exception as exc:
            logger.error("Unexpected error during OCR request: %s", exc)
            raise OCRServiceError(f"OCR processing failed: {exc}") from exc

        return self._format_ocr_response(payload)

    async def parse_image_url(
        self,
        url: str,
        language: str = "eng",
        is_table: bool = True,
        ocr_engine: int = 2,
        scale: bool = True,
    ) -> dict[str, Any]:
        """
        Extract text and tabular data from a publicly accessible image/PDF URL.
        """
        if not url:
            raise OCRServiceError("URL cannot be empty.")

        data = {
            "apikey": self.api_key,
            "url": url,
            "language": language,
            "isTable": str(is_table).lower(),
            "scale": str(scale).lower(),
            "OCREngine": str(ocr_engine),
            "detectOrientation": "true",
        }

        try:
            payload = await self._post_with_retry(data=data)
        except httpx.HTTPStatusError as exc:
            logger.error("OCR.space API HTTP error %s: %s", exc.response.status_code, exc.response.text)
            raise OCRServiceError(_upstream_detail(exc)) from exc
        except Exception as exc:
            logger.error("Failed to process OCR from URL: %s", exc)
            raise OCRServiceError(f"OCR URL processing failed: {exc}") from exc

        return self._format_ocr_response(payload)

    async def _post_with_retry(self, data: dict[str, Any], files: Any = None) -> dict[str, Any]:
        """POST once, retrying only the statuses that clear on their own.

        A 4xx means the file itself was rejected and will be rejected again,
        so retrying it just spends quota and makes the user wait. Overload and
        gateway errors are the opposite: the same request usually succeeds a
        moment later.
        """
        last_exc: httpx.HTTPStatusError | None = None

        for attempt in range(1, OCR_MAX_ATTEMPTS + 1):
            async with httpx.AsyncClient(timeout=self.timeout, verify=self._verify()) as client:
                response = await client.post(self.api_url, data=data, files=files)

            if response.status_code == 200:
                return response.json()

            exc = httpx.HTTPStatusError(
                f"HTTP {response.status_code}", request=response.request, response=response
            )
            if response.status_code not in OCR_RETRY_STATUSES:
                raise exc

            last_exc = exc
            if attempt < OCR_MAX_ATTEMPTS:
                logger.warning(
                    "OCR.space attempt %d/%d returned %s; retrying.",
                    attempt, OCR_MAX_ATTEMPTS, response.status_code,
                )
                await asyncio.sleep(OCR_RETRY_BACKOFF_SECONDS * attempt)

        assert last_exc is not None
        raise last_exc

    def _format_ocr_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Parse and sanitize the raw JSON response from OCR.space.
        """
        if payload.get("IsErroredOnProcessing"):
            error_details = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "Unknown OCR API error"
            if isinstance(error_details, list):
                error_details = "; ".join(error_details)
            raise OCRServiceError(f"OCR API error: {error_details}")

        parsed_results = payload.get("ParsedResults") or []
        pages = []
        full_text_parts = []
        table_rows = []

        for idx, item in enumerate(parsed_results, start=1):
            parsed_text = item.get("ParsedText", "").strip()
            exit_code = item.get("FileParseExitCode", 0)
            error_msg = item.get("ErrorMessage", "")

            if parsed_text:
                full_text_parts.append(parsed_text)

            # Process line-by-line text / tables
            lines = [line.strip() for line in parsed_text.splitlines() if line.strip()]
            for line in lines:
                # Tab-separated or multi-space separated values indicative of tables
                cols = [c.strip() for c in line.replace("\t", "  ").split("  ") if c.strip()]
                if cols:
                    table_rows.append(cols)

            pages.append({
                "page_number": idx,
                "parsed_text": parsed_text,
                "lines": lines,
                "exit_code": exit_code,
                "error_message": error_msg,
            })

        full_text = "\n\n".join(full_text_parts)
        words = full_text.split()

        return {
            "status": "success",
            "ocr_engine": payload.get("OCREngineExecuted", "2"),
            "exit_code": payload.get("OCRExitCode"),
            "full_text": full_text,
            "pages": pages,
            "table_rows": table_rows,
            "stats": {
                "page_count": len(pages),
                "word_count": len(words),
                "character_count": len(full_text),
            },
        }

    def convert_ocr_to_csv(self, ocr_result: dict[str, Any]) -> str:
        """
        Convert OCR extracted table rows or lines into CSV string format for dataset ingestion.
        """
        table_rows = normalize_table_rows(ocr_result.get("table_rows", []))
        if not table_rows:
            full_text = ocr_result.get("full_text", "")
            lines = [line.strip() for line in full_text.splitlines() if line.strip()]
            table_rows = [["text"]] + [[line] for line in lines]

        output = io.StringIO()
        import csv
        writer = csv.writer(output)
        for row in table_rows:
            writer.writerow(row)
        return output.getvalue()


# Default singleton service instance
ocr_service = OCRSpaceService()
