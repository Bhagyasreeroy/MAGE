"""
services/ocr_service.py
─────────────────────────
Service layer for Optical Character Recognition (OCR) via OCR.space API (https://ocr.space/).

Supports parsing images (PNG, JPG, WEBP, BMP, TIFF) and PDF documents to extract
plain text, structured table data, and line-by-line output.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import httpx

from backend.core.config import settings

logger = logging.getLogger(__name__)


class OCRServiceError(Exception):
    """Custom exception raised when OCR processing fails."""
    pass


class OCRSpaceService:
    """Async service interacting with the OCR.space REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or settings.ocr_space_api_key or "helloworld"
        self.api_url = api_url or settings.ocr_space_api_url or "https://api.ocr.space/parse/image"
        self.timeout = timeout

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
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, data=data, files=files)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("OCR.space API HTTP error %s: %s", exc.response.status_code, exc.response.text)
            raise OCRServiceError(f"OCR API returned status code {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            logger.error("OCR.space API connection error: %s", exc)
            raise OCRServiceError(f"Network error connecting to OCR API: {exc}") from exc
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
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, data=data)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            logger.error("Failed to process OCR from URL: %s", exc)
            raise OCRServiceError(f"OCR URL processing failed: {exc}") from exc

        return self._format_ocr_response(payload)

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
        table_rows = ocr_result.get("table_rows", [])
        if not table_rows:
            full_text = ocr_result.get("full_text", "")
            lines = [line.strip() for line in full_text.splitlines() if line.strip()]
            table_rows = [[line] for line in lines]

        output = io.StringIO()
        import csv
        writer = csv.writer(output)
        for row in table_rows:
            writer.writerow(row)
        return output.getvalue()


# Default singleton service instance
ocr_service = OCRSpaceService()
