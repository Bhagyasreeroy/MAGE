"""
schemas/ocr.py
──────────────
Pydantic schemas for OCR input validation and API responses.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field, HttpUrl


class OCRUrlRequest(BaseModel):
    """Request model for processing an image or PDF from a remote URL."""
    url: str = Field(..., description="Publicly accessible HTTP/HTTPS URL of the image or PDF file")
    language: str = Field(default="eng", description="OCR language code (e.g. 'eng', 'fre', 'ger', 'spa')")
    is_table: bool = Field(default=True, description="Enable structured table extraction")
    ocr_engine: int = Field(default=2, ge=1, le=2, description="OCR Engine version (1 or 2)")


class OCRPageResult(BaseModel):
    """Parsed output details for a single page."""
    page_number: int
    parsed_text: str
    lines: List[str]
    exit_code: int
    error_message: Optional[str] = ""


class OCRStats(BaseModel):
    """Statistics on extracted text."""
    page_count: int
    word_count: int
    character_count: int


class OCRResponse(BaseModel):
    """Response returned after running OCR on a document or image."""
    status: str
    ocr_engine: str
    exit_code: Optional[int] = 1
    full_text: str
    pages: List[OCRPageResult]
    table_rows: List[List[str]]
    stats: OCRStats


class OCRToDatasetResponse(BaseModel):
    """Response returned when saving OCR extracted text directly as a MAGE Dataset."""
    dataset_id: str
    filename: str
    row_count: Optional[int]
    column_count: Optional[int]
    message: str
    full_text_preview: str
