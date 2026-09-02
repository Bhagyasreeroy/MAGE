"""
routers/ocr.py
───────────────
OCR endpoints using OCR.space API (https://ocr.space/).

Features:
- Parse uploaded file (image or PDF) into extracted text and structured tables.
- Parse image or PDF from a remote HTTP URL.
- Extract table data from an image/PDF using OCR and automatically persist it
  as a dataset in MAGE.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.deps import get_current_user
from backend.models.user import User
from backend.schemas.ocr import (
    OCRPageResult,
    OCRResponse,
    OCRStats,
    OCRToDatasetResponse,
    OCRUrlRequest,
)
from backend.services.dataset_service import save_dataset
from backend.services.ocr_service import OCRServiceError, ocr_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.post(
    "/process-file",
    response_model=OCRResponse,
    summary="Extract text and table data from uploaded image or PDF file via OCR.space",
)
async def process_file(
    file: UploadFile = File(...),
    language: str = Form("eng"),
    is_table: bool = Form(True),
    ocr_engine: int = Form(2),
    current_user: User = Depends(get_current_user),
) -> OCRResponse:
    """
    Upload an image (PNG, JPG, WEBP, BMP, TIFF) or PDF document and run Optical Character Recognition (OCR)
    using the OCR.space API.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        result = await ocr_service.parse_image_bytes(
            content=content,
            filename=file.filename,
            language=language,
            is_table=is_table,
            ocr_engine=ocr_engine,
        )
    except OCRServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return OCRResponse(
        status=result["status"],
        ocr_engine=result["ocr_engine"],
        exit_code=result["exit_code"],
        full_text=result["full_text"],
        pages=[OCRPageResult(**p) for p in result["pages"]],
        table_rows=result["table_rows"],
        stats=OCRStats(**result["stats"]),
    )


@router.post(
    "/process-url",
    response_model=OCRResponse,
    summary="Extract text and table data from remote image or PDF URL via OCR.space",
)
async def process_url(
    payload: OCRUrlRequest,
    current_user: User = Depends(get_current_user),
) -> OCRResponse:
    """
    Pass an HTTP/HTTPS image or PDF URL to perform OCR processing via OCR.space API.
    """
    try:
        result = await ocr_service.parse_image_url(
            url=payload.url,
            language=payload.language,
            is_table=payload.is_table,
            ocr_engine=payload.ocr_engine,
        )
    except OCRServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return OCRResponse(
        status=result["status"],
        ocr_engine=result["ocr_engine"],
        exit_code=result["exit_code"],
        full_text=result["full_text"],
        pages=[OCRPageResult(**p) for p in result["pages"]],
        table_rows=result["table_rows"],
        stats=OCRStats(**result["stats"]),
    )


@router.post(
    "/convert-to-dataset",
    response_model=OCRToDatasetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Run OCR on document/image and persist extracted tabular data directly as a MAGE Dataset",
)
async def convert_to_dataset(
    file: UploadFile = File(...),
    language: str = Form("eng"),
    is_table: bool = Form(True),
    ocr_engine: int = Form(2),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> OCRToDatasetResponse:
    """
    Upload an image or document (scanned table, invoice, receipt, screenshot), extract text/tables
    using OCR.space, convert to CSV, and save directly to MAGE datasets table for analysis.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        result = await ocr_service.parse_image_bytes(
            content=content,
            filename=file.filename,
            language=language,
            is_table=is_table,
            ocr_engine=ocr_engine,
        )
    except OCRServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    csv_content = ocr_service.convert_ocr_to_csv(result)
    csv_bytes = csv_content.encode("utf-8")

    # Generate target CSV filename
    base_name = file.filename.rsplit(".", 1)[0]
    csv_filename = f"{base_name}_ocr.csv"

    # Calculate row count & column count from table_rows
    table_rows = result.get("table_rows", [])
    row_count = len(table_rows) if table_rows else len(result.get("full_text", "").splitlines())
    col_count = max([len(r) for r in table_rows]) if table_rows else 1

    dataset = await save_dataset(
        db=db,
        user_id=current_user.id,
        filename=csv_filename,
        content=csv_bytes,
        row_count=row_count,
        column_count=col_count,
    )

    full_text_preview = result["full_text"][:300] + "..." if len(result["full_text"]) > 300 else result["full_text"]

    return OCRToDatasetResponse(
        dataset_id=dataset.id,
        filename=dataset.filename,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
        message="Document OCR processed and converted to dataset successfully.",
        full_text_preview=full_text_preview,
    )
