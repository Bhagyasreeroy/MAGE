"""
services/dataset_service.py
─────────────────────────────
Persists uploaded datasets to Postgres (Dataset rows) and retrieves them
scoped to the owning user, so a follow-up analysis run can reference the
same dataset by id instead of re-uploading it — and so the Datasets page
can show a real, durable list per account.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.dataset import Dataset

logger = logging.getLogger(__name__)


@dataclass
class StoredFile:
    """Duck-types the parts of FastAPI's UploadFile that the ingestion
    engine actually uses (.filename, .file), so a stored dataset can be
    passed straight back through the same ingestion code path."""

    filename: str
    file: io.BytesIO


async def save_dataset(
    db: AsyncSession,
    user_id: str,
    filename: str,
    content: bytes,
    row_count: int | None = None,
    column_count: int | None = None,
) -> Dataset:
    """Persist an uploaded file as a new Dataset row owned by `user_id`."""
    dataset = Dataset(
        user_id=user_id,
        filename=filename,
        content=content,
        row_count=row_count,
        column_count=column_count,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset


async def update_dataset_stats(
    db: AsyncSession, dataset_id: str, row_count: int | None, column_count: int | None
) -> None:
    """Backfill row_count/column_count once ingestion has profiled the file."""
    dataset = await db.get(Dataset, dataset_id)
    if dataset is not None:
        dataset.row_count = row_count
        dataset.column_count = column_count
        await db.commit()


async def get_dataset(db: AsyncSession, user_id: str, dataset_id: str) -> Dataset | None:
    """Fetch a Dataset by id, scoped to the owning user (never returns
    another user's dataset, even if the id is guessed)."""
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def delete_dataset(db: AsyncSession, user_id: str, dataset_id: str) -> bool:
    """Delete a dataset owned by `user_id`. Returns False if not found/not owned."""
    dataset = await get_dataset(db, user_id, dataset_id)
    if dataset is None:
        return False
    await db.delete(dataset)
    await db.commit()
    return True


async def list_datasets(db: AsyncSession, user_id: str, limit: int = 50) -> list[Dataset]:
    """List a user's datasets, most recent first."""
    result = await db.execute(
        select(Dataset)
        .where(Dataset.user_id == user_id)
        .order_by(Dataset.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


def as_stored_file(dataset: Dataset) -> StoredFile:
    """Wrap a Dataset row's bytes so it can flow through DataIngestionEngine
    exactly like an UploadFile would."""
    return StoredFile(filename=dataset.filename, file=io.BytesIO(dataset.content))
