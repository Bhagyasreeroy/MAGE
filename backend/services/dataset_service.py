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
import uuid
from dataclasses import dataclass
from typing import Any

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
    parent_id: str | None = None,
    transform_type: str | None = None,
    transform_params: dict[str, Any] | None = None,
) -> Dataset:
    """Persist a Dataset row owned by `user_id`.

    With no `parent_id`, this is a fresh upload: it becomes its own lineage
    root (`root_id = id`, `version = 1`). With a `parent_id`, this is a
    transform-produced version: it inherits the parent's `root_id` and gets
    `version = parent.version + 1` — so "list every version of this
    dataset" is a single `WHERE root_id = ...` query, never a recursive walk.
    """
    new_id = str(uuid.uuid4())
    if parent_id is not None:
        parent = await db.get(Dataset, parent_id)
        if parent is None:
            raise ValueError(f"Parent dataset {parent_id!r} not found.")
        root_id = parent.root_id
        version = parent.version + 1
    else:
        root_id = new_id
        version = 1

    dataset = Dataset(
        id=new_id,
        user_id=user_id,
        filename=filename,
        content=content,
        row_count=row_count,
        column_count=column_count,
        root_id=root_id,
        parent_id=parent_id,
        version=version,
        transform_type=transform_type,
        transform_params=transform_params,
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


async def list_versions(db: AsyncSession, user_id: str, root_id: str) -> list[Dataset]:
    """All versions sharing a lineage root, oldest first — the full
    transform history of a dataset (each row's transform_type/params is
    its own audit-trail entry, so no separate log table is needed)."""
    result = await db.execute(
        select(Dataset)
        .where(Dataset.root_id == root_id, Dataset.user_id == user_id)
        .order_by(Dataset.version.asc())
    )
    return list(result.scalars().all())


def as_stored_file(dataset: Dataset) -> StoredFile:
    """Wrap a Dataset row's bytes so it can flow through DataIngestionEngine
    exactly like an UploadFile would."""
    return StoredFile(filename=dataset.filename, file=io.BytesIO(dataset.content))
