"""
models/dataset.py
──────────────────
SQLAlchemy ORM model for uploaded datasets.

Stores the raw file bytes directly in Postgres (LargeBinary). Simple and
sufficient at this project's scale — an object store (S3/MinIO) is the
natural upgrade if file sizes or volume grow, per the architecture notes
in docs/architecture.md.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class Dataset(Base):
    """A user-uploaded dataset file, referenced by id for re-ingestion
    across follow-up analysis runs without re-uploading."""

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    row_count: Mapped[int | None] = mapped_column(nullable=True)
    column_count: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
