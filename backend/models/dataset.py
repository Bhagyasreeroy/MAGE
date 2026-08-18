"""
models/dataset.py
──────────────────
SQLAlchemy ORM model for uploaded datasets.

Stores the raw file bytes directly in Postgres (LargeBinary). Simple and
sufficient at this project's scale — an object store (S3/MinIO) is the
natural upgrade if file sizes or volume grow, per the architecture notes
in docs/architecture.md.

Versioning: a dataset transformation (cleaning ops, cell edits, a saved
query) never mutates a row in place — it produces a new Dataset row linked
back to the version it was derived from. root_id/parent_id/version turn
"list every version of this dataset" into a single indexed query instead
of a recursive walk; transform_type/transform_params on a child row *is*
its audit-trail entry, so there's no separate transform-log table.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


def _retention_days() -> int:
    """Read at row-creation time so the TTL stays configurable per deployment."""
    from backend.core.config import settings

    return int(getattr(settings, "dataset_retention_days", 30))


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
    # NFR-04 — uploaded bytes are not kept forever. Nullable so that rows
    # predating retention (and any row a future feature wants to pin) are simply
    # never collected, rather than being swept the moment the feature ships.
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc) + timedelta(days=_retention_days()),
        nullable=True,
        index=True,
    )

    # ── Version lineage ──────────────────────────────────────────────────
    # A fresh upload sets root_id = its own id, version = 1. Every
    # transform-produced child copies the root's root_id, points parent_id
    # at the version it was derived from, and increments version.
    root_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    version: Mapped[int] = mapped_column(nullable=False, default=1)
    # "clean" | "query_save" | None (an original upload, not a transform).
    transform_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # {"ops": [...]} for "clean", {"sql": "..."} for "query_save".
    transform_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
