"""
models/analysis_run.py
────────────────────────
SQLAlchemy ORM model for a persisted analysis pipeline run.

Stores the full AnalysisResponse payload (steps, recommendations,
rag_sources) as JSON so a user's dashboard/history can reload and re-render
a past run exactly as it was returned, without re-running the pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class AnalysisRun(Base):
    """A single goal-conditioned EDA pipeline run, owned by a user."""

    __tablename__ = "analysis_runs"

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
    dataset_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Self-referential: equals `id` for a run that starts a fresh
    # conversation, or the conversation's original root `id` for a
    # follow-up. "A conversation" = every row sharing a `root_run_id`,
    # ordered by `created_at`. `is_shared` only ever matters on the root
    # row — sharing applies to the whole conversation, not one turn.
    root_run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    is_shared: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    expertise_level: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="success")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    recommendations: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    rag_sources: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
