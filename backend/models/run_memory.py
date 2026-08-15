"""
models/run_memory.py
─────────────────────
SQLAlchemy ORM model for the run-memory store.

This is the table the proposal's ER diagram specifies and the system did not
have. It is what makes the "improves with use" half of Objective 4 real: each
completed run records what it was about and what it found, so a later run on a
comparable question can be grounded in the user's own analytical history
alongside the curated knowledge base.

Deliberately stores *derived* material only — the goal text, the task type, a
hashed dataset fingerprint, and the headline findings. No column names, no
cell values, and no dataset contents, so a memory cannot become a second
uncontrolled copy of a user's data.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class RunMemory(Base):
    """One remembered analysis run, owned by a user."""

    __tablename__ = "run_memory"

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
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(String(40), nullable=False, default="")
    # Hash of (sorted column names, row count). Lets two runs be recognised as
    # being about comparable data without recording anything about that data.
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    # Headline patterns from the run, capped — see MAX_STORED_FINDINGS.
    findings: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    # Embedding of the goal text. Similarity is computed over this in Python
    # rather than in the database: the corpus here is one user's run history,
    # which is small enough that a vector index would be more machinery than
    # the problem needs.
    goal_embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
