"""
services/analysis_run_service.py
───────────────────────────────────
Persists completed pipeline runs and lists/retrieves them per user, so a
user's dashboard/history survives across logins and devices instead of
living only in browser sessionStorage.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.analysis_run import AnalysisRun


async def save_run(
    db: AsyncSession,
    user_id: str,
    goal: str,
    expertise_level: str,
    steps: list[dict[str, Any]],
    recommendations: list[str],
    rag_sources: list[str],
    summary: str,
    dataset_id: str | None,
    root_run_id: str | None = None,
) -> AnalysisRun:
    """
    Persist a completed run.

    `root_run_id`, when given, is the conversation this run continues — a
    follow-up question. Omitted (the common case: a fresh analysis), this
    run starts its own conversation and is its own root.
    """
    status = "error" if any(s.get("status") == "error" for s in steps) else "success"
    new_id = str(uuid.uuid4())
    run = AnalysisRun(
        id=new_id,
        user_id=user_id,
        dataset_id=dataset_id,
        root_run_id=root_run_id or new_id,
        goal=goal,
        expertise_level=expertise_level,
        status=status,
        summary=summary,
        steps=steps,
        recommendations=recommendations,
        rag_sources=rag_sources,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


async def get_run(db: AsyncSession, user_id: str, run_id: str) -> AnalysisRun | None:
    """Fetch a run by id, scoped to the owning user."""
    result = await db.execute(
        select(AnalysisRun).where(AnalysisRun.id == run_id, AnalysisRun.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def get_run_unscoped(db: AsyncSession, run_id: str) -> AnalysisRun | None:
    """Fetch a run by id with no ownership check — for the public share endpoint only."""
    result = await db.execute(select(AnalysisRun).where(AnalysisRun.id == run_id))
    return result.scalar_one_or_none()


async def resolve_root(db: AsyncSession, user_id: str, run_id: str) -> AnalysisRun | None:
    """
    Owner-scoped: fetch `run_id`, then follow it to its conversation's root
    row. Returns the root run, or None if `run_id` doesn't belong to
    `user_id`. A conversation's share state lives only on its root row, so
    every share/unshare/status/thread lookup goes through this.
    """
    run = await get_run(db, user_id, run_id)
    if run is None:
        return None
    if run.id == run.root_run_id:
        return run
    return await get_run(db, user_id, run.root_run_id)


async def list_thread(db: AsyncSession, root_run_id: str) -> list[AnalysisRun]:
    """
    Every run in a conversation, oldest first. No ownership filter — callers
    must have already authorized access to `root_run_id` (via `resolve_root`
    for an owner, or an `is_shared` check for the public endpoint).
    """
    result = await db.execute(
        select(AnalysisRun)
        .where(AnalysisRun.root_run_id == root_run_id)
        .order_by(AnalysisRun.created_at.asc())
    )
    return list(result.scalars().all())


async def list_runs(db: AsyncSession, user_id: str, limit: int = 50) -> list[AnalysisRun]:
    """List a user's analysis runs, most recent first."""
    result = await db.execute(
        select(AnalysisRun)
        .where(AnalysisRun.user_id == user_id)
        .order_by(AnalysisRun.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
