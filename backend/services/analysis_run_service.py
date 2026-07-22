"""
services/analysis_run_service.py
───────────────────────────────────
Persists completed pipeline runs and lists/retrieves them per user, so a
user's dashboard/history survives across logins and devices instead of
living only in browser sessionStorage.
"""

from __future__ import annotations

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
) -> AnalysisRun:
    status = "error" if any(s.get("status") == "error" for s in steps) else "success"
    run = AnalysisRun(
        user_id=user_id,
        dataset_id=dataset_id,
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


async def list_runs(db: AsyncSession, user_id: str, limit: int = 50) -> list[AnalysisRun]:
    """List a user's analysis runs, most recent first."""
    result = await db.execute(
        select(AnalysisRun)
        .where(AnalysisRun.user_id == user_id)
        .order_by(AnalysisRun.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
