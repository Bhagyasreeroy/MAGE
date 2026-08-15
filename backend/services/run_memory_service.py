"""
services/run_memory_service.py
───────────────────────────────
Record and retrieve prior-run memory (Objective 4).

Recommendations are grounded in a curated knowledge base; this adds the second
source the proposal specifies — the user's own analytical history. After a run
completes, what it was about and what it found is recorded. When a new goal
arrives, semantically similar prior runs are retrieved and offered as extra
grounding, which is the mechanism behind "improves with use".

Three constraints shape everything here:

* **Per-user isolation.** Every query is scoped to one ``user_id``. Run memory
  is derived from private data and private questions; leaking it across
  accounts would be worse than not having the feature.
* **Never fatal.** Memory is additive. Every entry point swallows its own
  failures and returns an empty result, because a recommendation grounded in
  the knowledge base alone is a degraded outcome, while a failed analysis is a
  broken one.
* **Derived data only.** The fingerprint is a hash; no column names or values
  are stored.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.run_memory import RunMemory

logger = logging.getLogger(__name__)

# Findings kept per memory. A run can emit dozens of patterns; the leading few
# carry the headline result, and storing all of them would make retrieved
# memory longer than the recommendation it is meant to support.
MAX_STORED_FINDINGS = 5

# Prior runs returned per query, unless the caller asks for fewer.
DEFAULT_TOP_K = 3

# Cosine-similarity floor for a prior run to count as relevant. Set well above
# the knowledge base's 0.15 because the failure mode differs: an unrelated
# methodology document is merely unhelpful, whereas an unrelated *prior run*
# asserts that the user previously learned something about this question when
# they did not. Weak matches are dropped rather than surfaced.
MIN_SIMILARITY = 0.45

# Rows scanned per query. Similarity is computed in Python, so this bounds the
# work; a user's recent history is what is relevant in any case.
_MAX_CANDIDATES = 200


def dataset_fingerprint(columns: list[str], row_count: int) -> str:
    """
    Identify comparable datasets without recording anything about them.

    Hashes the sorted column names together with the row count, so the same
    table recognises itself across runs while the schema itself never lands in
    the database. Order-insensitive, since column order is not a property of
    the data.
    """
    payload = "|".join(sorted(str(c) for c in columns)) + f"#{int(row_count)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _embed(text: str) -> list[float]:
    """Embed goal text, returning ``[]` if embedding is unavailable."""
    try:
        from rag.embeddings import embed_text

        return [float(x) for x in embed_text(text)]
    except Exception:  # noqa: BLE001 - memory must never break a run
        logger.warning("Could not embed goal for run memory.", exc_info=True)
        return []


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, clamped to [0, 1]; 0.0 for a degenerate vector."""
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / denominator)))


async def record(
    db: AsyncSession,
    user_id: str,
    goal: str,
    task_type: str,
    dataset_fingerprint: str,
    findings: list[str],
) -> RunMemory | None:
    """
    Persist one completed run as memory.

    Returns the stored row, or ``None`` when there is nothing worth
    remembering (an empty goal) or the write failed. Callers are expected to
    ignore the return value — it exists for tests and for logging.
    """
    if not goal or not goal.strip():
        return None

    embedding = _embed(goal)
    memory = RunMemory(
        user_id=user_id,
        goal=goal.strip(),
        task_type=task_type or "",
        dataset_fingerprint=dataset_fingerprint or "",
        findings=[str(f) for f in (findings or [])][:MAX_STORED_FINDINGS],
        goal_embedding=embedding,
    )
    try:
        db.add(memory)
        await db.commit()
        await db.refresh(memory)
    except Exception:  # noqa: BLE001 - additive grounding, never fatal
        logger.warning("Could not record run memory.", exc_info=True)
        await db.rollback()
        return None
    return memory


async def retrieve_similar(
    db: AsyncSession,
    user_id: str,
    goal: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Return this user's most semantically similar prior runs.

    Ranked by cosine similarity between goal embeddings, filtered by
    :data:`MIN_SIMILARITY`, and capped at ``top_k``. Returns ``[]`` on any
    failure, on an empty goal, and when nothing clears the floor — an empty
    result simply means the recommendation is grounded in the knowledge base
    alone, which is the behaviour that existed before this store.

    Records written without an embedding are skipped rather than treated as
    dissimilar, so a failed embed degrades one memory instead of poisoning the
    ranking.
    """
    if not goal or not goal.strip():
        return []

    query_embedding = _embed(goal)
    if not query_embedding:
        return []

    try:
        result = await db.execute(
            select(RunMemory)
            .where(RunMemory.user_id == user_id)
            .order_by(RunMemory.created_at.desc())
            .limit(_MAX_CANDIDATES)
        )
        candidates = list(result.scalars().all())
    except Exception:  # noqa: BLE001
        logger.warning("Could not read run memory.", exc_info=True)
        return []

    query_vector = np.asarray(query_embedding, dtype=float)
    scored: list[tuple[float, RunMemory]] = []
    for memory in candidates:
        embedding = memory.goal_embedding or []
        if len(embedding) != len(query_embedding):
            continue  # absent or incompatible embedding — skip, do not rank
        scored.append((_cosine(query_vector, np.asarray(embedding, dtype=float)), memory))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "goal": memory.goal,
            "task_type": memory.task_type,
            "findings": list(memory.findings or []),
            "similarity": round(similarity, 4),
            # ISO string, not a datetime: this dict travels into the agent
            # context, out through the step log, and is persisted as JSON.
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
        }
        for similarity, memory in scored[:top_k]
        if similarity >= MIN_SIMILARITY
    ]
