"""
tests/backend/test_run_memory.py
─────────────────────────────────
Tests for the run-memory store (Objective 4's second half).

The proposal claims recommendations are grounded in *both* a curated
knowledge base **and** the system's memory of prior runs — the mechanism
behind "improves with use". Only the knowledge base existed; the ER diagram's
`run_memory` table was never built, so the benefit was unrealised.

This store records what each completed run was about and what it found, then
retrieves the most semantically similar prior runs when a new goal arrives, so
recommendations can reference what the same user previously learned from
comparable data.

Two properties matter more than the retrieval quality itself:

  • **Per-user isolation.** Run memory is derived from a user's data and their
    questions. Leaking it across accounts would be worse than not having it,
    so the isolation tests here are the ones to keep if any are dropped.
  • **Never fatal.** Memory is additive grounding. A failure to record or
    retrieve it must degrade the recommendation, not the run.

Written test-first: none of this existed when these were written.

Requires PostgreSQL, like the rest of `tests/backend/`.
"""

from __future__ import annotations

import os
import sys
import uuid

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.core.database import async_session
from backend.models.user import User
from backend.services import run_memory_service


def _unique_email() -> str:
    return f"memory-{uuid.uuid4().hex[:12]}@example.com"


@pytest_asyncio.fixture(autouse=True)
async def _tables():
    """
    Ensure the schema exists.

    The rest of `tests/backend/` drives the app through TestClient, whose
    lifespan calls `init_db()`. These tests talk to the session directly, so
    nothing would create `run_memory` for them. `create_all` is a no-op when
    the tables are already present.
    """
    import backend.models  # noqa: F401 - registers every model on Base.metadata
    from backend.core.database import init_db

    await init_db()


@pytest_asyncio.fixture
async def db():
    async with async_session() as session:
        yield session


@pytest_asyncio.fixture
async def user(db) -> User:
    record = User(
        email=_unique_email(),
        full_name="Memory Test",
        hashed_password="x",
        auth_provider="local",
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    yield record
    await db.delete(record)
    await db.commit()


@pytest_asyncio.fixture
async def other_user(db) -> User:
    record = User(
        email=_unique_email(),
        full_name="Other User",
        hashed_password="x",
        auth_provider="local",
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    yield record
    await db.delete(record)
    await db.commit()


@pytest.mark.asyncio
class TestRecordingAMemory:
    async def test_a_memory_is_persisted(self, db, user: User) -> None:
        memory = await run_memory_service.record(
            db,
            user_id=user.id,
            goal="Find customer segments in the orders data",
            task_type="clustering",
            dataset_fingerprint="orders:8x400",
            findings=["Data separates into 3 clusters (silhouette=0.51)."],
        )
        assert memory.id
        assert memory.goal.startswith("Find customer segments")

    async def test_the_goal_embedding_is_stored(self, db, user: User) -> None:
        """Similarity is computed from this, so an absent embedding is silent failure."""
        memory = await run_memory_service.record(
            db, user_id=user.id, goal="Detect fraud", task_type="anomaly_detection",
            dataset_fingerprint="tx:5x100", findings=[],
        )
        assert isinstance(memory.goal_embedding, list)
        assert len(memory.goal_embedding) > 0

    async def test_findings_are_retained(self, db, user: User) -> None:
        memory = await run_memory_service.record(
            db, user_id=user.id, goal="Profile the data", task_type="reporting",
            dataset_fingerprint="d:3x50", findings=["'revenue' has 4 outliers."],
        )
        assert memory.findings == ["'revenue' has 4 outliers."]

    async def test_findings_are_capped(self, db, user: User) -> None:
        """A run can emit dozens of patterns; memory keeps the leading few."""
        memory = await run_memory_service.record(
            db, user_id=user.id, goal="Profile", task_type="reporting",
            dataset_fingerprint="d:3x50", findings=[f"finding {i}" for i in range(50)],
        )
        assert len(memory.findings) <= run_memory_service.MAX_STORED_FINDINGS


@pytest.mark.asyncio
class TestRetrievingSimilarRuns:
    async def test_no_history_returns_nothing(self, db, user: User) -> None:
        assert await run_memory_service.retrieve_similar(
            db, user_id=user.id, goal="anything at all"
        ) == []

    async def test_a_semantically_similar_prior_run_is_returned(
        self, db, user: User
    ) -> None:
        await run_memory_service.record(
            db, user_id=user.id, goal="Segment customers into groups",
            task_type="clustering", dataset_fingerprint="a:5x100",
            findings=["Found 3 clusters."],
        )
        similar = await run_memory_service.retrieve_similar(
            db, user_id=user.id, goal="Find natural customer segments"
        )
        assert similar
        assert "Segment customers" in similar[0]["goal"]

    async def test_the_more_similar_run_ranks_first(self, db, user: User) -> None:
        await run_memory_service.record(
            db, user_id=user.id, goal="Detect fraudulent transactions",
            task_type="anomaly_detection", dataset_fingerprint="a:5x100", findings=[],
        )
        await run_memory_service.record(
            db, user_id=user.id, goal="Group customers into market segments",
            task_type="clustering", dataset_fingerprint="b:5x100", findings=[],
        )
        similar = await run_memory_service.retrieve_similar(
            db, user_id=user.id, goal="Find customer segments in this data"
        )
        assert "segments" in similar[0]["goal"].lower()

    async def test_results_are_capped_by_top_k(self, db, user: User) -> None:
        for i in range(6):
            await run_memory_service.record(
                db, user_id=user.id, goal=f"Cluster the data run {i}",
                task_type="clustering", dataset_fingerprint=f"d{i}:5x100", findings=[],
            )
        similar = await run_memory_service.retrieve_similar(
            db, user_id=user.id, goal="Cluster the data", top_k=2
        )
        assert len(similar) == 2

    async def test_a_similarity_score_is_reported(self, db, user: User) -> None:
        await run_memory_service.record(
            db, user_id=user.id, goal="Cluster the customers", task_type="clustering",
            dataset_fingerprint="a:5x100", findings=[],
        )
        similar = await run_memory_service.retrieve_similar(
            db, user_id=user.id, goal="Cluster the customers"
        )
        assert 0.0 <= similar[0]["similarity"] <= 1.0

    async def test_unrelated_prior_runs_are_filtered_out(self, db, user: User) -> None:
        """Weak matches are worse than none — they ground advice in the irrelevant."""
        await run_memory_service.record(
            db, user_id=user.id, goal="Analyse rainfall seasonality in weather stations",
            task_type="reporting", dataset_fingerprint="w:4x80", findings=[],
        )
        similar = await run_memory_service.retrieve_similar(
            db, user_id=user.id, goal="Detect fraudulent credit card transactions"
        )
        assert similar == []


@pytest.mark.asyncio
class TestPerUserIsolation:
    """Run memory is derived from private data. These are the critical tests."""

    async def test_another_users_memory_is_never_returned(
        self, db, user: User, other_user: User
    ) -> None:
        await run_memory_service.record(
            db, user_id=other_user.id, goal="Segment our confidential customer base",
            task_type="clustering", dataset_fingerprint="secret:9x900",
            findings=["Confidential: 3 clusters found."],
        )
        similar = await run_memory_service.retrieve_similar(
            db, user_id=user.id, goal="Segment the customer base"
        )
        assert similar == []

    async def test_each_user_sees_only_their_own(
        self, db, user: User, other_user: User
    ) -> None:
        await run_memory_service.record(
            db, user_id=user.id, goal="Cluster my data", task_type="clustering",
            dataset_fingerprint="mine:5x100", findings=["mine"],
        )
        await run_memory_service.record(
            db, user_id=other_user.id, goal="Cluster my data", task_type="clustering",
            dataset_fingerprint="theirs:5x100", findings=["theirs"],
        )
        mine = await run_memory_service.retrieve_similar(
            db, user_id=user.id, goal="Cluster my data"
        )
        assert len(mine) == 1
        assert mine[0]["findings"] == ["mine"]


@pytest.mark.asyncio
class TestMemoryIsNeverFatal:
    """Additive grounding must not be able to take a run down with it."""

    async def test_recording_with_an_empty_goal_does_not_raise(
        self, db, user: User
    ) -> None:
        assert await run_memory_service.record(
            db, user_id=user.id, goal="", task_type="reporting",
            dataset_fingerprint="d:1x1", findings=[],
        ) is None

    async def test_retrieving_with_an_empty_goal_returns_nothing(
        self, db, user: User
    ) -> None:
        assert await run_memory_service.retrieve_similar(
            db, user_id=user.id, goal=""
        ) == []

    async def test_a_memory_with_no_embedding_is_skipped_not_fatal(
        self, db, user: User
    ) -> None:
        """Rows written before embeddings existed, or by a failed embed."""
        from backend.models.run_memory import RunMemory

        db.add(RunMemory(
            user_id=user.id, goal="Legacy row", task_type="reporting",
            dataset_fingerprint="d:1x1", findings=[], goal_embedding=[],
        ))
        await db.commit()
        assert await run_memory_service.retrieve_similar(
            db, user_id=user.id, goal="Legacy row"
        ) == []


class TestFingerprinting:
    """Identifies comparable datasets without storing anything about their contents."""

    def test_same_shape_and_columns_fingerprint_identically(self) -> None:
        a = run_memory_service.dataset_fingerprint(["a", "b"], 100)
        b = run_memory_service.dataset_fingerprint(["a", "b"], 100)
        assert a == b

    def test_column_order_does_not_matter(self) -> None:
        assert run_memory_service.dataset_fingerprint(
            ["b", "a"], 100
        ) == run_memory_service.dataset_fingerprint(["a", "b"], 100)

    def test_different_columns_fingerprint_differently(self) -> None:
        assert run_memory_service.dataset_fingerprint(
            ["a", "b"], 100
        ) != run_memory_service.dataset_fingerprint(["a", "c"], 100)

    def test_different_row_counts_fingerprint_differently(self) -> None:
        assert run_memory_service.dataset_fingerprint(
            ["a"], 100
        ) != run_memory_service.dataset_fingerprint(["a"], 200)

    def test_no_column_names_survive_into_the_fingerprint(self) -> None:
        """It is a hash, not a record of the user's schema."""
        fingerprint = run_memory_service.dataset_fingerprint(["revenue", "churn"], 100)
        assert "revenue" not in fingerprint
        assert "churn" not in fingerprint

    def test_empty_columns_are_handled(self) -> None:
        assert run_memory_service.dataset_fingerprint([], 0)
