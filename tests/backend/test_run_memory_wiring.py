"""
tests/backend/test_run_memory_wiring.py
────────────────────────────────────────
End-to-end tests that run memory is actually recorded and reused.

The store and the agent-side plumbing are covered separately; this checks the
loop closes over HTTP — a completed analysis leaves a memory behind, and a
later similar analysis by the same user gets it back.

Written test-first. Requires PostgreSQL.
"""

from __future__ import annotations

import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app

client = TestClient(app)

SAMPLE_CSV = (
    b"units,unit_price,revenue\n"
    + b"".join(f"{i % 9 + 1},{10 + i % 7},{(i % 9 + 1) * (10 + i % 7)}\n".encode()
              for i in range(60))
)


def _auth() -> dict[str, str]:
    email = f"wire-{uuid.uuid4().hex[:12]}@example.com"
    client.post(
        "/auth/register",
        json={"email": email, "full_name": "Wiring Test", "password": "Passw0rd!"},
    )
    token = client.post(
        "/auth/login", json={"email": email, "password": "Passw0rd!"}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _run(headers: dict[str, str], goal: str, dataset_id: str | None = None) -> dict:
    data = {"goal": goal, "expertise_level": "intermediate"}
    files = None
    if dataset_id:
        data["dataset_id"] = dataset_id
    else:
        files = {"file": ("d.csv", SAMPLE_CSV, "text/csv")}
    response = client.post("/analysis/run", headers=headers, data=data, files=files)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def headers() -> dict[str, str]:
    return _auth()


class TestMemoryIsRecorded:
    def test_a_completed_run_leaves_a_memory(self, headers: dict[str, str]) -> None:
        _run(headers, "Find natural clusters and segments in this data")

        import asyncio

        from backend.core.database import async_session
        from backend.services import run_memory_service

        async def check() -> list[dict]:
            async with async_session() as db:
                me = client.get("/auth/me", headers=headers).json()
                return await run_memory_service.retrieve_similar(
                    db, user_id=me["id"], goal="Find clusters and segments"
                )

        assert asyncio.run(check())


class TestMemoryIsReused:
    def test_a_later_similar_run_receives_prior_context(
        self, headers: dict[str, str]
    ) -> None:
        first = _run(headers, "Segment the customers into behavioural groups")
        second = _run(
            headers,
            "Find natural customer segments in this data",
            dataset_id=first["dataset_id"],
        )

        recommendation_step = next(
            s for s in second["steps"] if s["agent_name"] == "RecommendationAgent"
        )
        assert recommendation_step["output"]["prior_runs"], "no prior run was supplied"

    def test_an_unrelated_later_run_gets_no_spurious_context(
        self, headers: dict[str, str]
    ) -> None:
        """A weak match asserts the user learned something they did not."""
        fresh = _auth()
        _run(fresh, "Analyse rainfall seasonality across weather stations")
        second = _run(fresh, "Detect fraudulent credit card transactions")

        recommendation_step = next(
            s for s in second["steps"] if s["agent_name"] == "RecommendationAgent"
        )
        assert recommendation_step["output"]["prior_runs"] == []


class TestMemoryIsPerUser:
    def test_one_users_runs_never_reach_another(self) -> None:
        first_user = _auth()
        _run(first_user, "Segment our confidential customer base into groups")

        second_user = _auth()
        result = _run(second_user, "Segment the customer base into groups")

        recommendation_step = next(
            s for s in result["steps"] if s["agent_name"] == "RecommendationAgent"
        )
        assert recommendation_step["output"]["prior_runs"] == []


class TestCitationIntegrityOverHttp:
    def test_recommendations_still_cite_only_knowledge_base_documents(
        self, headers: dict[str, str]
    ) -> None:
        result = _run(headers, "Find natural clusters and segments in this data")
        assert result["rag_sources"]
        assert all(source.endswith(".md") for source in result["rag_sources"])
