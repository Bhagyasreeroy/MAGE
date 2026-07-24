"""
tests/backend/test_analysis_persistence.py
──────────────────────────────────────────────
Integration tests for auth-protected analysis endpoints: dataset/run
persistence, history listing, and dataset reuse across requests.

Runs against the configured DATABASE_URL (Postgres); each test creates
its own user and deletes it (cascade) when done.
"""

from __future__ import annotations

import os
import sys
import uuid

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app

client = TestClient(app)

SAMPLE_CSV = b"order_id,region,revenue\n1,East,45.0\n2,West,120.5\n3,East,75.0\n4,North,60.0\n"


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _auth_headers() -> tuple[dict[str, str], str]:
    email = _unique_email()
    client.post("/auth/register", json={"email": email, "full_name": "Test", "password": "Passw0rd!"})
    token = client.post("/auth/login", json={"email": email, "password": "Passw0rd!"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}, email


class TestRunRequiresAuth:
    def test_run_without_token_returns_401(self) -> None:
        res = client.post("/analysis/run", data={"goal": "Profile this dataset please"})
        assert res.status_code == 401

    def test_ingest_without_token_returns_401(self) -> None:
        res = client.post(
            "/analysis/ingest",
            files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
        )
        assert res.status_code == 401


class TestRunPersistence:
    def test_run_with_file_returns_dataset_and_run_id(self) -> None:
        headers, _ = _auth_headers()
        res = client.post(
            "/analysis/run",
            headers=headers,
            data={"goal": "Identify correlations in this dataset", "expertise_level": "intermediate"},
            files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["dataset_id"]
        assert body["run_id"]

        ingestion_step = next(s for s in body["steps"] if s["agent_name"] == "IngestionAgent")
        assert ingestion_step["status"] == "success"
        assert ingestion_step["output"]["row_count"] == 4

        client.delete("/auth/me", headers=headers)

    def test_followup_reuses_dataset_via_dataset_id(self) -> None:
        headers, _ = _auth_headers()
        first = client.post(
            "/analysis/run",
            headers=headers,
            data={"goal": "Identify correlations in this dataset", "expertise_level": "intermediate"},
            files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
        ).json()

        second = client.post(
            "/analysis/run",
            headers=headers,
            data={
                "goal": "What outliers exist?",
                "expertise_level": "intermediate",
                "dataset_id": first["dataset_id"],
            },
        ).json()

        ingestion_step = next(s for s in second["steps"] if s["agent_name"] == "IngestionAgent")
        assert ingestion_step["status"] == "success"
        assert ingestion_step["output"]["row_count"] == 4
        assert second["dataset_id"] == first["dataset_id"]

        client.delete("/auth/me", headers=headers)

    def test_dataset_from_another_user_is_not_reusable(self) -> None:
        headers_a, _ = _auth_headers()
        headers_b, _ = _auth_headers()

        run_a = client.post(
            "/analysis/run",
            headers=headers_a,
            data={"goal": "Identify correlations in this dataset", "expertise_level": "intermediate"},
            files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
        ).json()

        # User B tries to reuse user A's dataset_id — should silently get
        # no source (ingestion fails gracefully) rather than user A's data.
        run_b = client.post(
            "/analysis/run",
            headers=headers_b,
            data={
                "goal": "What outliers exist?",
                "expertise_level": "intermediate",
                "dataset_id": run_a["dataset_id"],
            },
        ).json()

        ingestion_step = next(s for s in run_b["steps"] if s["agent_name"] == "IngestionAgent")
        assert ingestion_step["status"] == "error"

        client.delete("/auth/me", headers=headers_a)
        client.delete("/auth/me", headers=headers_b)


class TestHistory:
    def test_history_lists_own_runs_only(self) -> None:
        headers_a, _ = _auth_headers()
        headers_b, _ = _auth_headers()

        client.post(
            "/analysis/run",
            headers=headers_a,
            data={"goal": "User A's goal for this run", "expertise_level": "intermediate"},
        )

        history_a = client.get("/analysis/history", headers=headers_a).json()
        history_b = client.get("/analysis/history", headers=headers_b).json()

        assert len(history_a) == 1
        assert history_a[0]["goal"] == "User A's goal for this run"
        assert len(history_b) == 0

        client.delete("/auth/me", headers=headers_a)
        client.delete("/auth/me", headers=headers_b)

    def test_history_detail_matches_original_run(self) -> None:
        headers, _ = _auth_headers()
        run = client.post(
            "/analysis/run",
            headers=headers,
            data={"goal": "Detail lookup test goal", "expertise_level": "expert"},
        ).json()

        detail = client.get(f"/analysis/history/{run['run_id']}", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["goal"] == "Detail lookup test goal"

        client.delete("/auth/me", headers=headers)

    def test_history_detail_not_found_for_other_user(self) -> None:
        headers_a, _ = _auth_headers()
        headers_b, _ = _auth_headers()

        run = client.post(
            "/analysis/run",
            headers=headers_a,
            data={"goal": "Private goal only A should see", "expertise_level": "intermediate"},
        ).json()

        res = client.get(f"/analysis/history/{run['run_id']}", headers=headers_b)
        assert res.status_code == 404

        client.delete("/auth/me", headers=headers_a)
        client.delete("/auth/me", headers=headers_b)


class TestDatasets:
    def test_ingest_persists_dataset(self) -> None:
        headers, _ = _auth_headers()
        res = client.post(
            "/analysis/ingest",
            headers=headers,
            files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
        )
        assert res.status_code == 200
        dataset_id = res.json()["dataset_id"]
        assert dataset_id

        listing = client.get("/analysis/datasets", headers=headers).json()
        assert any(d["id"] == dataset_id for d in listing)

        client.delete("/auth/me", headers=headers)

    def test_delete_dataset(self) -> None:
        headers, _ = _auth_headers()
        dataset_id = client.post(
            "/analysis/ingest",
            headers=headers,
            files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
        ).json()["dataset_id"]

        res = client.delete(f"/analysis/datasets/{dataset_id}", headers=headers)
        assert res.status_code == 200

        listing = client.get("/analysis/datasets", headers=headers).json()
        assert not any(d["id"] == dataset_id for d in listing)

        client.delete("/auth/me", headers=headers)

    def test_delete_nonexistent_dataset_returns_404(self) -> None:
        headers, _ = _auth_headers()
        res = client.delete(f"/analysis/datasets/{uuid.uuid4()}", headers=headers)
        assert res.status_code == 404
        client.delete("/auth/me", headers=headers)
