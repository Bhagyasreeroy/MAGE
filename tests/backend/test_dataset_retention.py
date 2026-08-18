"""
tests/backend/test_dataset_retention.py
────────────────────────────────────────
NFR-04 — uploaded data is not kept forever.

Current behaviour is "keep everything": 686 datasets are sitting in Postgres on
this machine, every one of them holding the uploaded file's bytes in a
LargeBinary column. The requirement asks for sandboxed, session-scoped files.

**This is the only feature in the project that deletes user data, so the guards
matter more than the sweep.** Two of them are load-bearing:

  * A dataset referenced by a completed ``analysis_run`` is never purged. The
    FK is ``ON DELETE SET NULL``, so deleting one would not error — it would
    quietly detach the run from its data and break the report the user already
    generated. Silent damage to an existing artefact is worse than keeping a
    file.
  * Only rows genuinely past ``expires_at`` are touched, and the default TTL is
    generous. A retention sweep that is too eager is indistinguishable from data
    loss.

The negative tests below are the point of this file. The positive one is easy.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.core.config import settings
from backend.main import app

client = TestClient(app)

CSV = b"a,b\n1,2\n3,4\n5,6\n"


def _auth() -> dict:
    email = f"ret-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/auth/register", json={"email": email, "password": "Str0ng!Pass123", "full_name": "Ret"})
    token = client.post(
        "/auth/login", json={"email": email, "password": "Str0ng!Pass123"}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _upload(headers) -> str:
    res = client.post("/analysis/ingest", headers=headers, files={"file": ("r.csv", CSV, "text/csv")})
    assert res.status_code == 200, res.text
    return res.json()["dataset_id"]


def _expire(dataset_id: str, *, days_ago: int = 1) -> None:
    """Backdate a dataset's expiry so the sweep should collect it."""
    import asyncio

    from sqlalchemy import update

    from backend.core.database import async_session
    from backend.models.dataset import Dataset

    async def go():
        async with async_session() as db:
            await db.execute(
                update(Dataset)
                .where(Dataset.id == dataset_id)
                .values(expires_at=datetime.now(timezone.utc) - timedelta(days=days_ago))
            )
            await db.commit()

    asyncio.run(go())


class TestExpiryIsRecorded:
    def test_an_upload_gets_an_expiry(self, ) -> None:
        headers = _auth()
        dsid = _upload(headers)
        res = client.get("/analysis/datasets", headers=headers)
        row = next(d for d in res.json() if d["id"] == dsid)
        assert row.get("expires_at"), "an uploaded dataset carries no retention date"

    def test_the_default_ttl_is_generous(self) -> None:
        """An aggressive default would delete a user's work mid-project."""
        assert settings.dataset_retention_days >= 7

    def test_the_ttl_is_configurable(self) -> None:
        assert isinstance(settings.dataset_retention_days, int)


class TestTheSweepCollectsExpiredData:
    def test_an_expired_unreferenced_dataset_is_purged(self) -> None:
        headers = _auth()
        dsid = _upload(headers)
        _expire(dsid)
        res = client.post("/analysis/datasets/purge-expired", headers=headers)
        assert res.status_code == 200
        assert res.json()["purged"] >= 1
        assert client.get(f"/analysis/datasets/{dsid}/preview", headers=headers).status_code == 404

    def test_the_sweep_reports_what_it_removed(self) -> None:
        headers = _auth()
        _expire(_upload(headers))
        _expire(_upload(headers))
        assert client.post("/analysis/datasets/purge-expired", headers=headers).json()["purged"] == 2

    def test_the_sweep_is_idempotent(self) -> None:
        headers = _auth()
        _expire(_upload(headers))
        client.post("/analysis/datasets/purge-expired", headers=headers)
        assert client.post("/analysis/datasets/purge-expired", headers=headers).json()["purged"] == 0


class TestTheGuards:
    """The half that keeps this from being a data-loss bug."""

    def test_an_unexpired_dataset_survives(self) -> None:
        headers = _auth()
        dsid = _upload(headers)
        client.post("/analysis/datasets/purge-expired", headers=headers)
        assert client.get(f"/analysis/datasets/{dsid}/preview", headers=headers).status_code == 200

    def test_a_dataset_referenced_by_a_run_survives_even_when_expired(self) -> None:
        """
        The FK is ON DELETE SET NULL, so purging this would not raise — it would
        silently detach a completed run from its data and break a report the
        user already has. Keeping the file is the lesser harm.
        """
        headers = _auth()
        dsid = _upload(headers)
        run = client.post(
            "/analysis/run",
            headers=headers,
            data={"goal": "profile this dataset", "dataset_id": dsid, "expertise_level": "beginner"},
        )
        assert run.status_code == 200, run.text
        _expire(dsid)

        purged = client.post("/analysis/datasets/purge-expired", headers=headers).json()["purged"]
        assert purged == 0, "purged a dataset that a completed run depends on"
        assert client.get(f"/analysis/datasets/{dsid}/preview", headers=headers).status_code == 200

    def test_the_run_report_still_resolves_after_a_sweep(self) -> None:
        headers = _auth()
        dsid = _upload(headers)
        run_id = client.post(
            "/analysis/run",
            headers=headers,
            data={"goal": "profile this dataset", "dataset_id": dsid, "expertise_level": "beginner"},
        ).json()["run_id"]
        _expire(dsid)
        client.post("/analysis/datasets/purge-expired", headers=headers)
        assert client.get(f"/analysis/history/{run_id}", headers=headers).status_code == 200
        assert client.get(f"/analysis/history/{run_id}/export/pdf", headers=headers).status_code == 200

    def test_the_sweep_never_crosses_users(self) -> None:
        """Retention is not a licence to touch someone else's rows."""
        victim = _auth()
        victim_ds = _upload(victim)
        _expire(victim_ds)

        attacker = _auth()
        client.post("/analysis/datasets/purge-expired", headers=attacker)
        assert client.get(f"/analysis/datasets/{victim_ds}/preview", headers=victim).status_code == 200

    def test_purging_requires_authentication(self) -> None:
        assert client.post("/analysis/datasets/purge-expired").status_code in (401, 403)
