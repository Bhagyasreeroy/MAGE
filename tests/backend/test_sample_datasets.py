"""
tests/backend/test_sample_datasets.py
──────────────────────────────────────
Integration tests for the bundled demo-dataset endpoints — lets a user try
MAGE without a file of their own on disk.
"""

from __future__ import annotations

import os
import sys
import uuid

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app

client = TestClient(app)


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _auth_headers() -> dict[str, str]:
    email = _unique_email()
    client.post("/auth/register", json={"email": email, "full_name": "Test", "password": "Passw0rd!"})
    token = client.post("/auth/login", json={"email": email, "password": "Passw0rd!"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestListSampleDatasets:
    def test_lists_bundled_datasets(self) -> None:
        # Public, like /knowledge-sources — no user data involved in just
        # listing what's available to load.
        res = client.get("/analysis/sample-datasets")
        assert res.status_code == 200
        body = res.json()
        assert len(body) >= 1
        filenames = {d["filename"] for d in body}
        assert "customer_orders.csv" in filenames
        for entry in body:
            assert entry["title"]
            assert entry["description"]
            assert entry["size_kb"] > 0


class TestLoadSampleDataset:
    def test_requires_auth(self) -> None:
        res = client.post("/analysis/sample-datasets/customer_orders.csv/load")
        assert res.status_code == 401

    def test_loads_and_persists_a_dataset(self) -> None:
        headers = _auth_headers()
        res = client.post("/analysis/sample-datasets/customer_orders.csv/load", headers=headers)
        assert res.status_code == 200
        body = res.json()
        assert body["row_count"] > 0
        assert body["column_count"] > 0
        assert body["dataset_id"] is not None

        # The persisted dataset shows up in the user's own dataset list.
        datasets = client.get("/analysis/datasets", headers=headers).json()
        assert any(d["id"] == body["dataset_id"] for d in datasets)

    def test_loaded_dataset_is_usable_in_a_run(self) -> None:
        headers = _auth_headers()
        loaded = client.post("/analysis/sample-datasets/customer_orders.csv/load", headers=headers).json()

        res = client.post(
            "/analysis/run",
            headers=headers,
            data={"goal": "Find outliers in this dataset", "dataset_id": loaded["dataset_id"]},
        )
        assert res.status_code == 200
        assert res.json()["dataset_id"] == loaded["dataset_id"]

    def test_unknown_filename_returns_404(self) -> None:
        headers = _auth_headers()
        res = client.post("/analysis/sample-datasets/does-not-exist.csv/load", headers=headers)
        assert res.status_code == 404

    def test_path_traversal_attempt_returns_404_not_500(self) -> None:
        headers = _auth_headers()
        res = client.post("/analysis/sample-datasets/..%2F..%2Fpyproject.toml/load", headers=headers)
        assert res.status_code == 404
