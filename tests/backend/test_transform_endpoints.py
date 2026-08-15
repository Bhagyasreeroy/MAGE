"""
tests/backend/test_transform_endpoints.py
───────────────────────────────────────────
Integration tests for the dataset-manipulation endpoints: detail,
preview, transform, query, query/save, and version listing.
"""

from __future__ import annotations

import os
import sys
import uuid

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app

client = TestClient(app)

SAMPLE_CSV = (
    b"id,region,revenue,units\n"
    b"1,East,100.0,3\n"
    b"2,West,,5\n"
    b"3,East,300.0,2\n"
    b"4,East,400.0,7\n"
    b"5,West,400.0,7\n"
)


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _auth_headers() -> dict[str, str]:
    email = _unique_email()
    client.post("/auth/register", json={"email": email, "full_name": "Test", "password": "Passw0rd!"})
    token = client.post("/auth/login", json={"email": email, "password": "Passw0rd!"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _ingest(headers: dict[str, str]) -> str:
    res = client.post("/analysis/ingest", headers=headers, files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")})
    assert res.status_code == 200
    return res.json()["dataset_id"]


class TestDatasetDetail:
    def test_requires_auth(self) -> None:
        res = client.get("/analysis/datasets/whatever")
        assert res.status_code == 401

    def test_returns_full_detail(self) -> None:
        headers = _auth_headers()
        dataset_id = _ingest(headers)
        res = client.get(f"/analysis/datasets/{dataset_id}", headers=headers)
        assert res.status_code == 200
        body = res.json()
        assert body["id"] == dataset_id
        assert body["version"] == 1
        assert body["root_id"] == dataset_id
        assert body["parent_id"] is None
        assert len(body["column_summary"]) == 4

    def test_not_found_for_missing_dataset(self) -> None:
        headers = _auth_headers()
        res = client.get("/analysis/datasets/does-not-exist", headers=headers)
        assert res.status_code == 404

    def test_not_found_for_other_users_dataset(self) -> None:
        headers_a = _auth_headers()
        headers_b = _auth_headers()
        dataset_id = _ingest(headers_a)
        res = client.get(f"/analysis/datasets/{dataset_id}", headers=headers_b)
        assert res.status_code == 404


class TestPreview:
    def test_returns_paginated_rows(self) -> None:
        headers = _auth_headers()
        dataset_id = _ingest(headers)
        res = client.get(f"/analysis/datasets/{dataset_id}/preview?offset=0&limit=2", headers=headers)
        assert res.status_code == 200
        body = res.json()
        assert body["columns"] == ["id", "region", "revenue", "units"]
        assert body["total_rows"] == 5
        assert len(body["rows"]) == 2

    def test_missing_value_serializes_as_null(self) -> None:
        headers = _auth_headers()
        dataset_id = _ingest(headers)
        res = client.get(f"/analysis/datasets/{dataset_id}/preview?limit=5", headers=headers)
        rows = res.json()["rows"]
        # row index 1 (id=2, West) has a missing revenue value
        assert rows[1][2] is None


class TestTransform:
    def test_drop_columns_creates_new_version(self) -> None:
        headers = _auth_headers()
        dataset_id = _ingest(headers)
        res = client.post(
            f"/analysis/datasets/{dataset_id}/transform",
            headers=headers,
            json={"ops": [{"type": "drop_columns", "columns": ["units"]}]},
        )
        assert res.status_code == 201
        body = res.json()
        assert body["parent_id"] == dataset_id
        assert body["version"] == 2
        assert body["transform_type"] == "clean"
        assert "units" not in [c["name"] for c in body["column_summary"]]
        assert "Dropped 1 column(s)" in body["report"][0]

    def test_invalid_op_returns_400(self) -> None:
        headers = _auth_headers()
        dataset_id = _ingest(headers)
        res = client.post(
            f"/analysis/datasets/{dataset_id}/transform",
            headers=headers,
            json={"ops": [{"type": "drop_columns", "columns": ["does_not_exist"]}]},
        )
        assert res.status_code == 400

    def test_unknown_dataset_returns_404(self) -> None:
        headers = _auth_headers()
        res = client.post(
            "/analysis/datasets/does-not-exist/transform",
            headers=headers,
            json={"ops": [{"type": "dedupe"}]},
        )
        assert res.status_code == 404

    def test_edit_cells_via_transform_endpoint(self) -> None:
        headers = _auth_headers()
        dataset_id = _ingest(headers)
        res = client.post(
            f"/analysis/datasets/{dataset_id}/transform",
            headers=headers,
            json={"ops": [{"type": "edit_cells", "edits": [{"row_index": 0, "column": "region", "value": "North"}]}]},
        )
        assert res.status_code == 201
        new_id = res.json()["id"]
        preview = client.get(f"/analysis/datasets/{new_id}/preview", headers=headers).json()
        assert preview["rows"][0][1] == "North"


class TestVersions:
    def test_lists_every_version_in_order(self) -> None:
        headers = _auth_headers()
        dataset_id = _ingest(headers)
        v2 = client.post(
            f"/analysis/datasets/{dataset_id}/transform",
            headers=headers,
            json={"ops": [{"type": "drop_columns", "columns": ["units"]}]},
        ).json()

        res = client.get(f"/analysis/datasets/{dataset_id}/versions", headers=headers)
        assert res.status_code == 200
        versions = res.json()
        assert [v["version"] for v in versions] == [1, 2]
        assert versions[1]["id"] == v2["id"]
        assert versions[1]["parent_id"] == dataset_id


class TestQuery:
    def test_preview_does_not_persist(self) -> None:
        headers = _auth_headers()
        dataset_id = _ingest(headers)
        res = client.post(
            f"/analysis/datasets/{dataset_id}/query",
            headers=headers,
            json={"sql": "SELECT * FROM df WHERE region = 'East'"},
        )
        assert res.status_code == 200
        assert res.json()["total_rows"] == 3

        versions = client.get(f"/analysis/datasets/{dataset_id}/versions", headers=headers).json()
        assert len(versions) == 1

    def test_rejects_non_select(self) -> None:
        headers = _auth_headers()
        dataset_id = _ingest(headers)
        res = client.post(
            f"/analysis/datasets/{dataset_id}/query",
            headers=headers,
            json={"sql": "DROP TABLE df"},
        )
        assert res.status_code == 400

    def test_blocks_filesystem_read(self) -> None:
        headers = _auth_headers()
        dataset_id = _ingest(headers)
        res = client.post(
            f"/analysis/datasets/{dataset_id}/query",
            headers=headers,
            json={"sql": "SELECT * FROM read_csv('/etc/passwd')"},
        )
        assert res.status_code == 400

    def test_save_creates_new_version(self) -> None:
        headers = _auth_headers()
        dataset_id = _ingest(headers)
        res = client.post(
            f"/analysis/datasets/{dataset_id}/query/save",
            headers=headers,
            json={"sql": "SELECT * FROM df WHERE region = 'East'"},
        )
        assert res.status_code == 201
        body = res.json()
        assert body["transform_type"] == "query_save"
        assert body["parent_id"] == dataset_id
        assert body["transform_params"]["sql"] == "SELECT * FROM df WHERE region = 'East'"

        versions = client.get(f"/analysis/datasets/{dataset_id}/versions", headers=headers).json()
        assert len(versions) == 2
