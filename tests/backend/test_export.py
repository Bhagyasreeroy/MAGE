"""
tests/backend/test_export.py
──────────────────────────────
Integration tests for the /analysis/history/{run_id}/export/* endpoints:
PDF report, JSON, and citation bundle downloads.

Runs against the configured DATABASE_URL (Postgres); each test creates
its own user and deletes it (cascade) when done.
"""

from __future__ import annotations

import json
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


def _auth_headers() -> dict[str, str]:
    email = _unique_email()
    client.post("/auth/register", json={"email": email, "full_name": "Test", "password": "Passw0rd!"})
    token = client.post("/auth/login", json={"email": email, "password": "Passw0rd!"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_run(headers: dict[str, str]) -> str:
    res = client.post(
        "/analysis/run",
        headers=headers,
        data={"goal": "Identify correlations and outliers in this dataset", "expertise_level": "intermediate"},
        files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
    )
    return res.json()["run_id"]


class TestExportAuth:
    def test_pdf_export_without_token_returns_401(self) -> None:
        res = client.get("/analysis/history/some-id/export/pdf")
        assert res.status_code == 401

    def test_json_export_without_token_returns_401(self) -> None:
        res = client.get("/analysis/history/some-id/export/json")
        assert res.status_code == 401

    def test_citations_export_without_token_returns_401(self) -> None:
        res = client.get("/analysis/history/some-id/export/citations")
        assert res.status_code == 401


class TestExportOwnership:
    def test_pdf_export_404_for_nonexistent_run(self) -> None:
        headers = _auth_headers()
        res = client.get("/analysis/history/does-not-exist/export/pdf", headers=headers)
        assert res.status_code == 404
        client.delete("/auth/me", headers=headers)

    def test_export_404_for_another_users_run(self) -> None:
        headers_a = _auth_headers()
        headers_b = _auth_headers()
        run_id = _create_run(headers_a)

        res = client.get(f"/analysis/history/{run_id}/export/pdf", headers=headers_b)
        assert res.status_code == 404

        client.delete("/auth/me", headers=headers_a)
        client.delete("/auth/me", headers=headers_b)


class TestPdfExport:
    def test_returns_a_valid_pdf(self) -> None:
        headers = _auth_headers()
        run_id = _create_run(headers)

        res = client.get(f"/analysis/history/{run_id}/export/pdf", headers=headers)
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/pdf"
        assert f"mage-report-{run_id}.pdf" in res.headers["content-disposition"]
        assert res.content.startswith(b"%PDF")

        client.delete("/auth/me", headers=headers)


class TestJsonExport:
    def test_returns_the_full_run_payload(self) -> None:
        headers = _auth_headers()
        run_id = _create_run(headers)

        res = client.get(f"/analysis/history/{run_id}/export/json", headers=headers)
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/json"
        assert f"mage-export-{run_id}.json" in res.headers["content-disposition"]

        body = json.loads(res.content)
        assert body["run_id"] == run_id
        assert body["goal"] == "Identify correlations and outliers in this dataset"
        assert any(s["agent_name"] == "IngestionAgent" for s in body["steps"])

        client.delete("/auth/me", headers=headers)


class TestCitationExport:
    def test_returns_bibtex_for_grounded_sources(self) -> None:
        headers = _auth_headers()
        run_id = _create_run(headers)

        res = client.get(f"/analysis/history/{run_id}/export/citations", headers=headers)
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/x-bibtex"
        assert f"mage-citations-{run_id}.bib" in res.headers["content-disposition"]

        text = res.content.decode()
        assert "@misc{" in text
        assert "MAGE Knowledge Base" in text

        client.delete("/auth/me", headers=headers)
