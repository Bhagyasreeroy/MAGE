"""
tests/backend/test_explain_endpoint.py
─────────────────────────────────────────
Integration tests for POST /analysis/explain. httpx.post is monkeypatched
so no real network/API key is needed.
"""

from __future__ import annotations

import os
import sys
import uuid

import httpx
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


def _fake_gemini_post(*args, **kwargs):
    class _R:
        status_code = 200

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "Synthesized explanation citing the methodology."}]}}]}

    return _R()


class TestExplainEndpoint:
    def test_requires_auth(self) -> None:
        res = client.post("/analysis/explain", json={"finding": "3 outliers detected in revenue."})
        assert res.status_code == 401

    def test_synthesized_when_llm_configured(self, monkeypatch) -> None:
        monkeypatch.setattr(httpx, "post", _fake_gemini_post)
        monkeypatch.setattr("backend.core.config.settings.gemini_api_key", "fake-key-for-test")

        # The live GeminiClient inside the router's ExplainAgent singleton
        # was constructed at import time — patch its instance directly,
        # same reasoning as the /run endpoint's LLM-mode tests.
        from backend.routers.analysis import _explain_agent
        monkeypatch.setattr(_explain_agent._llm_client, "api_key", "fake-key-for-test")

        headers = _auth_headers()
        res = client.post(
            "/analysis/explain",
            headers=headers,
            json={"finding": "3 outlier(s) detected in 'revenue' via IQR (10.0% of rows).", "goal": "find outliers"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["synthesized"] is True
        assert body["explanation"] == "Synthesized explanation citing the methodology."
        assert len(body["sources"]) > 0

    def test_falls_back_without_configured_key(self, monkeypatch) -> None:
        from backend.routers.analysis import _explain_agent
        monkeypatch.setattr(_explain_agent._llm_client, "api_key", "")

        headers = _auth_headers()
        res = client.post(
            "/analysis/explain",
            headers=headers,
            json={"finding": "How should I handle missing values?"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["synthesized"] is False
        assert body["explanation"] != ""
