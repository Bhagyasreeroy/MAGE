"""
tests/backend/test_llm_mode.py
─────────────────────────────────
Integration tests for the "mode" selector on POST /analysis/run — the
RAG-vs-LLM toggle. httpx.post is monkeypatched so no real network call or
API key is needed; this only exercises the plumbing (mode threaded from
the HTTP form field through OrchestratorService -> OrchestratorAgent ->
RecommendationAgent and back into the response).
"""

from __future__ import annotations

import os
import sys
import uuid

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app
from backend.routers.analysis import _orchestrator_service

client = TestClient(app)


def _live_llm_client():
    """The actual GeminiClient instance used by the running app — it's a
    singleton constructed once at module import (reads settings.gemini_api_key
    at construction time), so tests must patch *this instance's* .api_key
    rather than the Settings object after the fact."""
    return _orchestrator_service._agent._agents["RecommendationAgent"]._llm_client

SAMPLE_CSV = b"order_id,region,revenue\n1,East,45.0\n2,West,120.5\n3,East,75.0\n4,North,60.0\n"


class _FakeGeminiResponse:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body
        self.text = ""

    def json(self) -> dict:
        return self._body


def _fake_gemini_post(*args, **kwargs):
    body = {"candidates": [{"content": {"parts": [{"text": "Revenue looks concentrated in the East region."}]}}]}
    return _FakeGeminiResponse(200, body)


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


def _auth_headers() -> dict[str, str]:
    email = _unique_email()
    client.post("/auth/register", json={"email": email, "full_name": "Test", "password": "Passw0rd!"})
    token = client.post("/auth/login", json={"email": email, "password": "Passw0rd!"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestModeDefaultsToRag:
    def test_omitting_mode_defaults_to_rag(self) -> None:
        headers = _auth_headers()
        res = client.post(
            "/analysis/run",
            headers=headers,
            data={"goal": "Profile this dataset please"},
            files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
        )
        assert res.status_code == 200
        assert res.json()["mode"] == "rag"


class TestLLMMode:
    def test_llm_mode_returns_generated_text_and_no_sources(self, monkeypatch) -> None:
        monkeypatch.setattr(httpx, "post", _fake_gemini_post)
        # Backend needs a key configured for is_configured to be True —
        # patch the live singleton's client, not Settings (see _live_llm_client).
        monkeypatch.setattr(_live_llm_client(), "api_key", "fake-key-for-test")

        headers = _auth_headers()
        res = client.post(
            "/analysis/run",
            headers=headers,
            data={"goal": "What drives revenue in this dataset?", "mode": "llm"},
            files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["mode"] == "llm"
        assert len(body["recommendations"]) == 1
        assert "East region" in body["recommendations"][0]
        assert body["rag_sources"] == []

    def test_llm_mode_without_configured_key_degrades_gracefully(self, monkeypatch) -> None:
        monkeypatch.setattr(_live_llm_client(), "api_key", "")

        headers = _auth_headers()
        res = client.post(
            "/analysis/run",
            headers=headers,
            data={"goal": "What drives revenue in this dataset?", "mode": "llm"},
            files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
        )
        assert res.status_code == 200
        body = res.json()
        assert len(body["recommendations"]) == 1
        assert "API key" in body["recommendations"][0]

    def test_invalid_mode_rejected(self) -> None:
        headers = _auth_headers()
        res = client.post(
            "/analysis/run",
            headers=headers,
            data={"goal": "Profile this dataset please", "mode": "not-a-real-mode"},
            files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
        )
        assert res.status_code == 422
