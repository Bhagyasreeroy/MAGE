"""
tests/backend/test_transcribe_endpoint.py
─────────────────────────────────────────
Integration tests for POST /analysis/transcribe (voice input).

Unlike the other endpoint tests here, these never touch Postgres:
``get_current_user`` is overridden with a stub, and the endpoint takes no
database dependency of its own, so the whole file runs without a live
database. httpx.post is monkeypatched, so no API key or network is needed
either.
"""

from __future__ import annotations

import base64
import os
import struct
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.core.deps import get_current_user
from backend.main import app
from backend.models.user import User
from backend.routers.analysis import _gemini_client

client = TestClient(app)

TRANSCRIPT = "  Find the top factors driving customer churn.  "


def _wav_bytes(frames: int = 1600) -> bytes:
    """A minimal, valid 16 kHz mono WAV — the shape the browser hook uploads."""
    data = b"\x00\x00" * frames
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


WAV = _wav_bytes()


@pytest.fixture
def authed():
    """Stub out authentication; the endpoint needs a user, not a database."""
    app.dependency_overrides[get_current_user] = lambda: User(
        id="voice-test-user", email="voice@example.com", is_active=True
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def configured(monkeypatch):
    """The router holds a GeminiClient built at import time — patch that instance."""
    monkeypatch.setattr(_gemini_client, "api_key", "fake-key-for-test")


@pytest.fixture
def gemini(monkeypatch):
    """Capture the outgoing request body so the payload shape can be asserted."""
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured["json"] = kwargs.get("json")

        class _R:
            status_code = 200
            text = ""

            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": TRANSCRIPT}]}}]}

        return _R()

    monkeypatch.setattr(httpx, "post", fake_post)
    return captured


class TestAuthAndConfig:
    def test_requires_auth(self) -> None:
        res = client.post("/analysis/transcribe", files={"audio": ("a.wav", WAV, "audio/wav")})
        assert res.status_code == 401

    def test_503_when_no_api_key(self, authed, monkeypatch) -> None:
        monkeypatch.setattr(_gemini_client, "api_key", "")
        res = client.post("/analysis/transcribe", files={"audio": ("a.wav", WAV, "audio/wav")})
        # Not a 500: nothing the caller did is wrong, and nothing they can do
        # will fix it, so the message must point at server configuration.
        assert res.status_code == 503
        assert "Gemini API key" in res.json()["detail"]


class TestTranscription:
    def test_returns_stripped_transcript(self, authed, configured, gemini) -> None:
        res = client.post("/analysis/transcribe", files={"audio": ("a.wav", WAV, "audio/wav")})
        assert res.status_code == 200
        assert res.json()["transcript"] == "Find the top factors driving customer churn."

    def test_sends_audio_inline_with_its_mime_type(self, authed, configured, gemini) -> None:
        client.post("/analysis/transcribe", files={"audio": ("a.wav", WAV, "audio/wav")})
        parts = gemini["json"]["contents"][0]["parts"]
        assert "text" in parts[0], "the transcribe instruction must lead"
        assert parts[1]["inline_data"]["mime_type"] == "audio/wav"
        assert base64.b64decode(parts[1]["inline_data"]["data"]) == WAV

    def test_codec_parameter_is_stripped_from_mime_type(self, authed, configured, gemini) -> None:
        # Browsers send "audio/webm;codecs=opus"; Gemini wants the bare type.
        res = client.post(
            "/analysis/transcribe", files={"audio": ("a.webm", WAV, "audio/webm;codecs=opus")}
        )
        assert res.status_code == 200
        assert gemini["json"]["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "audio/webm"

    def test_silence_yields_empty_transcript_not_an_error(self, authed, configured, monkeypatch) -> None:
        # Asked to transcribe silence, Gemini returns a candidate with no
        # parts at all. That is "you said nothing", not a malformed response.
        def fake_post(url, **kwargs):
            class _R:
                status_code = 200
                text = ""

                def json(self):
                    return {"candidates": [{"finishReason": "STOP"}]}

            return _R()

        monkeypatch.setattr(httpx, "post", fake_post)
        res = client.post("/analysis/transcribe", files={"audio": ("a.wav", WAV, "audio/wav")})
        assert res.status_code == 200
        assert res.json()["transcript"] == ""

    def test_upstream_failure_reported_as_502(self, authed, configured, monkeypatch) -> None:
        def fake_post(url, **kwargs):
            class _R:
                status_code = 400
                text = "bad audio"

                def json(self):
                    return {}

            return _R()

        monkeypatch.setattr(httpx, "post", fake_post)
        res = client.post("/analysis/transcribe", files={"audio": ("a.wav", WAV, "audio/wav")})
        assert res.status_code == 502
        # The upstream detail is logged, never echoed to the client.
        assert "bad audio" not in res.json()["detail"]


class TestRejections:
    def test_415_for_a_non_audio_upload(self, authed, configured) -> None:
        res = client.post("/analysis/transcribe", files={"audio": ("a.csv", b"a,b\n1,2", "text/csv")})
        assert res.status_code == 415

    def test_400_for_an_empty_recording(self, authed, configured) -> None:
        res = client.post("/analysis/transcribe", files={"audio": ("a.wav", b"", "audio/wav")})
        assert res.status_code == 400

    def test_413_over_the_size_cap(self, authed, configured) -> None:
        from backend.routers.analysis import MAX_AUDIO_BYTES

        oversized = b"\x00" * (MAX_AUDIO_BYTES + 1)
        res = client.post("/analysis/transcribe", files={"audio": ("a.wav", oversized, "audio/wav")})
        assert res.status_code == 413
