"""Tests for agents/llm_client.py."""

from __future__ import annotations

import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.llm_client import GeminiClient, LLMError


class _FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self) -> dict:
        return self._body


def _gemini_success_body(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class TestIsConfigured:
    def test_no_key_is_unconfigured(self) -> None:
        assert GeminiClient(api_key="").is_configured is False

    def test_with_key_is_configured(self) -> None:
        assert GeminiClient(api_key="fake-key").is_configured is True


class TestGenerate:
    def test_raises_without_key(self) -> None:
        client = GeminiClient(api_key="")
        with pytest.raises(LLMError, match="No Gemini API key"):
            client.generate("hello")

    def test_successful_generation(self, monkeypatch) -> None:
        monkeypatch.setattr(
            httpx, "post", lambda *a, **k: _FakeResponse(200, _gemini_success_body("The answer is 42."))
        )
        client = GeminiClient(api_key="fake-key")
        result = client.generate("what is the answer?")
        assert result == "The answer is 42."

    def test_non_200_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(429, text="rate limited"))
        client = GeminiClient(api_key="fake-key")
        with pytest.raises(LLMError, match="HTTP 429"):
            client.generate("hello")

    def test_unexpected_response_shape_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(200, {"unexpected": "shape"}))
        client = GeminiClient(api_key="fake-key")
        with pytest.raises(LLMError, match="Unexpected Gemini response shape"):
            client.generate("hello")

    def test_network_error_raises(self, monkeypatch) -> None:
        def _raise(*a, **k):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "post", _raise)
        client = GeminiClient(api_key="fake-key")
        with pytest.raises(LLMError, match="Gemini request failed"):
            client.generate("hello")

    def test_uses_configured_model_in_url(self, monkeypatch) -> None:
        captured = {}

        def _fake_post(url, **kwargs):
            captured["url"] = url
            return _FakeResponse(200, _gemini_success_body("ok"))

        monkeypatch.setattr(httpx, "post", _fake_post)
        client = GeminiClient(api_key="fake-key", model="gemini-custom")
        client.generate("hello")
        assert "gemini-custom" in captured["url"]
