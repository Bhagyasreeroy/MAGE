"""Tests for agents/explain_agent.py."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.explain_agent import ExplainAgent
from agents.llm_client import LLMError
from rag.vector_store import VectorStore


class _FakeLLMClient:
    """Duck-types GeminiClient for tests — no network, no API key."""

    def __init__(self, configured: bool = True, response: str = "Synthesized explanation.", error: Exception | None = None) -> None:
        self._configured = configured
        self._response = response
        self._error = error
        self.last_prompt: str | None = None

    @property
    def is_configured(self) -> bool:
        return self._configured

    def generate(self, prompt: str, timeout: float = 20.0) -> str:
        self.last_prompt = prompt
        if self._error is not None:
            raise self._error
        return self._response


@pytest.fixture
def agent_factory(tmp_path):
    def _make(llm_client) -> ExplainAgent:
        vector_store = VectorStore(backend="faiss", persist_path=str(tmp_path / "vs"))
        return ExplainAgent(vector_store=vector_store, llm_client=llm_client)

    return _make


class TestSynthesizedPath:
    def test_grounded_and_synthesized_when_llm_configured(self, agent_factory) -> None:
        fake = _FakeLLMClient(response="This matters because missing data can bias downstream analysis.")
        agent = agent_factory(fake)

        result = agent.explain("'units' has 1 missing value.", goal="clean this dataset")

        assert result["synthesized"] is True
        assert result["explanation"] == "This matters because missing data can bias downstream analysis."
        assert len(result["sources"]) > 0
        assert any("missing_values" in s for s in result["sources"])

    def test_prompt_includes_finding_and_multiple_excerpts(self, agent_factory) -> None:
        fake = _FakeLLMClient()
        agent = agent_factory(fake)

        agent.explain("How should I handle missing values?")

        assert "How should I handle missing values?" in fake.last_prompt
        assert "Source:" in fake.last_prompt


class TestFallbackPath:
    def test_unconfigured_client_falls_back_to_raw_excerpt(self, agent_factory) -> None:
        fake = _FakeLLMClient(configured=False)
        agent = agent_factory(fake)

        result = agent.explain("How should I handle missing values?")

        assert result["synthesized"] is False
        assert result["explanation"] != ""
        assert len(result["sources"]) == 1
        assert fake.last_prompt is None  # LLM never called

    def test_failed_llm_call_falls_back_without_raising(self, agent_factory) -> None:
        fake = _FakeLLMClient(error=LLMError("Gemini returned HTTP 503"))
        agent = agent_factory(fake)

        result = agent.explain("How should I handle missing values?")

        assert result["synthesized"] is False
        assert result["explanation"] != ""

    def test_no_kb_match_returns_graceful_message(self, agent_factory) -> None:
        fake = _FakeLLMClient()
        agent = agent_factory(fake)

        result = agent.explain("asdkjfhalskdjfh completely unrelated gibberish query xyz123")

        assert result["synthesized"] is False
        assert result["sources"] == []
        assert result["explanation"] != ""
