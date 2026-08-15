"""Tests for agents/recommendation_agent.py."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.llm_client import LLMError
from agents.recommendation_agent import RecommendationAgent
from rag.vector_store import VectorStore


class _FakeLLMClient:
    """Duck-types GeminiClient for tests — no network, no API key."""

    def __init__(self, configured: bool = True, response: str = "This is the LLM's answer.", error: Exception | None = None) -> None:
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
def agent(tmp_path) -> RecommendationAgent:
    # Isolated, empty vector store per test so the seed KB gets freshly
    # ingested and tests don't depend on / pollute the shared dev store.
    vector_store = VectorStore(backend="faiss", persist_path=str(tmp_path / "vs"))
    return RecommendationAgent(vector_store=vector_store)


class TestRecommendationAgent:
    def test_run_with_no_goal_returns_empty_recommendations(self, agent: RecommendationAgent) -> None:
        result = agent.run(context={})
        assert result["recommendations"] == []
        assert result["rag_sources"] == []

    def test_run_grounds_recommendations_in_kb_sources(self, agent: RecommendationAgent) -> None:
        result = agent.run(context={"goal": "How should I handle missing values in my dataset?"})

        assert len(result["recommendations"]) > 0
        assert any("missing_values" in source for source in result["rag_sources"])

    def test_recommendation_shape(self, agent: RecommendationAgent) -> None:
        result = agent.run(context={"goal": "How do I detect outliers in a numeric column?"})
        rec = result["recommendations"][0]

        assert set(rec.keys()) == {"insight", "text_technical", "text_plain", "confidence", "sources"}
        assert isinstance(rec["confidence"], float)
        assert 0.0 <= rec["confidence"] <= 1.0
        assert rec["text_plain"] != ""
        assert rec["text_technical"] != ""

    def test_plain_text_is_shorter_and_markdown_free(self, agent: RecommendationAgent) -> None:
        result = agent.run(context={"goal": "What chart should I use to compare distributions across groups?"})
        rec = result["recommendations"][0]

        assert "##" not in rec["text_plain"]
        assert "|" not in rec["text_plain"]

    def test_recommendations_are_deduplicated_by_source(self, agent: RecommendationAgent) -> None:
        result = agent.run(context={"goal": "Tell me everything about handling missing data"})
        sources = [s for rec in result["recommendations"] for s in rec["sources"]]
        assert len(sources) == len(set(sources))

    def test_query_incorporates_mining_patterns(self, agent: RecommendationAgent) -> None:
        context = {
            "goal": "Summarize this dataset",
            "MiningAgent_output": {"patterns": ["unknown number of clusters in customer segments"]},
        }
        result = agent.run(context=context)
        assert any("clustering" in source for source in result["rag_sources"])

    def test_run_is_idempotent_across_calls(self, agent: RecommendationAgent) -> None:
        """Calling run() twice should not duplicate KB ingestion into the store."""
        agent.run(context={"goal": "How do I handle missing values?"})
        result = agent.run(context={"goal": "How do I handle missing values?"})
        assert len(result["recommendations"]) > 0


class TestQAShortCircuit:
    """A specific factual question should get a direct computed answer, not
    a RAG dump — this is what makes follow-up chat feel conversational."""

    def test_direct_question_returns_single_computed_recommendation(self, agent: RecommendationAgent) -> None:
        context = {
            "goal": "which column has the most missing values",
            "MiningAgent_output": {
                "data_quality": {
                    "units": {"completeness_pct": 90.0, "uniqueness_pct": 50.0, "missing_count": 3},
                    "revenue": {"completeness_pct": 100.0, "uniqueness_pct": 90.0, "missing_count": 0},
                },
            },
        }
        result = agent.run(context=context)

        assert len(result["recommendations"]) == 1
        rec = result["recommendations"][0]
        assert rec["insight"] == "Computed from your data"
        assert rec["confidence"] == 1.0
        assert "units" in rec["text_technical"]
        assert rec["text_technical"] == rec["text_plain"]

    def test_broad_goal_falls_through_to_rag(self, agent: RecommendationAgent) -> None:
        context = {
            "goal": "Summarize this dataset",
            "MiningAgent_output": {"data_quality": {"units": {"completeness_pct": 100.0, "uniqueness_pct": 50.0, "missing_count": 0}}},
        }
        result = agent.run(context=context)
        assert len(result["recommendations"]) > 0
        assert result["recommendations"][0]["insight"] != "Computed from your data"


class TestFindingLedRecommendations:
    def test_pattern_leads_the_technical_text(self, agent: RecommendationAgent) -> None:
        context = {
            "goal": "Find outliers in this dataset",
            "MiningAgent_output": {
                "patterns": ["3 outlier(s) detected in 'revenue' via IQR (10.0% of rows)."],
            },
        }
        result = agent.run(context=context)

        assert len(result["recommendations"]) > 0
        technical = result["recommendations"][0]["text_technical"]
        assert technical.startswith("**3 outlier(s) detected in 'revenue'")
        # The chunk's own heading must still be on its own line (not fused
        # onto the pattern line), or Markdown can't render it as a heading.
        assert "\n\n#" in technical or technical.count("\n\n") >= 1

    def test_no_mid_sentence_fragment_at_start_of_chunk_portion(self, agent: RecommendationAgent) -> None:
        context = {"goal": "Tell me about clustering methods for this data"}
        result = agent.run(context=context)
        for rec in result["recommendations"]:
            # A leading lowercase word (outside of markdown emphasis/heading
            # markers) indicates an un-trimmed overlap fragment.
            body = rec["text_technical"].lstrip("*#> \n")
            assert not body[:1].islower(), f"starts mid-sentence: {body[:60]!r}"


class TestLLMMode:
    """mode='llm' bypasses QAAgent and RAG entirely — no retrieval, no
    vector store touched, straight to the injected LLM client."""

    def test_llm_mode_returns_generated_text_in_both_registers(self, tmp_path) -> None:
        vector_store = VectorStore(backend="faiss", persist_path=str(tmp_path / "vs"))
        fake = _FakeLLMClient(response="Revenue and units are strongly correlated.")
        agent = RecommendationAgent(vector_store=vector_store, llm_client=fake)

        result = agent.run(context={"goal": "What drives revenue?", "mode": "llm"})

        assert len(result["recommendations"]) == 1
        rec = result["recommendations"][0]
        assert rec["text_technical"] == "Revenue and units are strongly correlated."
        assert rec["text_plain"] == rec["text_technical"]
        assert rec["sources"] == []
        assert result["rag_sources"] == []

    def test_llm_mode_does_not_touch_vector_store(self, tmp_path) -> None:
        # No KB seeding should happen — vector_store.retrieve would fail
        # loudly if _ensure_kb_loaded() ran, since we never call initialize().
        vector_store = VectorStore(backend="faiss", persist_path=str(tmp_path / "vs"))
        fake = _FakeLLMClient()
        agent = RecommendationAgent(vector_store=vector_store, llm_client=fake)

        agent.run(context={"goal": "Anything interesting here?", "mode": "llm"})
        assert fake.last_prompt is not None  # the LLM path actually ran

    def test_llm_mode_includes_mining_features_in_prompt(self, tmp_path) -> None:
        vector_store = VectorStore(backend="faiss", persist_path=str(tmp_path / "vs"))
        fake = _FakeLLMClient()
        agent = RecommendationAgent(vector_store=vector_store, llm_client=fake)

        context = {
            "goal": "What should I clean up first?",
            "mode": "llm",
            "MiningAgent_output": {
                "statistics": {"revenue": {"type": "numeric", "mean": 100.0, "min": 0.0, "max": 500.0}},
                "data_quality": {"revenue": {"completeness_pct": 80.0}},
                "patterns": ["3 outlier(s) detected in 'revenue' via IQR."],
            },
            "IngestionAgent_output": {"warnings": ["Column 'id' is constant."]},
        }
        agent.run(context=context)

        assert "revenue" in fake.last_prompt
        assert "20.0% missing" in fake.last_prompt
        assert "outlier(s) detected in 'revenue'" in fake.last_prompt
        assert "Column 'id' is constant." in fake.last_prompt
        assert "What should I clean up first?" in fake.last_prompt

    def test_llm_mode_unconfigured_client_returns_graceful_message(self, tmp_path) -> None:
        vector_store = VectorStore(backend="faiss", persist_path=str(tmp_path / "vs"))
        fake = _FakeLLMClient(configured=False)
        agent = RecommendationAgent(vector_store=vector_store, llm_client=fake)

        result = agent.run(context={"goal": "Anything interesting?", "mode": "llm"})

        assert len(result["recommendations"]) == 1
        assert "API key" in result["recommendations"][0]["text_technical"]
        assert result["recommendations"][0]["confidence"] == 0.0

    def test_llm_mode_failed_request_returns_graceful_message_not_raise(self, tmp_path) -> None:
        vector_store = VectorStore(backend="faiss", persist_path=str(tmp_path / "vs"))
        fake = _FakeLLMClient(error=LLMError("Gemini returned HTTP 429"))
        agent = RecommendationAgent(vector_store=vector_store, llm_client=fake)

        result = agent.run(context={"goal": "Anything interesting?", "mode": "llm"})

        assert len(result["recommendations"]) == 1
        assert "failed" in result["recommendations"][0]["text_technical"].lower()

    def test_default_mode_is_unaffected_by_llm_client_presence(self, tmp_path) -> None:
        # Regression guard: injecting an llm_client must not change RAG-mode
        # (or mode-absent) behavior at all.
        vector_store = VectorStore(backend="faiss", persist_path=str(tmp_path / "vs"))
        fake = _FakeLLMClient()
        agent = RecommendationAgent(vector_store=vector_store, llm_client=fake)

        result = agent.run(context={"goal": "How should I handle missing values in my dataset?"})

        assert fake.last_prompt is None  # LLM never called
        assert len(result["recommendations"]) > 0
        assert any("missing_values" in source for source in result["rag_sources"])
