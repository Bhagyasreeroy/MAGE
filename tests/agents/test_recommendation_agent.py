"""Tests for agents/recommendation_agent.py."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.recommendation_agent import RecommendationAgent
from rag.vector_store import VectorStore


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
