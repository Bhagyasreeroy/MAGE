"""
tests/agents/test_recommendation_cards.py
──────────────────────────────────────────
The structured form of a recommendation, and its trip out through the
orchestrator's response.

`AnalysisResponse.recommendations` is a `list[str]`: one register's prose per
recommendation, with the source document's title, the retrieval confidence, the
citation, and the boundary between "what we found in your data" and "what the
knowledge base says about it" all collapsed into a single line. The frontend
had nothing left to lay out, which is why a grounded answer rendered as a wall
of text next to the LLM's structured one.

These tests pin the structured form that fixes it. The flattened list stays
exactly as it was — exports, history persistence and every existing test read
it — so this is additive.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.orchestrator import OrchestratorAgent
from agents.recommendation_agent import RecommendationAgent
from backend.schemas.analysis import GoalClassification, TaskType


@pytest.fixture
def agent(tmp_path) -> RecommendationAgent:
    return RecommendationAgent()


class TestAgentEmitsTheParts:
    """
    The agent already knew the finding and the guidance separately — it just
    concatenated them on the way out. These assert the parts survive.
    """

    def test_a_data_grounded_recommendation_separates_finding_from_guidance(
        self, agent: RecommendationAgent,
    ) -> None:
        pattern = "Data separates into 2 clusters (silhouette score=0.481)."
        result = agent.run(context={
            "goal": "Find natural segments among our customers",
            "MiningAgent_output": {"patterns": [pattern]},
        })
        rec = result["recommendations"][0]

        assert rec["finding"] == pattern
        for key in ("guidance_technical", "guidance_analyst", "guidance_plain"):
            assert rec[key], f"{key} must carry the knowledge-base prose"
            assert pattern not in rec[key], (
                f"{key} must be guidance alone — the finding is a separate field, "
                "or the frontend cannot lay them out differently"
            )

    def test_a_goal_only_recommendation_has_no_finding(self, agent: RecommendationAgent) -> None:
        """
        With no dataset patterns the agent falls back to goal-only retrieval —
        the path the follow-up chat uses. There is no finding to lead with, and
        the card must say so rather than inventing one.
        """
        result = agent.run(context={"goal": "How should I handle missing values?"})
        rec = result["recommendations"][0]

        assert rec["finding"] is None
        assert rec["guidance_technical"]

    def test_the_flattened_registers_are_unchanged(self, agent: RecommendationAgent) -> None:
        """The existing contract still holds: exports and history read these."""
        pattern = "Data separates into 2 clusters (silhouette score=0.481)."
        result = agent.run(context={
            "goal": "Find natural segments among our customers",
            "MiningAgent_output": {"patterns": [pattern]},
        })
        rec = result["recommendations"][0]

        assert rec["text_technical"].startswith(f"**{pattern}**")
        assert rec["text_analyst"].startswith(pattern)
        assert rec["text_plain"].startswith(pattern)


class TestCardsReachTheResponse:
    def _aggregate(self, expertise: str, structured: list[dict]) -> dict:
        orch = OrchestratorAgent()
        classification = GoalClassification(
            task_type=TaskType.clustering, confidence=0.9, rationale="test", target_column=None,
        )
        return orch._aggregate(
            goal="Find natural segments",
            expertise_level=expertise,
            mode="rag",
            classification=classification,
            steps=[],
            context={"RecommendationAgent_output": {
                "recommendations": structured,
                "rag_sources": ["knowledge_base/clustering.md"],
            }},
        )

    @staticmethod
    def _structured() -> list[dict]:
        return [{
            "insight": "Clustering Method Selection",
            "finding": "Data separates into 2 clusters (silhouette score=0.481).",
            "guidance_technical": "DBSCAN is best when the cluster count is unknown.",
            "guidance_analyst": "Analyst-register guidance.",
            "guidance_plain": "Plain-register guidance.",
            "text_technical": "**Data separates into 2 clusters (silhouette score=0.481).**\n\nDBSCAN is best when the cluster count is unknown.",
            "text_analyst": "Data separates into 2 clusters (silhouette score=0.481). Analyst-register guidance.",
            "text_plain": "Data separates into 2 clusters (silhouette score=0.481). Plain-register guidance.",
            "confidence": 0.812,
            "sources": ["knowledge_base/clustering.md"],
        }]

    def test_the_response_carries_a_card_per_recommendation(self) -> None:
        result = self._aggregate("expert", self._structured())
        cards = result["recommendation_cards"]

        assert len(cards) == 1
        card = cards[0]
        assert card["insight"] == "Clustering Method Selection"
        assert card["finding"] == "Data separates into 2 clusters (silhouette score=0.481)."
        assert card["confidence"] == 0.812
        assert card["sources"] == ["knowledge_base/clustering.md"]

    @pytest.mark.parametrize(
        ("expertise", "expected"),
        [
            ("expert", "DBSCAN is best when the cluster count is unknown."),
            ("intermediate", "Analyst-register guidance."),
            ("beginner", "Plain-register guidance."),
        ],
    )
    def test_the_card_guidance_follows_the_reader_s_register(self, expertise: str, expected: str) -> None:
        """FR-04 governs the card exactly as it governs the flat list."""
        result = self._aggregate(expertise, self._structured())
        assert result["recommendation_cards"][0]["guidance"] == expected

    def test_the_card_does_not_repeat_its_own_title_in_the_guidance(self) -> None:
        """
        The analyst register names its methodology inline — "Per Clustering
        Method Selection: …" — and that attribution is deliberate: FR-04 uses
        it to separate the analyst register from the plain one structurally
        rather than by length (see RecommendationAgent._analyst).

        On a card the attribution is already the heading, so the inline copy
        reads as a stutter. Strip it *for the card only*; `text_analyst` keeps
        it, because the flat list has no heading to carry it.
        """
        structured = self._structured()
        structured[0]["guidance_analyst"] = (
            "Per Clustering Method Selection: DBSCAN is best when the cluster count is unknown."
        )
        card = self._aggregate("intermediate", structured)["recommendation_cards"][0]

        assert card["insight"] == "Clustering Method Selection"
        assert card["guidance"] == "DBSCAN is best when the cluster count is unknown."

    def test_an_attribution_to_a_different_document_is_left_alone(self) -> None:
        """Only the stutter goes. A genuine cross-reference is information."""
        structured = self._structured()
        structured[0]["guidance_analyst"] = "Per Outlier Detection: use IQR here."
        card = self._aggregate("intermediate", structured)["recommendation_cards"][0]

        assert card["guidance"] == "Per Outlier Detection: use IQR here."

    def test_the_flat_list_is_still_produced(self) -> None:
        result = self._aggregate("expert", self._structured())
        assert result["recommendations"] == [self._structured()[0]["text_technical"]]

    def test_a_run_persisted_before_cards_existed_still_aggregates(self) -> None:
        """
        History replays old runs through this code. A stored recommendation has
        no `finding`/`guidance_*` keys, and must degrade to a usable card
        rather than raising.
        """
        legacy = [{
            "insight": "Clustering Method Selection",
            "text_technical": "**Old finding.**\n\nOld guidance.",
            "confidence": 0.5,
            "sources": ["knowledge_base/clustering.md"],
        }]
        cards = self._aggregate("expert", legacy)["recommendation_cards"]

        assert len(cards) == 1
        assert cards[0]["insight"] == "Clustering Method Selection"
        assert cards[0]["guidance"], "must fall back to the flattened text"
