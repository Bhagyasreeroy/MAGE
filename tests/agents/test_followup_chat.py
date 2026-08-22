"""
tests/agents/test_followup_chat.py
───────────────────────────────────
Tests that a follow-up question is answered, rather than the opening report
repeated.

The bug these cover: every chat turn re-ran the full pipeline, and the
recommendations were built one per MiningAgent finding. Findings are a property
of the *dataset*, and a real dataset yields more than the five slots available,
so the goal-led pass never ran and the question reached retrieval only as a
prefix on a finding query. Ask anything at all as a second question and the
answer came back identical to the first.

The fix has three parts, and the tests are grouped by them: a slot budget that
reserves room for the question, relevance filtering so an unrelated finding
does not take a slot, and holding back documents already shown.

Two guards run alongside. The opening run must be **unchanged** — it was not
broken, and the budget is deliberately a no-op there. And a finding must never
be paired with guidance it has nothing to do with, which is the failure mode
the filtering itself can cause if it is made to always return something.

Written test-first.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.recommendation_agent import (
    FOLLOWUP_BUDGET,
    MAX_RECOMMENDATIONS,
    RecommendationAgent,
)
from agents.recommendation_agent import _columns_named as _named

# A realistic mining output: more findings than there are recommendation slots,
# which is the condition that made every turn identical.
PATTERNS = [
    "'units' and 'revenue' are strongly positively correlated (r=0.91).",
    "3 outlier(s) detected in 'revenue' via IQR (2.0% of rows).",
    "'signup_days' is strongly skewed (skew=2.41).",
    "Data separates into 3 clusters (silhouette=0.52).",
    "'plan_tier' has 4 distinct categories.",
    "'revenue' has 42 missing values (8.4% of rows).",
]
MINING_OUTPUT = {"patterns": PATTERNS, "statistics": {}, "data_quality": {}}

OPENING_GOAL = "Analyse this dataset and tell me what stands out"

# Deliberately not a shape QAAgent recognises — these have to reach the RAG
# path, which is where the bug lived.
MISSING_VALUES_QUESTION = "What is the right way to treat absent readings before modelling?"
CLUSTER_CHART_QUESTION = "Which chart should I use to compare the clusters?"


@pytest.fixture(scope="module")
def agent() -> RecommendationAgent:
    return RecommendationAgent()


def _context(goal: str, conversation=None, **extra) -> dict:
    context = {"goal": goal, "MiningAgent_output": dict(MINING_OUTPUT), **extra}
    if conversation is not None:
        context["conversation"] = conversation
    return context


def _history(sources: list[str] | None = None) -> list[dict]:
    return [
        {"role": "user", "content": OPENING_GOAL},
        {
            "role": "assistant",
            "content": "Here is what stands out in the data.",
            "sources": sources if sources is not None else [],
        },
    ]


def _sources(result: dict) -> list[str]:
    return [rec["sources"][0] for rec in result["recommendations"]]


def _findings(result: dict) -> list[str]:
    return [rec["finding"] for rec in result["recommendations"] if rec["finding"]]


class TestTheOpeningRunIsUnchanged:
    """The first run was never the problem. The budget makes it a no-op, and
    that has to stay true or this fix trades one regression for another."""

    def test_findings_still_lead_every_recommendation(self, agent: RecommendationAgent) -> None:
        result = agent.run(_context(OPENING_GOAL))
        assert len(result["recommendations"]) == MAX_RECOMMENDATIONS
        assert all(rec["finding"] for rec in result["recommendations"])

    def test_findings_keep_mining_order(self, agent: RecommendationAgent) -> None:
        """MiningAgent's order is deliberate. With no question to rank against,
        reordering would trade it for embedding noise."""
        result = agent.run(_context(OPENING_GOAL))
        assert _findings(result) == PATTERNS[: len(_findings(result))]

    def test_an_empty_conversation_is_the_same_as_none(self, agent: RecommendationAgent) -> None:
        assert _sources(agent.run(_context(OPENING_GOAL, conversation=[]))) == _sources(
            agent.run(_context(OPENING_GOAL))
        )


class TestAFollowUpIsAnswered:
    def test_a_different_question_gets_a_different_answer(self, agent: RecommendationAgent) -> None:
        """The bug, stated directly."""
        opening = agent.run(_context(OPENING_GOAL))
        follow_up = agent.run(_context(MISSING_VALUES_QUESTION, conversation=_history()))
        assert _sources(follow_up) != _sources(opening)

    def test_two_different_follow_ups_differ_from_each_other(
        self, agent: RecommendationAgent
    ) -> None:
        first = agent.run(_context(MISSING_VALUES_QUESTION, conversation=_history()))
        second = agent.run(_context(CLUSTER_CHART_QUESTION, conversation=_history()))
        assert _sources(first) != _sources(second)

    def test_the_answer_is_grounded_in_the_topic_asked_about(
        self, agent: RecommendationAgent
    ) -> None:
        result = agent.run(_context(MISSING_VALUES_QUESTION, conversation=_history()))
        assert any("missing_values" in source for source in _sources(result))

    def test_some_of_the_answer_is_retrieved_for_the_question_alone(
        self, agent: RecommendationAgent
    ) -> None:
        """Goal-led cards carry no finding. Their presence is what proves the
        question reached retrieval on its own, rather than only as a prefix on
        a finding query."""
        result = agent.run(_context(CLUSTER_CHART_QUESTION, conversation=_history()))
        assert any(rec["finding"] is None for rec in result["recommendations"])

    def test_findings_never_take_more_than_their_budget(
        self, agent: RecommendationAgent
    ) -> None:
        result = agent.run(_context(CLUSTER_CHART_QUESTION, conversation=_history()))
        assert len(_findings(result)) <= FOLLOWUP_BUDGET.pattern_led


class TestFindingsAreRankedAgainstTheQuestion:
    def test_the_relevant_finding_leads(self, agent: RecommendationAgent) -> None:
        """Asked about clusters, the user should not be handed a correlation
        card first merely because correlations are computed first."""
        result = agent.run(_context(CLUSTER_CHART_QUESTION, conversation=_history()))
        assert "clusters" in _findings(result)[0]

    def test_an_unrelated_finding_is_not_paired_with_the_guidance(
        self, agent: RecommendationAgent
    ) -> None:
        """The worst output this agent can produce: a real statistic about one
        thing presented as the reason for guidance about another, which reads
        as a causal claim and is not one. Ranking that always returns its
        best-scoring finding produces exactly this."""
        result = agent.run(_context(MISSING_VALUES_QUESTION, conversation=_history()))
        for rec in result["recommendations"]:
            if rec["finding"] and "missing_values" in rec["sources"][0]:
                assert "missing values" in rec["finding"]

    def test_a_question_only_the_goal_bears_on_is_still_answered(
        self, agent: RecommendationAgent
    ) -> None:
        """Filtering out every finding must not empty the turn — the reserved
        goal-led slots carry it, as long as the question is one the knowledge
        base can actually speak to."""
        result = agent.run(
            _context("How do I choose between mean and median imputation?", conversation=_history())
        )
        assert result["recommendations"]


class TestAlreadyCitedDocumentsAreHeldBack:
    def test_a_follow_up_prefers_material_not_yet_shown(
        self, agent: RecommendationAgent
    ) -> None:
        shown = _sources(agent.run(_context(OPENING_GOAL)))
        result = agent.run(_context(CLUSTER_CHART_QUESTION, conversation=_history(shown)))
        goal_led = [rec["sources"][0] for rec in result["recommendations"] if rec["finding"] is None]
        assert goal_led
        assert not set(goal_led) & set(shown)

    def test_a_finding_still_gets_its_own_document_back(
        self, agent: RecommendationAgent
    ) -> None:
        """Holding back a document the finding-led pass needs either drops a
        relevant finding outright or pairs it with a worse document. Asked
        about a correlation the opening report already cited, the run must
        still return its correlation card."""
        question = "Tell me more about the relationship between units and revenue"
        result = agent.run(
            _context(question, conversation=_history(["knowledge_base/correlation_analysis.md"]))
        )
        assert any("correlated" in finding for finding in _findings(result))

    def test_repeats_beat_an_empty_answer(self, agent: RecommendationAgent) -> None:
        """If the only document that answers the question has been shown
        already, saying it again is better than saying nothing."""
        question = "How should I handle the missing values before modelling?"
        answerable = _sources(agent.run(_context(question, conversation=_history())))
        assert answerable, "precondition: the knowledge base can answer this"

        result = agent.run(_context(question, conversation=_history(answerable)))
        assert result["recommendations"]


class TestResolvedGoalsDriveRetrieval:
    def test_a_resolved_goal_is_what_gets_retrieved_against(
        self, agent: RecommendationAgent
    ) -> None:
        """"Why?" on its own retrieves nothing useful. The orchestrator resolves
        it and passes the result as `resolved_goal`; this checks the agent
        reads it."""
        bare = agent.run(_context("Why?", conversation=_history()))
        resolved = agent.run(
            _context(
                "Why?",
                conversation=_history(),
                resolved_goal="Why? Which chart should I use to compare the clusters?",
            )
        )
        assert _sources(bare) != _sources(resolved)

    def test_the_query_builder_prefers_the_resolved_goal(
        self, agent: RecommendationAgent
    ) -> None:
        query = agent._build_query(  # noqa: SLF001 - the query is the integration point
            {"goal": "Why?", "resolved_goal": "Why? Find clusters"}
        )
        assert query.startswith("Why? Find clusters")


class TestMalformedConversationsAreHarmless:
    """Client-supplied, and additive grounding. A bad payload costs a little
    answer quality, never the run."""

    @pytest.mark.parametrize(
        "conversation",
        ["not a list", 42, [None, "nope"], [{"role": "wizard", "content": "x"}], [{}]],
    )
    def test_a_bad_payload_still_produces_an_answer(
        self, agent: RecommendationAgent, conversation
    ) -> None:
        result = agent.run(_context(MISSING_VALUES_QUESTION, conversation=conversation))
        assert result["recommendations"]
        assert all(rec["sources"] for rec in result["recommendations"])

    def test_an_unreadable_conversation_falls_back_to_opening_behaviour(
        self, agent: RecommendationAgent
    ) -> None:
        """Nothing normalises out of it, so there is no conversation — and the
        turn should behave exactly like a first run, not like a broken one."""
        assert _sources(agent.run(_context(OPENING_GOAL, conversation=["junk"]))) == _sources(
            agent.run(_context(OPENING_GOAL))
        )


class TestAQuestionTheKnowledgeBaseCannotAnswer:
    """
    Reported from the running app. With a Spotify dataset loaded, "tell me
    about drake artist" came back with a card about SHAP and LIME.

    The knowledge base is EDA *methodology*. It has nothing to say about a
    particular artist, and retrieval says so plainly — the best document scores
    ~0.10 against that question, well under `MIN_CONFIDENCE`. Both the
    finding-led and goal-led passes correctly produced nothing. The broad pass
    then padded the turn: its query is mostly MiningAgent's findings, so it
    scores well regardless of what was asked, and it returned a 0.28 match that
    was presented as the answer.

    The turn must now come back empty, so the caller can say it could not
    answer instead of implying it did.
    """

    # No finding in this mining output relates to a named entity, which is the
    # real shape of the failure: the question is about the data's *content*,
    # and everything computed is about its *shape*.
    UNANSWERABLE = [
        "tell me about drake artist",
        "which country has the most artists",
        "list the artists from Colombia",
        "how many streams does taylor swift have",
    ]

    @pytest.mark.parametrize("question", UNANSWERABLE)
    def test_it_returns_nothing_rather_than_the_nearest_document(
        self, agent: RecommendationAgent, question: str
    ) -> None:
        result = agent.run(_context(question, conversation=_history()))
        assert result["recommendations"] == []
        assert result["rag_sources"] == []

    def test_a_borderline_question_may_still_surface_a_real_finding(
        self, agent: RecommendationAgent
    ) -> None:
        """The boundary is a measured threshold, not a classifier, and this
        records where it actually falls rather than pretending it is sharper
        than it is.

        "who had the most streams in 2013" scores 0.246 for retrieval — under
        `GOAL_LED_MIN_CONFIDENCE`, so it earns no card claiming to answer it —
        but 0.204 against an outlier finding, just over
        `PATTERN_RELEVANCE_FLOOR`. What comes back is therefore a statistic
        computed from the user's own data, not a methodology document dressed
        up as an answer. Still not what was asked; a different kind of wrong,
        and the tolerable kind.
        """
        result = agent.run(_context("who had the most streams in 2013", conversation=_history()))
        assert all(rec["finding"] for rec in result["recommendations"]), (
            "nothing may claim to answer the question directly"
        )

    def test_the_opening_report_still_fills_its_slots(
        self, agent: RecommendationAgent
    ) -> None:
        """The broad pass is switched off for follow-ups only. An opening run
        with too few usable findings must still pad, or this fix quietly
        shortens every report."""
        sparse = {"patterns": ["'units' and 'revenue' are strongly positively correlated (r=0.91)."],
                  "statistics": {}, "data_quality": {}}
        result = agent.run({"goal": OPENING_GOAL, "MiningAgent_output": sparse})
        assert len(result["recommendations"]) > 1
        assert any(rec["finding"] is None for rec in result["recommendations"])


class TestAColumnNamedInTheQuestionWins:
    """
    Found by probing the running app against a Spotify dataset.

    Asked "which artists are the outliers in Total Streams?", the run led with
    the outliers in *Collaborative Streams*. Asked whether collaborating causes
    more total streams, it led with a correlation between two other columns.
    Cosine similarity cannot separate "Total Streams (in millions)" from
    "Collaborative Streams (in millions)" — as sentences they are nearly
    identical — so a named column is applied on top of the ranking rather than
    mixed into it.
    """

    COLUMNS = [
        "Artist Name",
        "Debut Year",
        "Total Streams (in millions)",
        "Lead Streams (in millions)",
        "Collaborative Streams (in millions)",
        "% of Solo Streams",
    ]
    FINDINGS = [
        "'Feature Streams (in millions)' and '% of Collaborative Streams' are strongly positively correlated (r=0.69).",
        "2 outlier(s) detected in 'Collaborative Streams (in millions)' via IQR (5.7% of rows).",
        "3 outlier(s) detected in 'Total Streams (in millions)' via IQR (8.6% of rows).",
        "'Total Streams (in millions)' and 'Collaborative Streams (in millions)' are strongly positively correlated (r=0.61).",
        "'Debut Year' is strongly skewed (skew=-1.28).",
    ]

    def _ranked(self, agent: RecommendationAgent, question: str) -> list[str]:
        return agent._patterns_for_goal(question, self.FINDINGS, self.COLUMNS)  # noqa: SLF001

    def test_the_named_columns_finding_leads(self, agent: RecommendationAgent) -> None:
        first = self._ranked(agent, "Which artists are the outliers in Total Streams?")[0]
        assert "Total Streams" in first
        assert "outlier" in first

    def test_a_near_identical_column_name_does_not_win(self, agent: RecommendationAgent) -> None:
        """The exact confusion that caused it — both findings are about
        outliers, and only one is about the column that was named."""
        ranked = self._ranked(agent, "Which artists are the outliers in Total Streams?")
        collaborative = next(i for i, p in enumerate(ranked) if "Collaborative Streams" in p and "outlier" in p)
        total = next(i for i, p in enumerate(ranked) if "Total Streams" in p and "outlier" in p)
        assert total < collaborative

    def test_a_column_named_in_lower_case_still_counts(self, agent: RecommendationAgent) -> None:
        """Nobody types "Total Streams (in millions)"."""
        first = self._ranked(agent, "does collaborating more cause more total streams?")[0]
        assert "Total Streams" in first

    def test_a_merely_similar_word_does_not_name_a_column(self, agent: RecommendationAgent) -> None:
        """"collaborating" is not "Collaborative Streams". Treating it as one
        would put the preference back where it started."""
        assert not _named("does collaborating help?", self.COLUMNS)

    def test_naming_no_column_leaves_embedding_order_alone(self, agent: RecommendationAgent) -> None:
        question = "How should I handle the missing values?"
        assert self._ranked(agent, question) == agent._patterns_for_goal(  # noqa: SLF001
            question, self.FINDINGS, []
        )

    def test_a_short_column_name_is_not_matched_against_prose(self) -> None:
        """A two-letter column would otherwise match half of every sentence."""
        assert not _named("what is the id of this run", ["id"])
