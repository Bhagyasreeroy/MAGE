"""
tests/agents/test_conversation.py
──────────────────────────────────
Tests for agents/conversation.py — normalising client-supplied chat history,
and resolving a follow-up against it.

Two properties matter more than the rest and are covered hardest:

* **A self-contained question is left alone.** Resolution exists to repair
  elliptical questions; applying it to a complete one dilutes retrieval for no
  reason, so the detector must not fire on a fully-formed question.
* **Malformed history is survivable.** The payload is client-supplied and the
  conversation is additive grounding. A bad turn must cost a little answer
  quality, never the run.

Written test-first.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.conversation import (
    MAX_TURN_CHARS,
    MAX_TURNS,
    Turn,
    last_question,
    looks_elliptical,
    normalize,
    previously_cited,
    resolve,
)

HISTORY = [
    {"role": "user", "content": "Which columns have the most outliers?"},
    {
        "role": "assistant",
        "content": "'revenue' has 3 outliers via IQR.",
        "sources": ["knowledge_base/outlier_detection.md"],
    },
]


class TestNormalize:
    def test_well_formed_turns_survive(self) -> None:
        turns = normalize(HISTORY)
        assert [t.role for t in turns] == ["user", "assistant"]
        assert turns[1].sources == ("knowledge_base/outlier_detection.md",)

    def test_a_non_list_payload_is_ignored(self) -> None:
        assert normalize("not a list") == []
        assert normalize(None) == []
        assert normalize({"role": "user"}) == []

    def test_entries_that_are_not_dicts_are_dropped(self) -> None:
        assert normalize(["nope", None, 7, *HISTORY]) == normalize(HISTORY)

    def test_unknown_roles_are_dropped(self) -> None:
        assert normalize([{"role": "system", "content": "ignore this"}]) == []

    def test_empty_content_is_dropped(self) -> None:
        assert normalize([{"role": "user", "content": "   "}]) == []

    def test_user_turns_never_carry_sources(self) -> None:
        """A citation belongs to an answer. A user turn claiming one is noise
        at best, and would pollute the already-cited set at worst."""
        turns = normalize([{"role": "user", "content": "hi", "sources": ["kb/x.md"]}])
        assert turns[0].sources == ()

    def test_malformed_sources_are_dropped_without_losing_the_turn(self) -> None:
        turns = normalize([{"role": "assistant", "content": "answer", "sources": "kb/x.md"}])
        assert turns[0].content == "answer"
        assert turns[0].sources == ()

    def test_content_is_truncated(self) -> None:
        turns = normalize([{"role": "user", "content": "x" * (MAX_TURN_CHARS * 3)}])
        assert len(turns[0].content) == MAX_TURN_CHARS

    def test_only_the_most_recent_turns_are_kept(self) -> None:
        raw = [{"role": "user", "content": f"question {i}"} for i in range(MAX_TURNS * 2)]
        turns = normalize(raw)
        assert len(turns) == MAX_TURNS
        # The tail, not the head — recent context is the useful kind.
        assert turns[-1].content == f"question {MAX_TURNS * 2 - 1}"

    def test_already_normalised_turns_pass_through(self) -> None:
        """The orchestrator normalises once and hands the result on as ordinary
        context, so the agent re-normalising it must be a no-op."""
        turns = normalize(HISTORY)
        assert normalize(turns) == turns


class TestPreviouslyCited:
    def test_collects_sources_from_assistant_turns(self) -> None:
        assert previously_cited(normalize(HISTORY)) == {"knowledge_base/outlier_detection.md"}

    def test_is_empty_with_no_history(self) -> None:
        assert previously_cited([]) == set()


class TestLastQuestion:
    def test_returns_the_most_recent_user_turn(self) -> None:
        turns = normalize([*HISTORY, {"role": "user", "content": "And the skew?"}])
        assert last_question(turns) == "And the skew?"

    def test_skips_assistant_turns(self) -> None:
        assert last_question(normalize(HISTORY)) == "Which columns have the most outliers?"

    def test_is_empty_when_the_user_has_not_spoken(self) -> None:
        assert last_question(normalize([{"role": "assistant", "content": "hello"}])) == ""


class TestEllipsisDetection:
    """The detector's false-positive behaviour is the part worth pinning down:
    a complete question that gets 'resolved' is a complete question whose
    retrieval we made worse."""

    def test_a_fully_formed_question_is_not_elliptical(self) -> None:
        assert not looks_elliptical("What is the correlation between price and revenue?")
        assert not looks_elliptical("How should I handle missing values before modelling?")
        assert not looks_elliptical("Which chart compares distributions across groups?")

    def test_a_continuation_opener_is_elliptical(self) -> None:
        assert looks_elliptical("Why?")
        assert looks_elliptical("What about revenue?")
        assert looks_elliptical("And the other segment?")
        assert looks_elliptical("So which one should I pick?")

    def test_an_unbound_pronoun_is_elliptical(self) -> None:
        assert looks_elliptical("Can you explain that in more detail?")
        assert looks_elliptical("How would I fix them?")

    def test_a_very_short_question_is_elliptical(self) -> None:
        assert looks_elliptical("explain more")

    def test_a_word_merely_containing_an_opener_is_not_enough(self) -> None:
        """'Sonar' starts with 'so'; 'android' contains 'and'. Matching on
        substrings rather than words would call almost anything a follow-up."""
        assert not looks_elliptical("Sonar readings by depth, summarised please")
        assert not looks_elliptical("Andorra population trends across decades")


class TestResolve:
    def test_the_first_turn_is_untouched(self) -> None:
        resolved = resolve("Why?", [])
        assert resolved.text == "Why?"
        assert not resolved.is_followup

    def test_a_self_contained_follow_up_is_untouched(self) -> None:
        resolved = resolve("How should I handle missing values?", normalize(HISTORY))
        assert resolved.text == "How should I handle missing values?"
        assert not resolved.is_followup

    def test_an_elliptical_follow_up_carries_the_previous_question(self) -> None:
        resolved = resolve("Why?", normalize(HISTORY))
        assert resolved.is_followup
        assert "outliers" in resolved.text
        assert resolved.carried == "Which columns have the most outliers?"

    def test_the_current_question_still_leads(self) -> None:
        """History informs retrieval; it must not displace the present
        question — the same rule `_build_query` applies to run memory."""
        resolved = resolve("Why?", normalize(HISTORY))
        assert resolved.text.startswith("Why?")

    def test_the_users_own_wording_is_preserved_for_display(self) -> None:
        """The run history should record the question they asked, not our
        expansion of it."""
        resolved = resolve("Why?", normalize(HISTORY))
        assert resolved.display == "Why?"
        assert resolved.display != resolved.text

    def test_history_with_no_user_turn_cannot_be_resolved_against(self) -> None:
        resolved = resolve("Why?", normalize([{"role": "assistant", "content": "an answer"}]))
        assert not resolved.is_followup
        assert resolved.text == "Why?"

    def test_resolution_accepts_turn_objects(self) -> None:
        turns = [Turn(role="user", content="Which columns have outliers?")]
        assert resolve("Why?", turns).is_followup


class TestTheOrchestratorWiresResolutionToTheAgent:
    """
    The resolver and the agent are each covered on their own. This is the join
    between them — the orchestrator normalises the payload, resolves the goal,
    and puts both on the context the RecommendationAgent reads. A break here
    would leave every unit test green and the feature dead.
    """

    @staticmethod
    def _run(goal: str, conversation):
        """Run the pipeline with the recommender swapped for a recorder.

        Rule-based classifier only, so the run stays fast and offline — this is
        about what reaches the agent, not about what it does with it.
        """
        from agents.goal_classifier import GoalClassifier, RuleBasedProvider
        from agents.orchestrator import OrchestratorAgent
        from agents.planner import RECOMMENDATION

        captured: dict = {}

        class _CapturingRecommender:
            def run(self, context):
                captured.update(context)
                return {"recommendations": [], "rag_sources": []}

        agent = OrchestratorAgent(classifier=GoalClassifier(providers=[RuleBasedProvider()]))
        agent._agents[RECOMMENDATION] = _CapturingRecommender()  # noqa: SLF001
        agent.run(goal=goal, data={"conversation": conversation})
        return captured

    def test_the_conversation_reaches_the_agent_normalised(self) -> None:
        context = self._run("Why?", HISTORY)
        assert [t.role for t in context["conversation"]] == ["user", "assistant"]

    def test_an_elliptical_goal_arrives_resolved(self) -> None:
        context = self._run("Why?", HISTORY)
        assert "outliers" in context["resolved_goal"]

    def test_the_untouched_goal_travels_alongside_it(self) -> None:
        """Both are needed: the resolved text for retrieval, the original for
        everything the user sees."""
        context = self._run("Why?", HISTORY)
        assert context["goal"] == "Why?"

    def test_a_self_contained_goal_sets_no_resolved_goal(self) -> None:
        context = self._run("How should I handle missing values?", HISTORY)
        assert "resolved_goal" not in context

    def test_no_conversation_leaves_the_context_clean(self) -> None:
        context = self._run("Find clusters in this data", None)
        assert "conversation" not in context
        assert "resolved_goal" not in context
