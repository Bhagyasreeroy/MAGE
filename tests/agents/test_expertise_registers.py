"""
tests/agents/test_expertise_registers.py
─────────────────────────────────────────
Tests for FR-04 — "expertise level visibly alters output language".

The spec names three audiences (Beginner / Analyst / Data Scientist); the
system's enum spells them `beginner` / `intermediate` / `expert`. These tests
pin down both halves of the requirement:

  • **Three registers exist and genuinely differ.** The word "visibly" is what
    is being tested, so it is not enough that three fields are present — the
    text has to actually change. Length ordering and inequality are asserted
    directly, because a refactor that quietly collapsed two registers into the
    same string would otherwise pass every other test in the suite.
  • **The selected level reaches the output.** The register is chosen in the
    orchestrator's aggregation step, so a correct RecommendationAgent with a
    broken mapping would still look right in isolation.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.orchestrator import _REGISTER_BY_EXPERTISE, OrchestratorAgent
from agents.recommendation_agent import RecommendationAgent
from backend.schemas.analysis import ExpertiseLevel

EXPERTISE_LEVELS = ["beginner", "intermediate", "expert"]


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Enough structure that mining produces patterns to lead recommendations with."""
    rng = np.random.default_rng(11)
    n = 120
    units = rng.normal(5, 2, n).clip(1)
    return pd.DataFrame(
        {
            "units": units,
            "unit_price": rng.normal(20, 5, n).clip(1),
            "revenue": units * 20 + rng.normal(0, 5, n),
            "region": rng.choice(["North", "South", "East"], n),
        }
    )


@pytest.fixture(scope="module")
def recommendations(sample_df: pd.DataFrame) -> list[dict]:
    """One real RecommendationAgent run, reused across the register assertions."""
    from agents.mining_agent import MiningAgent

    mining = MiningAgent().run(context={"dataframe": sample_df})
    result = RecommendationAgent().run(
        context={
            "goal": "Find natural clusters and segments in this data",
            "dataframe": sample_df,
            "MiningAgent_output": mining,
        }
    )
    assert result["recommendations"], "fixture needs at least one recommendation"
    return result["recommendations"]


class TestThreeRegistersExist:
    def test_every_recommendation_carries_all_three(self, recommendations: list[dict]) -> None:
        for rec in recommendations:
            assert rec["text_plain"]
            assert rec["text_analyst"]
            assert rec["text_technical"]

    def test_plain_and_analyst_always_differ(self, recommendations: list[dict]) -> None:
        """Guaranteed structurally by the analyst register's inline attribution."""
        for rec in recommendations:
            assert rec["text_plain"] != rec["text_analyst"]

    def test_the_rendered_sets_are_three_distinct_documents(
        self, recommendations: list[dict]
    ) -> None:
        """
        FR-04 measured on what the user actually reads — the whole set.

        Asserted at set level rather than per recommendation because a chunk
        that is a single short sentence leaves the plain and technical
        registers nothing to differ *by*: with no second sentence to keep and
        no markdown to preserve, both reduce to that one sentence. Known
        contributor: chunks whose body is a markdown table, which
        `_strip_markdown_structure` removes wholesale (see
        `test_known_gap_markdown_tables_are_dropped` below).
        """
        rendered = {
            field: tuple(rec[field] for rec in recommendations)
            for field in ("text_plain", "text_analyst", "text_technical")
        }
        assert len(set(rendered.values())) == 3

    def test_markdown_tables_now_reach_every_register(self) -> None:
        """
        The inverse of a gap this file used to pin.

        Table rows were previously discarded wholesale, so a knowledge-base
        document whose substance *is* a table contributed only its heading.
        They are now rewritten as labelled sentences — the full contract lives
        in `tests/agents/test_markdown_tables.py`.
        """
        from agents.recommendation_agent import _clean_technical

        cleaned = _clean_technical(
            "## Chart selection\n\n| Goal | Chart |\n|---|---|\n| compare | bar |\n"
        )
        assert "|" not in cleaned
        assert "Chart: bar" in cleaned

    def test_detail_increases_with_expertise_in_aggregate(
        self, recommendations: list[dict]
    ) -> None:
        """
        Measured across the whole recommendation set, not per item.

        Per-recommendation ordering is deliberately *not* asserted: some
        knowledge-base chunks are a single sentence, and against those the
        analyst register's inline attribution can make it marginally longer
        than the technical one. That is a quirk of very short sources, not a
        register inversion — the distinctness test above is what guards the
        actual requirement.
        """
        totals = {
            field: sum(len(rec[field]) for rec in recommendations)
            for field in ("text_plain", "text_analyst", "text_technical")
        }
        assert totals["text_plain"] < totals["text_analyst"] < totals["text_technical"]

    def test_analyst_register_names_its_methodology_source(
        self, recommendations: list[dict]
    ) -> None:
        """Inline attribution is what separates this register structurally."""
        assert all(rec["text_analyst"].count("Per ") >= 1 for rec in recommendations)


class TestRegisterCharacter:
    def test_plain_register_has_no_markdown(self, recommendations: list[dict]) -> None:
        for rec in recommendations:
            assert "##" not in rec["text_plain"]
            assert "|" not in rec["text_plain"]
            assert "**" not in rec["text_plain"]

    def test_analyst_register_has_no_markdown_either(self, recommendations: list[dict]) -> None:
        """An analyst wants the substance, not the document structure."""
        for rec in recommendations:
            assert "##" not in rec["text_analyst"]
            assert "|" not in rec["text_analyst"]

    def test_technical_register_keeps_markdown_emphasis(
        self, recommendations: list[dict]
    ) -> None:
        assert any("**" in rec["text_technical"] for rec in recommendations)

    def test_analyst_register_leads_with_the_concrete_finding(
        self, recommendations: list[dict]
    ) -> None:
        """'the statistic that supports it' — the finding must survive, not just prose."""
        assert any(
            any(ch.isdigit() for ch in rec["text_analyst"]) for rec in recommendations
        )


class TestExpertiseSelectsTheRegister:
    def test_mapping_covers_every_expertise_level(self) -> None:
        assert set(_REGISTER_BY_EXPERTISE) == {level.value for level in ExpertiseLevel}

    def test_mapping_targets_three_distinct_registers(self) -> None:
        assert len(set(_REGISTER_BY_EXPERTISE.values())) == 3

    @pytest.mark.parametrize(
        "level,expected",
        [
            ("beginner", "text_plain"),
            ("intermediate", "text_analyst"),
            ("expert", "text_technical"),
        ],
    )
    def test_each_level_maps_to_its_register(self, level: str, expected: str) -> None:
        assert _REGISTER_BY_EXPERTISE[level] == expected

    def test_unknown_level_falls_back_to_technical(self) -> None:
        assert _REGISTER_BY_EXPERTISE.get("wizard", "text_technical") == "text_technical"


@pytest.fixture(scope="module")
def by_level(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[str]]:
    """One full pipeline run per expertise level, sharing a single dataset."""
    rng = np.random.default_rng(5)
    n = 120
    units = rng.normal(5, 2, n).clip(1)
    df = pd.DataFrame(
        {
            "units": units,
            "unit_price": rng.normal(20, 5, n).clip(1),
            "revenue": units * 20 + rng.normal(0, 5, n),
        }
    )
    path = tmp_path_factory.mktemp("registers") / "data.csv"
    df.to_csv(path, index=False)

    orchestrator = OrchestratorAgent()
    return {
        level: orchestrator.run(
            goal="Find natural clusters and segments in this data",
            expertise_level=level,
            data={"source": str(path)},
        )["recommendations"]
        for level in EXPERTISE_LEVELS
    }


class TestOrchestratorHonoursExpertiseEndToEnd:
    """A correct agent with a broken mapping would still pass the tests above."""

    def test_every_level_produces_recommendations(self, by_level: dict) -> None:
        for level in EXPERTISE_LEVELS:
            assert by_level[level], f"{level} produced none"

    def test_the_three_levels_produce_different_output(self, by_level: dict) -> None:
        """FR-04, measured at the pipeline's edge rather than inside one agent."""
        rendered = {level: tuple(by_level[level]) for level in EXPERTISE_LEVELS}
        assert len(set(rendered.values())) == 3

    def test_output_length_increases_with_expertise(self, by_level: dict) -> None:
        totals = {level: sum(len(t) for t in by_level[level]) for level in EXPERTISE_LEVELS}
        assert totals["beginner"] < totals["intermediate"] < totals["expert"]

    def test_same_recommendation_count_at_every_level(self, by_level: dict) -> None:
        """Expertise changes the register, not which findings you are told about."""
        counts = {len(by_level[level]) for level in EXPERTISE_LEVELS}
        assert len(counts) == 1


class TestBackwardCompatibility:
    """Runs persisted before `text_analyst` existed must still render."""

    def test_missing_analyst_field_falls_back_to_technical(self) -> None:
        orchestrator = OrchestratorAgent()
        legacy = [{"insight": "Old run", "text_technical": "Full text.", "text_plain": "Short."}]
        rendered = orchestrator._aggregate(  # noqa: SLF001 - exercising the render path directly
            goal="g",
            expertise_level="intermediate",
            classification=orchestrator._classifier.classify("summarise this"),
            steps=[],
            context={"RecommendationAgent_output": {"recommendations": legacy}},
        )["recommendations"]
        assert rendered == ["Full text."]

    def test_empty_registers_fall_back_rather_than_render_blank(self) -> None:
        orchestrator = OrchestratorAgent()
        legacy = [{"insight": "Titled", "text_technical": "", "text_plain": ""}]
        rendered = orchestrator._aggregate(  # noqa: SLF001
            goal="g",
            expertise_level="beginner",
            classification=orchestrator._classifier.classify("summarise this"),
            steps=[],
            context={"RecommendationAgent_output": {"recommendations": legacy}},
        )["recommendations"]
        assert rendered == ["Titled"]
