"""
tests/backend/test_export_trail.py
───────────────────────────────────
Tests for the Agent Execution Trail in the PDF export (FR-06), and for the
XML escaping that section depends on.

FR-06 claims every agent step is logged and inspectable. That was true of the
JSON export and of the API, but not of the PDF — the artefact a reader
actually opens. These tests assert the trail is present, complete, ordered,
and carries the reasoning and observation rather than just agent names.

The escaping tests cover a live bug found while building this: ReportLab's
``Paragraph`` parses its input as mini-XML, so a goal containing ``<b `` used
to fail the entire export with a 500, and ``<revenue>`` was silently swallowed
as an unknown tag. Both are reachable from anything a user can type, and from
any column name in their data, since agent observations quote column names
back.

These run against the pure ``export_service`` rather than over HTTP, so they
need no database — the HTTP path is already covered in ``test_export.py``.
"""

from __future__ import annotations

import io
import os
import sys

import pdfplumber
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.services import export_service

STEPS = [
    {
        "agent_name": "IngestionAgent",
        "action": "run IngestionAgent",
        "reasoning": "Ingest and profile the dataset into a canonical schema.",
        "observation": "Ingested 400 rows x 8 columns.",
        "status": "success",
        "latency_ms": 12,
    },
    {
        "agent_name": "MiningAgent",
        "action": "run MiningAgent",
        "reasoning": "Clustering goal, so run standardize, kmeans, dbscan, silhouette.",
        "observation": "Profiled 400 rows across 8 columns.",
        "status": "success",
        "latency_ms": 540,
    },
    {
        "agent_name": "VisualizationAgent",
        "action": "run VisualizationAgent",
        "reasoning": "Generate clustering-appropriate charts.",
        "observation": "Selected 1 goal-conditioned chart(s).",
        "status": "success",
        "latency_ms": 30,
    },
    {
        "agent_name": "RecommendationAgent",
        "action": "run RecommendationAgent",
        "reasoning": "Ground findings in retrievable methodology.",
        "observation": "Produced 3 grounded recommendation(s).",
        "status": "success",
        "latency_ms": 210,
    },
]


class FakeRun:
    """Minimal stand-in for a persisted AnalysisRun — export never re-computes."""

    def __init__(self, **overrides) -> None:
        self.id = "run-abc123"
        self.goal = "Find natural clusters and segments"
        self.summary = "Goal classified as 'clustering'."
        self.expertise_level = "expert"
        self.steps = list(STEPS)
        self.recommendations = ["Use silhouette score to select k."]
        self.rag_sources = ["clustering.md"]
        for key, value in overrides.items():
            setattr(self, key, value)


def _pdf_text(run: FakeRun) -> str:
    """
    Extracted PDF text with all whitespace collapsed to single spaces.

    Table cells wrap, so a phrase like "no numeric columns" comes back split
    across lines. Flattening keeps the assertions about *content* from being
    coupled to the column widths that decide where the text happens to break.
    """
    with pdfplumber.open(io.BytesIO(export_service.generate_pdf(run))) as pdf:
        raw = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return " ".join(raw.split())


@pytest.fixture(scope="module")
def trail_text() -> str:
    """The PDF text from the trail heading onward."""
    text = _pdf_text(FakeRun())
    assert "Agent Execution Trail" in text
    return text[text.index("Agent Execution Trail"):]


class TestTrailIsPresent:
    def test_the_pdf_has_a_trail_section(self) -> None:
        assert "Agent Execution Trail" in _pdf_text(FakeRun())

    def test_the_table_has_the_expected_columns(self, trail_text: str) -> None:
        for column in ("Agent", "Reasoning", "Observation", "Status", "Latency"):
            assert column in trail_text

    def test_every_agent_appears(self, trail_text: str) -> None:
        # The redundant "Agent" suffix is trimmed — the column is headed "Agent"
        # and the full name wraps mid-word in that column width.
        for agent in ("Ingestion", "Mining", "Visualization", "Recommendation"):
            assert agent in trail_text

    def test_reasoning_is_included_not_just_agent_names(self, trail_text: str) -> None:
        """A list of agent names is not an explainability trail."""
        assert "canonical schema" in trail_text
        assert "clustering-appropriate charts" in trail_text

    def test_observations_are_included(self, trail_text: str) -> None:
        # Probes are kept short deliberately. pdfplumber extracts a table
        # row-wise across columns, so a cell that wraps has its continuation
        # emitted after the later columns of the same row — a longer phrase
        # would fail on layout rather than on content.
        assert "400 rows" in trail_text
        assert "grounded" in trail_text

    def test_latencies_are_included(self, trail_text: str) -> None:
        assert "540 ms" in trail_text

    def test_total_agent_time_is_summarised(self, trail_text: str) -> None:
        assert f"{12 + 540 + 30 + 210} ms total" in trail_text

    def test_step_count_is_reported(self, trail_text: str) -> None:
        assert "4 step(s)" in trail_text

    def test_steps_appear_in_execution_order(self, trail_text: str) -> None:
        positions = [
            trail_text.index(agent)
            for agent in ("Ingestion", "Mining", "Visualization", "Recommendation")
        ]
        assert positions == sorted(positions)


class TestTrailDegradesSafely:
    def test_a_run_with_no_steps_omits_the_section(self) -> None:
        """Better an absent section than an empty table with a promising heading."""
        assert "Agent Execution Trail" not in _pdf_text(FakeRun(steps=[]))

    def test_a_run_with_no_steps_still_exports(self) -> None:
        assert export_service.generate_pdf(FakeRun(steps=[])).startswith(b"%PDF")

    def test_missing_step_fields_do_not_break_the_export(self) -> None:
        pdf = export_service.generate_pdf(FakeRun(steps=[{"agent_name": "MiningAgent"}]))
        assert pdf.startswith(b"%PDF")

    def test_an_errored_step_is_still_shown(self) -> None:
        """A failed step is exactly what someone inspecting the trail is looking for."""
        text = _pdf_text(
            FakeRun(steps=[{
                "agent_name": "MiningAgent",
                "reasoning": "Attempt the profile.",
                "observation": "MiningAgent failed: no numeric columns.",
                "status": "error",
                "latency_ms": 3,
            }])
        )
        assert "error" in text
        assert "no numeric" in text

    def test_a_long_trail_paginates_without_error(self) -> None:
        pdf = export_service.generate_pdf(FakeRun(steps=STEPS * 15))
        assert pdf.startswith(b"%PDF")


class TestXmlEscaping:
    """
    Regression tests for a live bug: unescaped user text reaching a ReportLab
    Paragraph either crashed the export or silently deleted content.
    """

    def test_goal_with_an_unclosed_tag_does_not_crash(self) -> None:
        """Previously raised 'paraparser: syntax error' and 500'd the endpoint."""
        pdf = export_service.generate_pdf(FakeRun(goal="Compare a <b vs c"))
        assert pdf.startswith(b"%PDF")

    def test_goal_with_angle_brackets_is_preserved_not_swallowed(self) -> None:
        """`<revenue>` used to be parsed as an unknown tag and vanish."""
        text = _pdf_text(FakeRun(goal="Analyse <revenue> by region"))
        assert "<revenue>" in text

    def test_ampersand_in_goal_survives(self) -> None:
        assert "R&D spend" in _pdf_text(FakeRun(goal="Break down R&D spend"))

    def test_unsafe_summary_does_not_crash(self) -> None:
        assert export_service.generate_pdf(FakeRun(summary="a < b & c > d")).startswith(b"%PDF")

    def test_unsafe_recommendation_does_not_crash(self) -> None:
        run = FakeRun(recommendations=["Drop rows where <threshold is unmet"])
        assert export_service.generate_pdf(run).startswith(b"%PDF")

    def test_unsafe_observation_in_a_step_does_not_crash(self) -> None:
        """Observations quote column names back, so user data reaches this path."""
        run = FakeRun(steps=[{
            "agent_name": "MiningAgent",
            "reasoning": "Profile the columns.",
            "observation": "Column '<qty' is constant (only one unique value).",
            "status": "success",
            "latency_ms": 5,
        }])
        assert export_service.generate_pdf(run).startswith(b"%PDF")

    def test_escaped_column_name_is_rendered_verbatim(self) -> None:
        run = FakeRun(steps=[{
            "agent_name": "MiningAgent",
            "reasoning": "Profile the columns.",
            "observation": "Column 'a<b' has 3 missing values.",
            "status": "success",
            "latency_ms": 5,
        }])
        assert "a<b" in _pdf_text(run)
