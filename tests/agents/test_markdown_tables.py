"""
tests/agents/test_markdown_tables.py
─────────────────────────────────────
Tests for how knowledge-base markdown tables reach recommendation text.

Several knowledge-base documents carry their most useful content as a
*decision table* — the goal→chart mapping in `distribution_profiling.md`, the
correlation-method selector, the missingness strategy matrix. Those tables
were previously discarded wholesale, so the documents contributed only their
headings and the actual guidance never reached the user.

Dropping them was not arbitrary: inlining a raw table flattens it into an
unreadable run of cell text ("numeric Line chart ... Choropleth ..."), and
leaving the pipes in is worse. The contract below is the third option —
**each row is rebuilt as a labelled sentence using the header row as field
names**, so the information survives in a form that reads as prose.

Written test-first: these failed before `_flatten_markdown_tables` existed.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.recommendation_agent import _clean_technical, _strip_markdown_structure

CHART_TABLE = """## Chart selection by goal and data shape

| Goal | Data shape | Recommended chart |
|---|---|---|
| Understand single-column distribution | Numeric | Histogram + KDE overlay |
| Trend over time | Time-indexed numeric | Line chart, with rolling average |

Choose the chart that matches the question being asked.
"""


class TestTableContentSurvives:
    """The bug being fixed: table content used to vanish entirely."""

    def test_cell_values_are_not_discarded(self) -> None:
        out = _strip_markdown_structure(CHART_TABLE)
        assert "Histogram + KDE overlay" in out
        assert "Line chart, with rolling average" in out

    def test_row_subjects_survive(self) -> None:
        out = _strip_markdown_structure(CHART_TABLE)
        assert "Understand single-column distribution" in out
        assert "Trend over time" in out

    def test_the_goal_to_chart_table_reaches_the_technical_register(self) -> None:
        """The clearest real casualty of the old behaviour."""
        assert "Histogram" in _clean_technical(CHART_TABLE)


class TestTableIsReadableNotRawCells:
    """Preserving the table is only useful if the result reads as prose."""

    def test_no_pipe_characters_leak_into_the_text(self) -> None:
        assert "|" not in _strip_markdown_structure(CHART_TABLE)

    def test_separator_rows_never_appear(self) -> None:
        out = _strip_markdown_structure(CHART_TABLE)
        assert "---" not in out

    def test_each_value_is_labelled_by_its_column(self) -> None:
        """This is what stops a row collapsing into an unreadable run of cells."""
        out = _strip_markdown_structure(CHART_TABLE)
        assert "Recommended chart: Histogram + KDE overlay" in out

    def test_a_row_reads_as_one_sentence(self) -> None:
        out = _strip_markdown_structure(CHART_TABLE)
        assert (
            "Goal: Understand single-column distribution; "
            "Data shape: Numeric; "
            "Recommended chart: Histogram + KDE overlay."
        ) in out

    def test_the_header_row_is_used_as_labels_not_emitted_as_content(self) -> None:
        """A bare 'Goal Data shape Recommended chart' line would be noise."""
        out = _strip_markdown_structure(CHART_TABLE)
        assert "Goal Data shape Recommended chart" not in out


class TestSurroundingProseIsPreserved:
    def test_heading_text_survives(self) -> None:
        assert "Chart selection by goal and data shape" in _strip_markdown_structure(CHART_TABLE)

    def test_prose_after_the_table_survives(self) -> None:
        assert "Choose the chart that matches" in _strip_markdown_structure(CHART_TABLE)

    def test_text_without_tables_is_unaffected(self) -> None:
        plain = "Some guidance about outliers.\n\nA second paragraph."
        out = _strip_markdown_structure(plain)
        assert "Some guidance about outliers." in out
        assert "A second paragraph." in out


class TestMalformedTablesDegradeSafely:
    def test_a_row_with_fewer_cells_than_headers_does_not_crash(self) -> None:
        ragged = "| A | B | C |\n|---|---|---|\n| only one |\n"
        out = _strip_markdown_structure(ragged)
        assert "only one" in out

    def test_a_row_with_more_cells_than_headers_does_not_crash(self) -> None:
        ragged = "| A | B |\n|---|---|\n| x | y | z |\n"
        out = _strip_markdown_structure(ragged)
        assert "x" in out and "z" in out

    def test_a_table_with_no_separator_row_still_reads(self) -> None:
        no_sep = "| Method | Use when |\n| IQR | Skewed data |\n"
        out = _strip_markdown_structure(no_sep)
        assert "IQR" in out and "Skewed data" in out

    def test_empty_cells_are_dropped_rather_than_labelled_blank(self) -> None:
        table = "| A | B |\n|---|---|\n| value |  |\n"
        out = _strip_markdown_structure(table)
        assert "B:" not in out
        assert "value" in out

    def test_empty_input(self) -> None:
        assert _strip_markdown_structure("") == ""


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    """The real knowledge-base chunks, loaded once."""
    from rag.knowledge_loader import KnowledgeBaseLoader

    return KnowledgeBaseLoader().load_all()


class TestRealCorpusTables:
    """Against the actual knowledge base, not a hand-made fixture."""

    def test_the_corpus_contains_tables_to_begin_with(self, chunks: list[dict]) -> None:
        assert any("|" in c["text"] for c in chunks), "fixture assumption broken"

    def test_no_cleaned_chunk_leaks_pipes(self, chunks: list[dict]) -> None:
        for chunk in chunks:
            assert "|" not in _clean_technical(chunk["text"])

    def test_table_bearing_chunks_still_carry_content(self, chunks: list[dict]) -> None:
        """Previously these reduced to a bare heading."""
        for chunk in (c for c in chunks if c["text"].count("|") >= 4):
            cleaned = _clean_technical(chunk["text"])
            assert len(cleaned) > 60, f"chunk from {chunk['source']} collapsed to {cleaned!r}"
