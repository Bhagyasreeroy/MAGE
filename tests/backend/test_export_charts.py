"""
tests/backend/test_export_charts.py
────────────────────────────────────
Tests that the M4 chart types reach the PDF.

The project keeps **one source of truth and two renderers**: the
VisualizationAgent emits plain-dict specs, the frontend draws them in React and
``export_service`` draws the same specs in ReportLab. A chart type added to the
agent but not to the exporter is therefore a silent regression — the run shows
it on screen and the PDF, the artefact an examiner actually opens, quietly drops
it.

These tests pin the six types added with the dedicated builders
(``grouped_bar``, ``box_by_class``, ``pairplot``, ``highlighted_scatter``,
``violin``, ``line``) as renderable, and pin the defensive behaviour around them:
a malformed spec must be skipped, never fatal.

Run against the pure ``export_service``, so no database is needed.
"""

from __future__ import annotations

import io
import os
import sys

import pdfplumber
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.services import export_service

# ── Representative specs, shaped exactly as VisualizationAgent emits them ────

GROUPED_BAR = {
    "type": "grouped_bar",
    "title": "'region' by 'churned'",
    "x_label": "region",
    "group_label": "churned",
    "categories": ["East", "North", "West"],
    "series": [
        {"name": "no", "values": [12, 9, 14]},
        {"name": "yes", "values": [8, 11, 6]},
    ],
}

BOX_BY_CLASS = {
    "type": "box_by_class",
    "title": "'tenure' by 'churned'",
    "column": "tenure",
    "target": "churned",
    "groups": [
        {"label": "no", "count": 35, "min": 10.0, "q1": 24.0, "median": 30.0, "q3": 36.0, "max": 52.0},
        {"label": "yes", "count": 25, "min": 8.0, "q1": 19.0, "median": 26.0, "q3": 33.0, "max": 47.0},
    ],
}

PAIRPLOT = {
    "type": "pairplot",
    "title": "Pairwise Relationships",
    "columns": ["tenure", "spend", "visits"],
    "pairs": [
        {
            "x_label": "tenure",
            "y_label": "spend",
            "r": 0.87,
            "points": [{"x": 1.0, "y": 3.0}, {"x": 2.0, "y": 6.5}, {"x": 3.0, "y": 8.9}],
        },
        {
            "x_label": "tenure",
            "y_label": "visits",
            "r": -0.12,
            "points": [{"x": 1.0, "y": 9.0}, {"x": 2.0, "y": 4.0}, {"x": 3.0, "y": 7.0}],
        },
    ],
}

HIGHLIGHTED_SCATTER = {
    "type": "highlighted_scatter",
    "title": "'amount' vs 'score' — 3 outliers highlighted",
    "x_label": "amount",
    "y_label": "score",
    "highlighted_count": 3,
    "points": [
        {"x": 50.0, "y": 70.0, "outlier": False},
        {"x": 52.0, "y": 73.0, "outlier": False},
        {"x": 900.0, "y": 71.0, "outlier": True},
        {"x": 950.0, "y": 69.0, "outlier": True},
        {"x": -400.0, "y": 72.0, "outlier": True},
    ],
}

VIOLIN = {
    "type": "violin",
    "title": "Distribution shape of 'tenure'",
    "column": "tenure",
    "min": 8.0,
    "q1": 22.0,
    "median": 29.0,
    "q3": 35.0,
    "max": 52.0,
    "bands": [
        {"center": 10.0, "count": 3, "width": 0.25},
        {"center": 20.0, "count": 8, "width": 0.67},
        {"center": 30.0, "count": 12, "width": 1.0},
        {"center": 40.0, "count": 5, "width": 0.42},
        {"center": 50.0, "count": 2, "width": 0.17},
    ],
}

LINE = {
    "type": "line",
    "title": "'revenue' over 'order_date'",
    "x_label": "order_date",
    "y_label": "revenue",
    "points": [
        {"x": "2026-01-01", "y": 4.2},
        {"x": "2026-01-02", "y": 7.1},
        {"x": "2026-01-03", "y": 9.8},
        {"x": "2026-01-04", "y": 12.0},
    ],
}

NEW_SPECS = {
    "grouped_bar": GROUPED_BAR,
    "box_by_class": BOX_BY_CLASS,
    "pairplot": PAIRPLOT,
    "highlighted_scatter": HIGHLIGHTED_SCATTER,
    "violin": VIOLIN,
    "line": LINE,
}


class FakeRun:
    """Minimal stand-in for a persisted AnalysisRun — export never re-computes."""

    def __init__(self, specs: list[dict]) -> None:
        self.id = "run-charts"
        self.goal = "Profile this dataset"
        self.summary = "Goal classified as 'reporting'."
        self.expertise_level = "expert"
        self.recommendations = ["Check the class balance before modelling."]
        self.rag_sources = ["class_imbalance.md"]
        self.steps = [
            {
                "agent_name": "VisualizationAgent",
                "action": "run VisualizationAgent",
                "reasoning": "Generate reporting-appropriate charts.",
                "observation": f"Selected {len(specs)} chart(s).",
                "status": "success",
                "latency_ms": 25,
                "output": {"viz_specs": specs},
            }
        ]


def _pdf_text(run: FakeRun) -> str:
    with pdfplumber.open(io.BytesIO(export_service.generate_pdf(run))) as pdf:
        raw = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return " ".join(raw.split())


# ── Each new type is renderable ──────────────────────────────────────────────


class TestNewChartTypesRender:
    @pytest.mark.parametrize("name", sorted(NEW_SPECS))
    def test_spec_maps_to_a_flowable(self, name: str) -> None:
        assert export_service._spec_to_flowable(NEW_SPECS[name]) is not None, (
            f"{name} is emitted by the agent but the PDF exporter drops it"
        )

    @pytest.mark.parametrize("name", sorted(NEW_SPECS))
    def test_title_appears_in_the_pdf(self, name: str) -> None:
        spec = NEW_SPECS[name]
        text = _pdf_text(FakeRun([spec]))
        assert "Visualizations" in text
        # The title is the caption; check a distinctive fragment of it, since
        # long captions wrap and quoting is normalised by the extractor.
        assert spec["title"].split("—")[0].strip()[:18] in text

    def test_all_six_render_together(self) -> None:
        """The exporter caps charts, but every one it does draw must be a real
        chart rather than a skipped spec leaving a bare caption."""
        flowables = [export_service._spec_to_flowable(s) for s in NEW_SPECS.values()]
        assert all(f is not None for f in flowables)


# ── Content assertions ───────────────────────────────────────────────────────


class TestRenderedContent:
    def test_grouped_bar_keeps_every_series(self) -> None:
        """A grouped bar drawn with one series has lost the comparison that
        makes it a grouped bar."""
        drawing = export_service._spec_to_flowable(GROUPED_BAR)
        chart = drawing.contents[0]
        assert len(chart.data) == len(GROUPED_BAR["series"])
        assert list(chart.data[0]) == GROUPED_BAR["series"][0]["values"]

    def test_box_by_class_names_each_class(self) -> None:
        text = _pdf_text(FakeRun([BOX_BY_CLASS]))
        assert "no" in text and "yes" in text
        assert "median" in text.lower()

    def test_highlighted_scatter_separates_flagged_points(self) -> None:
        """Outliers must be their own series, otherwise they cannot be coloured
        differently and the chart is just a scatter."""
        drawing = export_service._spec_to_flowable(HIGHLIGHTED_SCATTER)
        plot = drawing.contents[0]
        assert len(plot.data) == 2, "expected a normal series and a highlighted series"
        assert len(plot.data[1]) == HIGHLIGHTED_SCATTER["highlighted_count"]

    def test_pairplot_renders_every_panel(self) -> None:
        drawing = export_service._spec_to_flowable(PAIRPLOT)
        plot = drawing.contents[0]
        assert len(plot.data) == len(PAIRPLOT["pairs"])

    def test_line_preserves_point_order(self) -> None:
        drawing = export_service._spec_to_flowable(LINE)
        plot = drawing.contents[0]
        ys = [y for _, y in plot.data[0]]
        assert ys == [p["y"] for p in LINE["points"]]

    def test_violin_reports_the_band_profile(self) -> None:
        drawing = export_service._spec_to_flowable(VIOLIN)
        chart = drawing.contents[0]
        assert list(chart.data[0]) == [b["count"] for b in VIOLIN["bands"]]


# ── Defensive behaviour ──────────────────────────────────────────────────────


class TestMalformedSpecsAreSkippedNotFatal:
    @pytest.mark.parametrize("name", sorted(NEW_SPECS))
    def test_empty_payload_returns_none(self, name: str) -> None:
        """An empty chart is skipped so the report has no bare caption."""
        spec = dict(NEW_SPECS[name])
        for key in ("series", "groups", "pairs", "points", "bands"):
            if key in spec:
                spec[key] = []
        assert export_service._spec_to_flowable(spec) is None

    def test_a_broken_spec_does_not_fail_the_export(self) -> None:
        broken = {"type": "grouped_bar", "title": "Broken", "categories": None, "series": "nonsense"}
        text = _pdf_text(FakeRun([broken, LINE]))
        # The good chart still renders; the broken one is simply absent.
        assert "revenue" in text

    def test_unsafe_characters_in_a_chart_title_do_not_crash(self) -> None:
        """ReportLab parses Paragraph input as mini-XML; column names reach
        captions, so a column called '<revenue>' must not break the export."""
        spec = dict(LINE, title="'<revenue>' over 'order_date' & more")
        text = _pdf_text(FakeRun([spec]))
        assert "revenue" in text
