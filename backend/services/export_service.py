"""
services/export_service.py
────────────────────────────
Generates downloadable artifacts (PDF report, JSON, citation bundle) from a
persisted AnalysisRun. Everything here reads from data already computed and
stored by the pipeline — no new analysis is run, no fabricated content.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.models.analysis_run import AnalysisRun
from rag.knowledge_loader import KnowledgeBaseLoader

# Categorical palette for cluster colouring / multi-series charts.
_CHART_PALETTE = [
    colors.HexColor("#22223b"), colors.HexColor("#9a8c98"), colors.HexColor("#4a4e69"),
    colors.HexColor("#c9ada7"), colors.HexColor("#6d6875"), colors.HexColor("#b5838d"),
]
# Cap charts per report so the PDF stays a sensible length.
_MAX_CHARTS = 6

# ── Shared: source path -> real document title ─────────────────────────────

_TITLE_CACHE: dict[str, str] | None = None


def _source_titles() -> dict[str, str]:
    """Map a knowledge-base source path to its real document title.

    Cached at module scope — the knowledge base is static seed content, not
    something that changes per-request, so re-parsing it on every export
    would be wasted work.
    """
    global _TITLE_CACHE
    if _TITLE_CACHE is None:
        _TITLE_CACHE = {
            chunk["source"]: chunk["metadata"].get("title", chunk["source"])
            for chunk in KnowledgeBaseLoader().load_all()
        }
    return _TITLE_CACHE


def _esc(value: Any) -> str:
    """
    Make arbitrary text safe to put inside a ReportLab ``Paragraph``.

    Paragraph parses its input as mini-XML, so unescaped user text is not
    merely a formatting nuisance — a goal containing ``<b `` raises
    ``paraparser: syntax error`` and fails the whole export with a 500, while
    something like ``<revenue>`` is silently swallowed as an unknown tag and
    the text disappears from the report. Both are reachable from anything a
    user can type into a goal, and from any column name in their data, since
    agent observations quote column names back.
    """
    return escape("" if value is None else str(value))


def _step_output(run: AnalysisRun, agent_name: str) -> dict[str, Any]:
    """Pull one agent's stored output dict out of a persisted run's step log."""
    for step in run.steps:
        if step.get("agent_name") == agent_name and step.get("status") == "success":
            return step.get("output") or {}
    return {}


# ── JSON export ──────────────────────────────────────────────────────────

def generate_json(run: AnalysisRun) -> bytes:
    """Full run payload as pretty-printed JSON, ready for download."""
    payload = {
        "run_id": run.id,
        "goal": run.goal,
        "expertise_level": run.expertise_level,
        "status": run.status,
        "summary": run.summary,
        "steps": run.steps,
        "recommendations": run.recommendations,
        "rag_sources": run.rag_sources,
        "created_at": run.created_at.isoformat(),
    }
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


# ── Citation bundle (BibTeX) ─────────────────────────────────────────────

def generate_citation_bundle(run: AnalysisRun) -> bytes:
    """BibTeX entries for every knowledge-base source this run's
    recommendations were grounded in."""
    titles = _source_titles()
    lines: list[str] = []

    for source in run.rag_sources:
        title = titles.get(source, source)
        # A short, stable citation key derived from the filename.
        key = source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        lines.append(
            "@misc{%s,\n"
            "  title        = {%s},\n"
            "  howpublished = {MAGE Knowledge Base, %s},\n"
            "  note         = {Grounding source for analysis run %s}\n"
            "}\n" % (key, title, source, run.id)
        )

    if not lines:
        lines.append("% No knowledge-base sources were cited in this run.\n")

    return "\n".join(lines).encode("utf-8")


# ── Chart rendering (from stored VisualizationAgent viz_specs) ─────────────

def _bar_drawing(labels: list[Any], values: list[float]) -> Drawing:
    """A simple vertical bar chart used for histograms, feature importance,
    category counts, and missingness."""
    d = Drawing(460, 175)
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 35, 40, 410, 120
    chart.data = [values]
    chart.categoryAxis.categoryNames = [str(v)[:12] for v in labels]
    chart.categoryAxis.labels.fontSize = 6
    chart.categoryAxis.labels.angle = 30
    chart.categoryAxis.labels.dy = -6
    chart.categoryAxis.labels.boxAnchor = "ne"
    chart.valueAxis.labels.fontSize = 6
    chart.valueAxis.valueMin = min(0.0, min(values, default=0.0))
    chart.bars[0].fillColor = _CHART_PALETTE[0]
    d.add(chart)
    return d


def _scatter_drawing(groups: list[list[tuple[float, float]]]) -> Drawing:
    """A scatter plot; each group is drawn as its own colour (used for
    cluster colouring). Connecting lines are suppressed — markers only."""
    d = Drawing(460, 200)
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 40, 30, 400, 150
    plot.data = groups
    for i in range(len(groups)):
        plot.lines[i].strokeColor = None  # no connecting line — scatter only
        plot.lines[i].symbol = makeMarker("FilledCircle")
        plot.lines[i].symbol.size = 3
        plot.lines[i].symbol.fillColor = _CHART_PALETTE[i % len(_CHART_PALETTE)]
    d.add(plot)
    return d


def _heatmap_table(columns: list[str], matrix: list[list[Any]]) -> Table:
    """Correlation matrix as a shaded table (blue = positive, rose = negative)."""
    header = [""] + [str(c)[:8] for c in columns]
    rows = [header]
    style: list[Any] = [
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#22223b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#22223b")),
        ("TEXTCOLOR", (0, 1), (0, -1), colors.white),
    ]
    for i, _ in enumerate(columns):
        row = [str(columns[i])[:8]]
        for j, _ in enumerate(columns):
            val = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else None
            row.append(f"{val:.2f}" if isinstance(val, (int, float)) else "")
            if isinstance(val, (int, float)):
                mag = min(abs(val), 1.0)
                # blue-ish for positive, rose for negative; intensity ∝ |r|.
                base = (0x4a, 0x4e, 0x69) if val >= 0 else (0xb5, 0x83, 0x8d)
                shade = colors.Color(
                    1 - (1 - base[0] / 255) * mag,
                    1 - (1 - base[1] / 255) * mag,
                    1 - (1 - base[2] / 255) * mag,
                )
                style.append(("BACKGROUND", (j + 1, i + 1), (j + 1, i + 1), shade))
        rows.append(row)
    table = Table(rows, hAlign="LEFT")
    table.setStyle(TableStyle(style))
    return table


def _chart_flowables(run: AnalysisRun, heading_style: ParagraphStyle, caption_style: ParagraphStyle) -> list[Any]:
    """Turn the stored VisualizationAgent chart specs into PDF flowables.

    Each chart is rendered defensively — a spec that can't be drawn is skipped
    rather than failing the whole export.
    """
    specs = _step_output(run, "VisualizationAgent").get("viz_specs") or []
    if not specs:
        return []

    flowables: list[Any] = [Paragraph("Visualizations", heading_style)]
    rendered = 0
    for spec in specs:
        if rendered >= _MAX_CHARTS:
            break
        try:
            drawing = _spec_to_flowable(spec)
        except Exception:  # noqa: BLE001 - a bad spec must not break the report
            drawing = None
        if drawing is None:
            continue
        flowables.append(Paragraph(str(spec.get("title", spec.get("type", "Chart"))), caption_style))
        flowables.append(drawing)
        flowables.append(Spacer(1, 10))
        rendered += 1

    # Only keep the heading if at least one chart rendered.
    return flowables if rendered else []


def _spec_to_flowable(spec: dict[str, Any]) -> Any | None:
    """Map one chart spec to a reportlab Drawing/Table, or None if unsupported."""
    ctype = spec.get("type")

    if ctype in ("feature_importance", "bar", "missingness_matrix"):
        items = spec.get("items") or []
        if not items:
            return None
        return _bar_drawing([it.get("label") for it in items], [float(it.get("value", 0)) for it in items])

    if ctype == "histogram":
        bins = spec.get("bins") or []
        if not bins:
            return None
        return _bar_drawing([b.get("label") for b in bins], [float(b.get("count", 0)) for b in bins])

    if ctype in ("scatter", "cluster_scatter"):
        points = spec.get("points") or []
        if not points:
            return None
        # Group by cluster when present so each cluster gets its own colour.
        groups: dict[Any, list[tuple[float, float]]] = {}
        for p in points:
            key = p.get("cluster", 0)
            groups.setdefault(key, []).append((float(p.get("x", 0)), float(p.get("y", 0))))
        return _scatter_drawing(list(groups.values()))

    if ctype == "correlation_heatmap":
        columns = spec.get("columns") or []
        matrix = spec.get("matrix") or []
        if not columns or not matrix:
            return None
        return _heatmap_table(columns, matrix)

    if ctype == "boxplot":
        # Render the five-number summary as a compact table.
        keys = ["min", "q1", "median", "q3", "max"]
        if not any(spec.get(k) is not None for k in keys):
            return None
        rows = [["min", "q1", "median", "q3", "max"],
                [f"{spec.get(k, '')}" for k in keys]]
        table = Table(rows, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9a8c98")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#22223b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]))
        return table

    return None


# ── PDF report ────────────────────────────────────────────────────────────

def _agent_trail_flowables(
    run: AnalysisRun, heading_style: ParagraphStyle, body_style: ParagraphStyle,
) -> list[Any]:
    """
    Render the Reason/Act/Observe log as a table (FR-06).

    The step log is the project's explainability trail, and until now it
    existed only in the JSON export — absent from the PDF, which is the
    artefact a reader actually opens. Including it here is what makes the
    claim "every agent step is inspectable" true of the deliverable and not
    just of the API.

    Reasoning and observation are wrapped in ``Paragraph`` rather than passed
    as bare strings so long text wraps inside its cell instead of overflowing
    the page width.
    """
    steps = run.steps or []
    if not steps:
        return []

    navy = colors.HexColor("#22223b")
    dusty_rose = colors.HexColor("#9a8c98")
    cell_style = ParagraphStyle(
        "MageTrailCell", parent=body_style, fontSize=8, leading=10.5,
    )
    header_style = ParagraphStyle(
        "MageTrailHeader", parent=cell_style, textColor=colors.white,
        fontName="Helvetica-Bold",
    )

    rows: list[list[Any]] = [[
        Paragraph(label, header_style)
        for label in ("#", "Agent", "Reasoning", "Observation", "Status", "Latency")
    ]]
    for index, step in enumerate(steps, start=1):
        # "RecommendationAgent" does not fit the column and wraps mid-word; the
        # column is already headed "Agent", so the suffix carries no meaning.
        agent = str(step.get("agent_name") or "—")
        agent = agent[: -len("Agent")] if agent.endswith("Agent") and agent != "Agent" else agent
        rows.append([
            Paragraph(str(index), cell_style),
            Paragraph(_esc(agent), cell_style),
            Paragraph(_esc(step.get("reasoning", "")), cell_style),
            Paragraph(_esc(step.get("observation", "")), cell_style),
            Paragraph(_esc(step.get("status", "")), cell_style),
            Paragraph(f"{step.get('latency_ms', 0)} ms", cell_style),
        ])

    table = Table(
        rows,
        hAlign="LEFT",
        repeatRows=1,  # re-print the header if the trail splits across pages
        colWidths=[0.3 * inch, 1.05 * inch, 2.1 * inch, 2.1 * inch, 0.6 * inch, 0.6 * inch],
    )
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("GRID", (0, 0), (-1, -1), 0.5, dusty_rose),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2e9e4")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )

    total_ms = sum(int(s.get("latency_ms") or 0) for s in steps)
    caption = ParagraphStyle(
        "MageTrailCaption", parent=body_style, fontSize=8.5,
        textColor=dusty_rose, spaceBefore=6,
    )
    return [
        Paragraph("Agent Execution Trail", heading_style),
        Paragraph(
            "Every step the pipeline executed, in order, with the reasoning that "
            "selected it and what it observed. This is the full inspectable trail — "
            "nothing below is reconstructed after the fact.",
            ParagraphStyle("MageTrailIntro", parent=body_style, spaceAfter=8),
        ),
        table,
        Paragraph(f"{len(steps)} step(s), {total_ms} ms total agent time.", caption),
    ]


def generate_pdf(run: AnalysisRun) -> bytes:
    """Render a PDF report: summary, data quality, recommendations, citations."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=f"MAGE Analysis Report — {run.id}",
    )

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#22223b")
    dusty_rose = colors.HexColor("#9a8c98")

    title_style = ParagraphStyle(
        "MageTitle", parent=styles["Title"], textColor=navy, spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "MageMeta", parent=styles["Normal"], textColor=dusty_rose, fontSize=9, spaceAfter=18,
    )
    heading_style = ParagraphStyle(
        "MageHeading", parent=styles["Heading2"], textColor=navy, spaceBefore=16, spaceAfter=8,
    )
    body_style = ParagraphStyle("MageBody", parent=styles["Normal"], leading=15)

    story: list[Any] = [
        Paragraph("MAGE — Analysis Report", title_style),
        Paragraph(
            f"Run ID: {run.id} &nbsp;&bull;&nbsp; Generated: "
            f"{datetime.now().strftime('%B %d, %Y')} &nbsp;&bull;&nbsp; "
            f"Expertise: {run.expertise_level.title()}",
            meta_style,
        ),
        Paragraph("Goal", heading_style),
        Paragraph(_esc(run.goal), body_style),
        Paragraph("Executive Summary", heading_style),
        Paragraph(_esc(run.summary) or "No summary available.", body_style),
    ]

    # ── Data Quality table (from MiningAgent's stored output, if present) ──
    data_quality = _step_output(run, "MiningAgent").get("data_quality") or {}
    if data_quality:
        story.append(Paragraph("Data Quality", heading_style))
        rows = [["Column", "Completeness", "Uniqueness", "Missing"]]
        for col, q in data_quality.items():
            rows.append([
                col,
                f"{q.get('completeness_pct', 0)}%",
                f"{q.get('uniqueness_pct', 0)}%",
                str(q.get("missing_count", 0)),
            ])
        table = Table(rows, hAlign="LEFT", colWidths=[2.2 * inch, 1.3 * inch, 1.3 * inch, 1 * inch])
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), navy),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, dusty_rose),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2e9e4")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(table)

    # ── Visualizations (from stored VisualizationAgent viz_specs) ─────────
    caption_style = ParagraphStyle(
        "MageCaption", parent=body_style, fontSize=9, textColor=dusty_rose, spaceBefore=6, spaceAfter=2,
    )
    story.extend(_chart_flowables(run, heading_style, caption_style))

    # ── Recommendations ──────────────────────────────────────────────────
    story.append(Paragraph("Recommendations", heading_style))
    if run.recommendations:
        for i, rec in enumerate(run.recommendations, start=1):
            story.append(Paragraph(f"{i}. {_esc(rec)}", body_style))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No recommendations were grounded for this goal.", body_style))

    # ── Citations ─────────────────────────────────────────────────────────
    story.append(Paragraph("Grounded In", heading_style))
    titles = _source_titles()
    if run.rag_sources:
        for source in run.rag_sources:
            story.append(
                Paragraph(
                    f"&bull; {_esc(titles.get(source, source))} ({_esc(source)})", body_style
                )
            )
    else:
        story.append(Paragraph("No knowledge-base sources were cited.", body_style))

    # ── Agent Execution Trail (FR-06) ─────────────────────────────────────
    story.extend(_agent_trail_flowables(run, heading_style, body_style))

    doc.build(story)
    return buffer.getvalue()
