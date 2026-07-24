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


# ── PDF report ────────────────────────────────────────────────────────────

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
        Paragraph(run.goal, body_style),
        Paragraph("Executive Summary", heading_style),
        Paragraph(run.summary or "No summary available.", body_style),
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

    # ── Recommendations ──────────────────────────────────────────────────
    story.append(Paragraph("Recommendations", heading_style))
    if run.recommendations:
        for i, rec in enumerate(run.recommendations, start=1):
            story.append(Paragraph(f"{i}. {rec}", body_style))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No recommendations were grounded for this goal.", body_style))

    # ── Citations ─────────────────────────────────────────────────────────
    story.append(Paragraph("Grounded In", heading_style))
    titles = _source_titles()
    if run.rag_sources:
        for source in run.rag_sources:
            story.append(Paragraph(f"&bull; {titles.get(source, source)} ({source})", body_style))
    else:
        story.append(Paragraph("No knowledge-base sources were cited.", body_style))

    doc.build(story)
    return buffer.getvalue()
