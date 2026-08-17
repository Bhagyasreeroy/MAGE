"""
scripts/generate_project_report.py
───────────────────────────────────
Generates a comprehensive MAGE project report as a PDF, covering the full
system design and everything implemented to date. Run:

    python scripts/generate_project_report.py

Writes: MAGE_Project_Report.pdf (project root)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parents[1] / "MAGE_Project_Report.pdf"

NAVY = colors.HexColor("#22223b")
SLATE = colors.HexColor("#4a4e69")
ROSE = colors.HexColor("#9a8c98")
CREAM = colors.HexColor("#f2e9e4")
GREEN = colors.HexColor("#2d6a4f")
AMBER = colors.HexColor("#9c6644")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=NAVY, fontSize=16,
                    spaceBefore=18, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=SLATE, fontSize=12.5,
                    spaceBefore=12, spaceAfter=5)
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=15,
                      alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = ParagraphStyle("Bullet", parent=BODY, alignment=TA_LEFT, spaceAfter=2)
META = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9, textColor=ROSE)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontSize=8.5, textColor=SLATE,
                      backColor=CREAM, leading=12)


def h1(t): return Paragraph(t, H1)
def h2(t): return Paragraph(t, H2)
def p(t): return Paragraph(t, BODY)
def code(t): return Paragraph(t.replace(" ", "&nbsp;"), CODE)
def sp(h=8): return Spacer(1, h)


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(i, BULLET), leftIndent=6) for i in items],
        bulletType="bullet", bulletColor=ROSE, leftIndent=14, bulletFontSize=7,
    )


def table(rows, widths, header=True, status_col=None):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, ROSE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CREAM]),
        ]
    t.setStyle(TableStyle(style))
    return t


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(ROSE)
    canvas.drawString(0.75 * inch, 0.5 * inch,
                      "MAGE — Multimodal Agentic Goal-conditioned EDA with RAG-grounded Recommendations")
    canvas.drawRightString(A4[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.setStrokeColor(CREAM)
    canvas.line(0.75 * inch, 0.62 * inch, A4[0] - 0.75 * inch, 0.62 * inch)
    canvas.restoreState()


def build():
    story = []

    # ── Title page ──────────────────────────────────────────────────────────
    story += [
        Spacer(1, 1.4 * inch),
        Paragraph("MAGE", ParagraphStyle("T", parent=styles["Title"], fontSize=42, textColor=NAVY, alignment=TA_CENTER, leading=48)),
        sp(18),
        Paragraph("Multimodal Agentic Goal-conditioned EDA<br/>with RAG-grounded Recommendations",
                  ParagraphStyle("ST", parent=styles["Title"], fontSize=15, textColor=SLATE, alignment=TA_CENTER, leading=20)),
        sp(40),
        Paragraph("Project Report", ParagraphStyle("PR", parent=styles["Normal"], fontSize=13, textColor=ROSE, alignment=TA_CENTER)),
        sp(30),
        Paragraph("Bhagyasree Roy (2547118)<br/>Nandini Singh (2547135)<br/>Neha N (2547160)",
                  ParagraphStyle("AU", parent=styles["Normal"], fontSize=11, alignment=TA_CENTER, leading=17)),
        sp(24),
        Paragraph("Under the guidance of<br/>Dr. Tegil J John",
                  ParagraphStyle("GU", parent=styles["Normal"], fontSize=10.5, textColor=SLATE, alignment=TA_CENTER, leading=15)),
        sp(30),
        Paragraph("Department of Computer Science<br/>CHRIST (Deemed to be University), Bengaluru",
                  ParagraphStyle("UN", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER, leading=15)),
        sp(20),
        Paragraph(datetime.now().strftime("%B %Y"),
                  ParagraphStyle("DT", parent=styles["Normal"], fontSize=10, textColor=ROSE, alignment=TA_CENTER)),
        PageBreak(),
    ]

    # ── Abstract ────────────────────────────────────────────────────────────
    story += [
        h1("Abstract"),
        p("Exploratory Data Analysis (EDA) is typically manual and expertise-dependent, and existing automated "
          "EDA tools compound this by producing the same generic statistical report regardless of <i>why</i> the "
          "data is being examined. Large language model (LLM) agents offer a path toward goal-aware analysis, but "
          "agents that improvise statistical interpretation without grounding risk producing confident, plausible, "
          "and unverifiable recommendations."),
        p("This project presents <b>MAGE (Multimodal Agentic Goal-conditioned EDA with RAG-grounded "
          "Recommendations)</b>, a multi-agent system that accepts multimodal data inputs — CSV, Excel, JSON, "
          "Parquet, PDF tables, SQL databases, and REST endpoints — alongside a natural-language goal statement, "
          "and conducts EDA conditioned on that goal. A planning agent classifies the underlying task type and "
          "orchestrates specialist sub-agents through a reason–act–observe loop. Critically, the goal conditions "
          "<i>which computations are prioritised and executed</i>, not merely which results are surfaced — the "
          "distinction that separates MAGE from prior goal-aware systems. A retrieval-augmented generation (RAG) "
          "layer grounds every recommendation in a curated EDA methodology knowledge base, attaching a citation "
          "chain to each output rather than relying on unverified model judgment. Recommendations are further "
          "adapted to the reader across three registers, from plain language to full technical framing."),
        p("This report documents the system design, the implementation, and an empirical evaluation of the central "
          "claim. Under a controlled same-dataset / different-goal design, varying only the stated goal changes "
          "the set of statistical computations MAGE executes at a mean pairwise Jaccard distance of 0.836, against "
          "0.000 for a generic AutoEDA baseline which by construction cannot vary its output by goal."),
    ]

    # ── 1. Introduction ─────────────────────────────────────────────────────
    story += [
        h1("1. Introduction"),
        h2("1.1 Overview"),
        p("Exploratory Data Analysis is the stage where an analyst decides what a dataset is actually telling them "
          "— which variables matter, what is broken, what is worth modelling further. In practice this stage is "
          "almost entirely manual and expertise-dependent, which makes the quality of EDA highly inconsistent "
          "across practitioners and projects."),
        p("Automated EDA tools were built to close this gap, but they solve a different problem: they generate the "
          "same comprehensive report regardless of intent — a churn investigation and a pricing analysis on the "
          "identical dataset produce identical output. LLM-driven agents can reason about a stated goal, but "
          "without grounding they produce confident, plausible-sounding, and sometimes wrong recommendations. "
          "MAGE is motivated by closing both gaps at once: making EDA goal-aware <i>and</i> keeping every "
          "recommendation traceable to real methodology."),
        h2("1.2 Problem Statement"),
        p("There is no EDA system in which a natural-language goal statement conditions which analytical "
          "computations are performed and prioritised, where those computations' resulting recommendations are "
          "grounded in retrievable methodology with citations (reducing hallucination), and where the same "
          "analysis is communicated differently depending on the reader's expertise."),
        h2("1.3 Objectives"),
        bullets([
            "<b>Multimodal ingestion</b> — accept heterogeneous input formats and normalise them into a single unified internal schema.",
            "<b>Goal-conditioned planning</b> — parse a natural-language goal, classify the task type (classification, regression, clustering, anomaly detection, reporting), and build a conditional analysis pipeline.",
            "<b>Multi-agent orchestration</b> — a planning agent that reasons about the goal, routes work to specialist sub-agents, and aggregates their outputs.",
            "<b>RAG-grounded recommendations</b> — ground every recommendation in a retrieval-augmented methodology knowledge base so recommendations are citable rather than improvised.",
            "<b>Explainability</b> — attach confidence scores and an inspectable reason–act–observe trail to every run.",
            "<b>Role-adaptive output</b> — generate the same analysis in a technical register and a plain-language register.",
            "<b>Empirical validation</b> — a same-dataset / different-goal evaluation that isolates goal-conditioning as the tested variable.",
        ]),
    ]

    # ── 2. Existing systems ─────────────────────────────────────────────────
    story += [
        h1("2. Existing Systems & Limitations"),
        bullets([
            "<b>Rule-based profiling tools</b> (ydata-profiling, Sweetviz, AutoViz, Lux) — fast and mature for statistical summaries, but run a fixed set of computations identical regardless of user intent. They neither accept nor reason about a goal.",
            "<b>LLM-driven data-science / visualization agents</b> (LIDA, DataInterpreter, DS-Agent) — shape questions or charts from a prompt, but lack a retrieval-grounded verification step; recommendations are unverified model outputs.",
            "<b>Goal-oriented EDA research</b> — conditions <i>what gets visualised</i>, not <i>which statistical computations run</i>; none combine goal-conditioned computation with RAG-grounded citation and dual-audience output.",
        ]),
        p("<b>Positioning.</b> MAGE's contribution is a specific, testable combination: goal-conditioning that "
          "changes which computations run (not just which are surfaced), grounded in retrievable methodology rather "
          "than LLM improvisation, with output that adapts to the reader's expertise, evaluated empirically against "
          "a generic baseline using a same-dataset / different-goal design."),
    ]

    # ── 3. System Architecture ──────────────────────────────────────────────
    story += [
        PageBreak(),
        h1("3. System Architecture"),
        p("MAGE is organised into five architectural layers:"),
        bullets([
            "<b>Layer 1 — User Interface:</b> natural-language goal input, expertise-level selector, multimodal data upload, and an explainability dashboard (React / Next.js).",
            "<b>Layer 2 — Orchestrator Agent:</b> goal conditioning, task classification, task decomposition, agent routing, and session state, driven by a reason–act–observe loop with a hard step cap.",
            "<b>Layer 3 — Specialist Agents:</b> ingestion, mining, visualization, and recommendation agents, each operating within its defined scope under orchestrator direction.",
            "<b>Layer 4 — Data & RAG Pipeline:</b> the vector store, curated methodology knowledge base, and the statistical processing engine.",
            "<b>Layer 5 — Deployment & Infrastructure:</b> FastAPI backend, Next.js SPA frontend, PostgreSQL, and containerised deployment via Docker Compose.",
        ]),
        h2("3.1 Pipeline Flow"),
        p("A single analysis flows: <b>Goal + Data</b> → Orchestrator classifies the goal → Planner builds a "
          "conditional pipeline → IngestionAgent normalises the data → MiningAgent runs the goal-selected "
          "computations → VisualizationAgent draws the goal-selected charts → RecommendationAgent grounds findings "
          "in the RAG knowledge base and produces cited, expertise-adapted recommendations → results are persisted "
          "and exportable as PDF / JSON / citation bundle."),
    ]

    # ── 4. Technology Stack ─────────────────────────────────────────────────
    story += [
        h1("4. Technology Stack"),
        table([
            ["Layer", "Technologies"],
            ["Backend / API", "FastAPI (Python 3.11+), Pydantic, Uvicorn"],
            ["Agents / Orchestration", "Custom multi-agent orchestrator (ReAct-style loop), swappable LLM-provider interface"],
            ["RAG / Embeddings", "sentence-transformers (all-MiniLM-L6-v2, local, 384-dim), FAISS / ChromaDB vector store"],
            ["Data / Mining", "pandas, NumPy, scikit-learn, SciPy, PyArrow, openpyxl"],
            ["Ingestion", "pdfplumber (PDF tables), SQLAlchemy (databases), httpx (REST)"],
            ["Auth", "JWT (python-jose), Google OAuth2 (Authlib), passlib/bcrypt"],
            ["Persistence", "PostgreSQL (async SQLAlchemy + asyncpg)"],
            ["Export", "reportlab (PDF report + charts)"],
            ["Frontend", "React, Next.js (App Router), TypeScript"],
            ["Infrastructure", "Docker, Docker Compose"],
        ], widths=[1.6 * inch, 4.4 * inch]),
    ]

    # ── 5. Databases ────────────────────────────────────────────────────────
    story += [
        h1("5. Databases & Data Stores"),
        table([
            ["Store", "Type", "Purpose"],
            ["PostgreSQL", "Relational (async)", "Users, datasets, and analysis runs — persisted per account."],
            ["ChromaDB / FAISS", "Vector", "Embeddings of the EDA methodology knowledge base for semantic retrieval (RAG). Runs embedded, no separate server."],
            ["SQLite", "Relational", "Dependency-free fallback for tests only."],
        ], widths=[1.3 * inch, 1.4 * inch, 3.3 * inch]),
        sp(6),
        p("The relational schema comprises <b>users</b> (accounts, hashed passwords, role, default expertise level), "
          "<b>datasets</b> (uploaded-dataset metadata), and <b>analysis_runs</b> (goal, task type, status, and the "
          "findings / recommendations / citations for each run). Embeddings are computed locally with "
          "sentence-transformers, so the RAG layer requires no external API key."),
    ]

    # ── 6. Modules ──────────────────────────────────────────────────────────
    story += [
        PageBreak(),
        h1("6. Module Descriptions & Implementation Status"),
        p("Legend: <font color='#2d6a4f'><b>Done</b></font> · "
          "<font color='#9c6644'><b>Partial</b></font> · Pending."),
        table([
            ["Module", "Responsibility", "Status"],
            ["M1 — Ingestion & Normalization",
             "Normalise heterogeneous inputs into a canonical DataFrame.",
             "Done (CSV/TSV/JSON/Parquet/Excel/PDF + DB URI + REST; OCR deferred)"],
            ["M2 — Goal Conditioning & Orchestrator",
             "Classify the goal, build a conditional pipeline, drive the ReAct loop.",
             "Done — directives executed by mining & viz"],
            ["M3 — Data Mining Agent",
             "Goal-conditioned statistical profiling & pattern discovery.",
             "Done (corr, IQR, PCA, KMeans, DBSCAN, Isolation Forest, class balance, linearity); time-series pending"],
            ["M4 — Visualization Agent",
             "Goal-conditioned chart selection.",
             "Done (14 chart types incl. grouped bar, box-by-class, pairplot, highlighted scatter, violin, line); choropleth pending"],
            ["M5 — RAG Pipeline",
             "Retrieval-grounded methodology with citations.",
             "Done (local embeddings, FAISS/Chroma, top-k retrieval, citations)"],
            ["M6 — Recommendation & Explainability",
             "RAG-grounded, expertise-adapted, cited recommendations + Q&A.",
             "Done (3 registers, SHAP attribution, confidence, citations); LLM synthesis pending"],
            ["M7 — Frontend & UX",
             "Onboarding, live dashboard, report view, PDF export.",
             "Done, incl. live WebSocket agent-step streaming"],
            ["M8 — API Layer & Authentication",
             "REST + WebSocket endpoints, JWT + Google OAuth, persistence.",
             "Done, incl. WebSocket streaming; Celery/Redis async queue pending"],
            ["Evaluation — Empirical Validation",
             "Same-dataset / different-goal experiment vs. an AutoEDA baseline.",
             "Done (evaluation/ harness; 25 runs, results reproducible)"],
        ], widths=[1.5 * inch, 2.6 * inch, 1.9 * inch]),
    ]

    # ── 7. Core contribution: goal-conditioning ─────────────────────────────
    story += [
        PageBreak(),
        h1("7. Core Contribution — Goal-Conditioned Execution"),
        p("The project's central claim is that the goal conditions <b>which computations run</b>, not just which "
          "results are shown. The planner emits per-task directives, and — critically — the specialist agents now "
          "<b>execute</b> them, so two different goals on the same dataset produce genuinely different analysis:"),
        table([
            ["Goal / Task type", "Mining computations run", "Charts drawn"],
            ["Anomaly detection", "IQR outliers + Isolation Forest", "Box / highlighted scatter"],
            ["Clustering", "KMeans (silhouette-selected k) + DBSCAN", "Cluster scatter"],
            ["Regression", "Correlation + feature↔target linearity + SHAP attribution", "Scatter + correlation heatmap + feature attribution"],
            ["Classification", "Class balance / imbalance ratio + feature importance + SHAP attribution", "Grouped bar / feature attribution"],
            ["Reporting", "Descriptive profile + missingness", "Histograms + missingness matrix"],
        ], widths=[1.5 * inch, 2.6 * inch, 1.9 * inch]),
        sp(6),
        p("Identifier columns (e.g. <font face='Courier'>order_id</font>) are excluded from distance- and "
          "variance-based models (PCA, KMeans, DBSCAN, Isolation Forest) so results are not skewed by row IDs, "
          "while continuous measurements are retained. When no directives are supplied the agents fall back to a "
          "full default profile, so they remain useful standalone. This behaviour is locked in by a dedicated test "
          "suite that asserts different task types produce different computations and chart sets."),
        h2("7.1 RAG Grounding & Role-Adaptive Output"),
        p("Every recommendation is grounded in a curated knowledge base of five EDA methodology documents "
          "(clustering, correlation, distribution profiling, missing values, outlier detection), retrieved by "
          "semantic similarity and cited by source. Each recommendation is produced in <b>three registers</b> — "
          "plain language for a beginner, the finding plus its supporting statistic and an inline methodology "
          "attribution for an analyst, and the full cited excerpt for a data scientist — selected by the user's "
          "declared expertise level."),
        h2("7.2 Feature Attribution"),
        p("For classification and regression goals the pipeline additionally fits a small gradient-boosted tree to "
          "the detected target and explains it with SHAP, reporting each feature's mean absolute SHAP value as a "
          "share of the total explanation. This is distinct from the unsupervised PCA ranking, which describes "
          "variance rather than the target. Two fields accompany every attribution so it cannot be read "
          "uncritically: the explanation method actually used (SHAP, or permutation importance if SHAP is "
          "unavailable), and the fitted model's score — attributions from a model that cannot predict its target "
          "are not meaningful, and the reader is given what they need to judge that."),
    ]

    # ── 7.5 Empirical evaluation ────────────────────────────────────────────
    story += [
        PageBreak(),
        h1("8. Empirical Evaluation"),
        p("The goal-conditioning claim is tested rather than asserted, using the proposal's "
          "<b>same-dataset / different-goal</b> design: each dataset is held constant and pushed through the "
          "pipeline once per goal, one goal per task type, so that any difference between runs is attributable to "
          "the goal and nothing else. Five datasets (four scikit-learn built-ins plus one seeded synthetic set "
          "carrying planted clusters, outliers, class imbalance and missingness) across five goals gives 25 runs."),
        sp(4),
        p("The comparison baseline is <b>ydata-profiling</b>, a representative rule-based AutoEDA tool. Its "
          "defining property is structural: <font face='Courier'>ProfileReport</font> exposes no parameter through "
          "which an analytical goal could be expressed, so its computation set cannot vary by goal. This was "
          "verified empirically — a live ydata-profiling run produced an identical computation set on all five "
          "datasets."),
        sp(6),
        table([
            ["Metric", "MAGE", "AutoEDA baseline"],
            ["Cross-goal computation divergence (Jaccard)", "0.836", "0.000"],
            ["Cross-goal chart divergence (Jaccard)", "0.920", "0.000"],
            ["Task-relevant computation precision", "1.000", "0.400"],
            ["Task-relevant computation recall", "0.601", "0.539"],
            ["Task-relevant F1", "0.749", "0.458"],
            ["Citation coverage (FR-03)", "1.000", "—"],
            ["Goal classification accuracy", "1.000", "—"],
            ["Maximum end-to-end runtime (FR-05)", "2.33 s", "—"],
        ], widths=[3.0 * inch, 1.5 * inch, 1.5 * inch]),
        sp(6),
        p("The headline figure is the first row. A divergence of 0.836 means that changing only the stated goal "
          "changes almost the entire set of statistical computations that execute; the baseline's 0.000 means it "
          "runs exactly the same work every time. The runtime figure also discharges FR-05, with a worst case of "
          "roughly 2 seconds against a 60-second budget."),
        h2("8.1 Interpreting these numbers honestly"),
        bullets([
            "The precision of 1.000 is <b>ceilinged</b>: it shows the planner emits nothing outside the "
            "task-relevant set, but cannot distinguish a good conditional pipeline from an excellent one. Recall "
            "is the more informative measure for MAGE.",
            "The baseline reaches comparable recall by <b>brute force</b> — it runs its entire fixed set every "
            "time, so it incidentally covers the relevant computations while also running many irrelevant ones. "
            "Its low precision is what that costs, and F1 is the fair single comparison.",
            "The relevance sets used to score precision and recall are <b>hand-labelled</b>, and deliberately not "
            "derived from the planner; deriving them from the system under test would make the metric circular "
            "and guarantee a perfect score.",
            "Reported runtimes exclude a discarded warm-up run. The embedding model loads lazily on first use, "
            "and left in place that one-off cost lands entirely on whichever run happens to execute first.",
        ]),
        sp(4),
        p("The experiment is reproducible from a single command "
          "(<font face='Courier'>python -m evaluation.harness</font>), which regenerates both the machine-readable "
          "results and the tables above."),
    ]

    # ── 8. Requirements ─────────────────────────────────────────────────────
    story += [
        h1("9. Functional & Non-Functional Requirements"),
        table([
            ["Req", "Description", "Status"],
            ["FR-01", "Accept ≥5 input formats (CSV, Excel, JSON, PDF, DB URI)", "Met (+ REST; OCR deferred)"],
            ["FR-02", "Goal conditions all computations, not post-hoc", "Met (plan + execution)"],
            ["FR-03", "Every recommendation carries a RAG citation", "Met"],
            ["FR-04", "Expertise level visibly alters output language", "Met (3 registers)"],
            ["FR-05", "End-to-end analysis < 60s for <100k rows", "Met (benchmarked; max 2.33 s)"],
            ["FR-06", "Every agent step logged / inspectable", "Met"],
            ["NFR-01", "Horizontal scaling via containerization", "Docker; K8s stretch"],
            ["NFR-02", "Incremental vector indexing", "Pending"],
            ["NFR-03", "LLM API retry with backoff", "N/A until LLM wired"],
            ["NFR-04", "Files sandboxed / deleted post-session", "Pending verification"],
        ], widths=[0.7 * inch, 3.6 * inch, 1.7 * inch]),
    ]

    # ── 9. Work done this cycle ─────────────────────────────────────────────
    story += [
        PageBreak(),
        h1("10. Work Completed in the Current Development Cycle"),
        h2("10.1 Ingestion (M1)"),
        bullets([
            "PDF table extraction via pdfplumber, with clear handling of scanned (image) PDFs.",
            "Database ingestion via SQLAlchemy (any dialect) — table / query / URI-fragment selection.",
            "REST/HTTP ingestion via httpx — JSON / CSV / TSV / Parquet / Excel by content-type or hint.",
            "Automatic source-type routing (file / URL / database) — one source string, no type flag needed.",
        ]),
        h2("10.2 Goal-Conditioned Execution (M2 / M3 / M4)"),
        bullets([
            "MiningAgent and VisualizationAgent now honour the planner's per-task directives (previously ignored, so every goal produced identical output).",
            "New computations: Isolation Forest (anomaly), DBSCAN (clustering), class balance (classification), feature↔target linearity (regression).",
            "New charts: scatter and missingness matrix; the chart set now varies by goal.",
            "Identifier columns excluded from distance/variance-based models to prevent skew.",
            "Backward compatible — no directives yields the full default profile.",
        ]),
        h2("10.3 Reporting, Export & Demo"),
        bullets([
            "PDF export now renders visualizations (bar charts, scatter / cluster scatter, correlation heatmap, box summaries) in addition to data-quality and recommendation sections.",
            "Recommendation text cleaned of flattened markdown tables and headings for readable prose.",
            "A realistic 60-row demo dataset (with outliers, clusters, correlation, a label, and missing values) plus a reproducible generator, so all goal-conditioned computations are exercised.",
        ]),
        h2("10.4 Authentication & Infrastructure"),
        bullets([
            "Google OAuth2 sign-in fixed end-to-end: environment forwarding into the backend container and cookie-based session recognition for the route guard.",
            "Full stack runs under Docker Compose (backend, frontend, PostgreSQL).",
        ]),
        h2("10.5 Explainability, Streaming & Evaluation"),
        bullets([
            "SHAP feature attribution against the detected target for classification and regression goals, gated on a planner directive so it does not run for unsupervised goals.",
            "A third output register (Analyst) between the plain-language and technical forms, closing FR-04.",
            "Live WebSocket streaming of each agent step, surfaced as a real-time trail in the dashboard.",
            "An empirical evaluation harness implementing the same-dataset / different-goal experiment, reproducible from a single command.",
        ]),
        h2("10.6 Documentation"),
        bullets([
            "docs/PROJECT_PROGRESS.md — a full done / partial / pending tracker across all modules, requirements, and data stores.",
            "docs/CONTINUATION_PLAN.md — architecture, current state, known gaps, and a prioritised plan.",
            "evaluation/BASELINE_VALIDATION.md — evidence that the declared baseline matches the real ydata-profiling tool.",
        ]),
    ]

    # ── 10. Testing ─────────────────────────────────────────────────────────
    story += [
        h1("11. Testing & Verification"),
        p("The system is covered by an automated test suite of <b>669 tests</b> spanning the agents, the RAG "
          "layer, the backend services, and the evaluation harness. Coverage includes:"),
        bullets([
            "Ingestion — every file format plus PDF, database, and REST sources, and the source-type router.",
            "Goal-conditioning — assertions that different task types produce different computations and chart sets.",
            "Mining — statistics, correlation, outliers, feature importance, KMeans, DBSCAN, Isolation Forest, class balance, linearity, and identifier-column exclusion.",
            "Feature attribution — that SHAP recovers a planted driver over a pure-noise feature, and that attribution runs for supervised goals and not for unsupervised ones.",
            "Expertise registers — that all three registers are populated and produce measurably different text.",
            "RAG — embeddings, knowledge-base loading, and vector-store retrieval.",
            "Live streaming — WebSocket authentication parity with the REST routes, ordered step delivery, and cross-user dataset isolation.",
            "Evaluation — the divergence, precision/recall, and citation-coverage metrics, including the edge cases where a naive implementation would report a flattering result.",
            "Export — PDF, JSON, and BibTeX citation bundle generation.",
        ]),
        sp(4),
        p("The backend suites require a running PostgreSQL instance; without one they fail on connection rather "
          "than on behaviour."),
    ]

    # ── 11. Deployment ──────────────────────────────────────────────────────
    story += [
        h1("12. Deployment & How to Run"),
        p("The application runs as a three-service Docker Compose stack:"),
        code("cp .env.example .env&nbsp;&nbsp;# then fill in secrets"),
        code("docker compose up --build"),
        sp(4),
        p("This starts the FastAPI backend (port 8000), the Next.js frontend (port 3000), and PostgreSQL "
          "(port 5432). The vector store runs embedded; no external LLM key is required for the current "
          "deterministic pipeline. The user uploads a dataset, states a goal, selects an expertise level, and "
          "receives cited, goal-conditioned recommendations with visualizations, exportable as PDF / JSON."),
    ]

    # ── 12. Future work ─────────────────────────────────────────────────────
    story += [
        h1("13. Future Work"),
        bullets([
            "<b>LLM retry and backoff (NFR-03)</b> — Gemini is now wired for opt-in synthesis, but the client surfaces failures directly rather than retrying with exponential backoff.",
            "<b>Asynchronous job queue</b> — Celery + Redis. The WebSocket layer already delivers live progress, so this is architectural completeness rather than user-facing gain.",
            "<b>Session-scoped data retention (NFR-04)</b> — uploaded datasets are currently kept indefinitely; a TTL-based cleanup tied to the session would close the requirement.",
            "<b>Rate limiting</b> — currently recommended via a reverse proxy in production rather than enforced in the application.",
            "<b>Model-driven orchestration</b> — the Reason/Act/Observe loop is presently a fixed four-step traversal of a statically built plan rather than a model-driven cycle.",
            "<b>Remaining algorithms</b> — time-series decomposition; choropleth charts for geospatial data; LIME as a second attribution method.",
            "<b>Extended ingestion</b> — OCR for scanned documents (deliberately deferred).",
        ]),
    ]

    # ── 13. Conclusion ──────────────────────────────────────────────────────
    story += [
        h1("14. Conclusion"),
        p("MAGE addresses a specific and well-defined gap in the automated-EDA landscape: the absence of a system "
          "in which a natural-language goal conditions which analytical computations are performed, every resulting "
          "recommendation is grounded in retrievable methodology with a traceable citation chain, and the same "
          "analysis is communicated differently depending on the reader's expertise. The implementation delivers a "
          "complete, tested end-to-end pipeline in which the core goal-conditioning claim is not merely "
          "demonstrable but <b>measured</b>: across 25 controlled runs holding the dataset constant and varying "
          "only the goal, MAGE's executed computations diverge at a mean pairwise Jaccard distance of 0.836, "
          "against 0.000 for a generic AutoEDA baseline that has no mechanism to receive a goal at all. Every "
          "recommendation carries a retrievable citation, feature attributions are produced by SHAP against the "
          "user's actual target, output adapts across three expertise registers, and the full agent trail streams "
          "live over a WebSocket. The remaining work — LLM synthesis, an asynchronous job queue, and a larger "
          "knowledge base — extends this foundation rather than completing it."),
        sp(20),
        Paragraph("— End of Report —", ParagraphStyle("END", parent=META, alignment=TA_CENTER)),
    ]

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.85 * inch,
        title="MAGE — Project Report",
        author="Bhagyasree Roy, Nandini Singh, Neha N",
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
