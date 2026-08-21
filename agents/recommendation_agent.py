"""
agents/recommendation_agent.py
───────────────────────────────
RecommendationAgent — RAG-grounded, expertise-adapted EDA recommendations.

Two paths, tried in order:

1. **Direct answer** (QAAgent): if the goal is a specific factual question
   ("which column has the most missing values?", "what's the correlation
   between X and Y?"), answer it directly from MiningAgent's already-computed
   output — no retrieval needed. This is what makes follow-up chat feel
   conversational instead of re-dumping a wall of methodology text for every
   message.

2. **Finding-led recommendations** (fallback, for broader goals): for each
   concrete pattern MiningAgent found (a specific outlier count, a specific
   correlation, a specific skew), retrieve the one most relevant methodology
   chunk and lead the recommendation with the concrete finding, e.g.
   "'revenue' has 3 outliers (IQR method). <methodology excerpt>" — rather
   than surfacing the raw excerpt alone, which reads as generic textbook
   material disconnected from the user's actual data.

Recommendation text is produced in three registers (FR-04), one per expertise
level the system offers:
    - ``text_plain``     : plain language, no jargon — **Beginner**.
    - ``text_analyst``   : the finding plus the statistic supporting it, in
      working language but without full methodological framing — **Analyst**.
    - ``text_technical`` : the finding in context with the complete
      methodology excerpt, markdown intact — **Data Scientist**.

The three are deliberately nested rather than independently written: each
level adds detail the one below omits, so a reader moving up a level never
loses information they had. ``ANALYST_SENTENCES`` and ``PLAIN_SENTENCES`` set
where each stops.

**LLM mode** (opt-in, ``context["mode"] == "llm"``): bypasses both paths
above entirely. Instead of retrieval, the dataset's already-computed
features (MiningAgent's stats/patterns, IngestionAgent's warnings — never
raw rows) and the user's goal are handed to Gemini for a freeform,
conversational response. Deliberately not grounded — LLM recommendations
always carry empty ``sources``, which is what keeps them visually distinct
from RAG output (no citation chips) rather than something to "fix" later.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agents.llm_client import GeminiClient, LLMError
from agents.mining_agent import ATTRIBUTION_PATTERN_MARKER
from agents.qa_agent import QAAgent
from rag.knowledge_loader import KnowledgeBaseLoader
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Recommendations to return per run. Keep small — these are meant to be
# the highest-signal, most goal-relevant grounded suggestions, not an
# exhaustive dump of the knowledge base.
TOP_K_RETRIEVAL = 8
MAX_RECOMMENDATIONS = 5

# Chunks below this similarity are considered too weak to ground a
# recommendation in and are dropped rather than surfaced with low confidence.
MIN_CONFIDENCE = 0.15

# How much of the retrieved methodology each register keeps. The technical
# register is uncapped (it keeps the whole excerpt), so these two are what
# separate the three levels — keep them distinct or the registers converge and
# FR-04's "visibly alters output language" stops being true.
PLAIN_SENTENCES = 2
ANALYST_SENTENCES = 4

_MARKDOWN_STRIP_RE = re.compile(r"[#*`|>]|^-\s+", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\s+")
_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s*")
_INLINE_MD_RE = re.compile(r"[*`_]")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from `obj`, whether it's a dict or a Pydantic/attr object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _is_separator_row(line: str) -> bool:
    """True for ``|---|:--:|`` separators and bare horizontal rules."""
    return bool(line) and set(line) <= set("-|:= ")


def _split_row(line: str) -> list[str]:
    """Split one markdown table row into stripped cell values."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_block_to_sentences(block: list[str]) -> list[str]:
    """
    Rewrite one markdown table as one labelled sentence per row.

    ``| Goal | Chart |`` / ``| Compare groups | Box plot |`` becomes
    ``Goal: Compare groups; Chart: Box plot.`` — the header row supplies the
    field names rather than being emitted as content.

    This is the point of the whole function. Several knowledge-base documents
    carry their real guidance as a decision table, so dropping tables meant
    those documents contributed only their headings. Inlining the raw cells
    instead produces an unreadable run ("numeric Line chart … Choropleth …"),
    which is why the values are labelled.
    """
    rows = [_split_row(line) for line in block if not _is_separator_row(line)]
    if not rows:
        return []

    headers, data_rows = rows[0], rows[1:]
    if not data_rows:
        # A single row with no body is a header with nothing under it; keep the
        # values rather than inventing labels for them.
        return [" ".join(cell for cell in headers if cell)]

    sentences: list[str] = []
    for cells in data_rows:
        parts: list[str] = []
        for index, cell in enumerate(cells):
            if not cell:
                continue  # an empty cell labelled "Header:" is noise
            label = headers[index] if index < len(headers) else ""
            parts.append(f"{label}: {cell}" if label else cell)
        if parts:
            sentences.append("; ".join(parts) + ".")
    return sentences


def _flatten_markdown_tables(text: str) -> str:
    """Replace every markdown table in `text` with labelled per-row sentences."""
    out: list[str] = []
    block: list[str] = []

    def flush() -> None:
        if block:
            out.extend(_table_block_to_sentences(block))
            block.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.count("|") >= 2:
            block.append(stripped)
        elif _is_separator_row(stripped):
            # Inside a table this is the header separator; outside it is a bare
            # horizontal rule, which carries nothing worth keeping.
            if block:
                block.append(stripped)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def _strip_markdown_structure(text: str) -> str:
    """Convert a retrieved knowledge-base chunk into plain readable prose.

    Tables are rewritten row-by-row into labelled sentences (see
    :func:`_table_block_to_sentences`) rather than discarded; headings and
    inline emphasis markers are removed. Prose before and after a table is
    preserved in place.
    """
    joined = _flatten_markdown_tables(text)
    joined = _HEADING_RE.sub("", joined)      # drop "## Heading" markers
    joined = _INLINE_MD_RE.sub("", joined)    # drop * ` _ emphasis
    return joined


def _drop_leading_fragment(sentences: list[str]) -> list[str]:
    """Overlapping chunks can start mid-sentence (lowercase, no leading
    capital). Drop that leading fragment so text starts clean — used for
    both registers, not just the plain-language one."""
    while len(sentences) > 1 and sentences[0].strip()[:1].islower():
        sentences = sentences[1:]
    return sentences


def _clean_technical(text: str) -> str:
    """Readable chunk prose: markdown tables/headings removed, leading
    sentence fragment dropped, whitespace normalised."""
    cleaned = _strip_markdown_structure(text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", cleaned.strip()) if s.strip()]
    result = " ".join(_drop_leading_fragment(sentences)).strip()
    result = _WHITESPACE_RE.sub(" ", result)
    return result or _WHITESPACE_RE.sub(" ", cleaned).strip() or text.strip()


def _first_sentences(text: str, limit: int) -> str:
    """Strip Markdown and return the first `limit` complete sentences."""
    cleaned = _strip_markdown_structure(text)
    cleaned = _MARKDOWN_STRIP_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    sentences = _drop_leading_fragment([s for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()])
    trimmed = " ".join(sentences[:limit]).strip()
    return trimmed or cleaned[:200]


def _simplify(text: str) -> str:
    """Plain-language register (Beginner): Markdown stripped, first couple of sentences."""
    return _first_sentences(text, PLAIN_SENTENCES)


def _analyst(text: str, title: str) -> str:
    """
    Analyst register: working language, attributed, with the reasoning kept.

    Two things separate this from the plain register:

    * **More of the excerpt** — ``ANALYST_SENTENCES`` rather than
      ``PLAIN_SENTENCES``, so the reasoning behind the advice survives and not
      just its headline. Markdown is still stripped; an analyst wants the
      substance, not the document structure.
    * **Inline attribution** — the methodology is named in the sentence. This
      is not decoration: it is the difference between "handle the missing
      values" and knowing *which* documented procedure is being invoked, which
      is precisely what a working analyst needs in order to check it.

    The attribution also makes the separation *structural* rather than merely
    a matter of length. Several knowledge-base chunks are a single sentence
    long, and against those a purely length-based rule collapses this register
    onto the plain one — which would quietly break FR-04 for exactly the
    recommendations whose source is shortest.
    """
    body = _first_sentences(text, ANALYST_SENTENCES)
    return f"Per {title}: {body}" if title else body


class RecommendationAgent:
    """
    Generates RAG-grounded, expertise-adapted EDA recommendations — or a
    direct computed answer when the goal is a specific factual question.

    Input context keys consumed:
        - ``goal``                      : analytical goal
        - ``expertise_level``           : adapts language complexity
        - ``mode``                      : "rag" (default) | "llm"
        - ``MiningAgent_output``        : statistical profile + patterns
        - ``IngestionAgent_output``     : data-quality warnings / row count
        - ``VisualizationAgent_output`` : selected chart specs

    Output keys produced:
        - ``recommendations`` : list of recommendation dicts
        - ``rag_sources``     : list of unique knowledge-base sources cited
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        qa_agent: QAAgent | None = None,
        llm_client: GeminiClient | None = None,
    ) -> None:
        self._vector_store = vector_store or VectorStore()
        self._qa_agent = qa_agent or QAAgent()
        self._llm_client = llm_client or GeminiClient()
        self._kb_loaded = False

    def _ensure_kb_loaded(self) -> None:
        """Lazily populate the vector store from the knowledge base on first use."""
        if self._kb_loaded:
            return

        self._vector_store.initialize()
        # Sync unconditionally rather than skipping when the store is
        # non-empty. The old "is anything in here?" guard meant a knowledge
        # base document added after the first ever run was never indexed —
        # invisible, with no error. Sync is content-addressed (NFR-02), so an
        # unchanged corpus costs one id lookup and embeds nothing.
        chunks = KnowledgeBaseLoader().load_all()
        if chunks:
            added = self._vector_store.sync_documents(
                [c["text"] for c in chunks],
                metadata=[{**c["metadata"], "source": c["source"]} for c in chunks],
            )
            if added:
                logger.info("Indexed %d new knowledge-base chunk(s).", added)
        self._kb_loaded = True

    @staticmethod
    def _prior_runs(context: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Prior-run memory from the context, defensively normalised.

        Supplied by the service layer, which owns the database. Anything
        malformed is discarded rather than raised on: memory is additive
        grounding, and a bad memory row must not be able to fail an analysis.
        """
        prior = context.get("prior_runs")
        if not isinstance(prior, list):
            return []
        return [run for run in prior if isinstance(run, dict)]

    def _build_query(self, context: dict[str, Any]) -> str:
        """Compose a goal-conditioned retrieval query from goal + upstream findings."""
        parts = [str(context.get("goal", ""))]

        mining_output = context.get("MiningAgent_output")
        patterns = _get(mining_output, "patterns", []) or []
        parts.extend(str(p) for p in patterns)

        ingestion_output = context.get("IngestionAgent_output")
        warnings = _get(ingestion_output, "warnings", []) or []
        parts.extend(str(w) for w in warnings)

        # What this user previously found on comparable questions (Objective 4).
        # Appended last so the current goal and this run's own findings still
        # dominate the query — history informs retrieval, it does not replace
        # the present question.
        for run in self._prior_runs(context):
            parts.extend(str(f) for f in (run.get("findings") or []))

        return " ".join(p for p in parts if p)

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        goal = context.get("goal", "")
        expertise_level = context.get("expertise_level", "intermediate")
        mode = context.get("mode", "rag")
        logger.info("RecommendationAgent.run() | goal=%r expertise=%s mode=%s", goal, expertise_level, mode)

        mining_output = context.get("MiningAgent_output")
        mining_dict = mining_output if isinstance(mining_output, dict) else (
            mining_output.model_dump() if hasattr(mining_output, "model_dump") else {}
        )
        ingestion_output = context.get("IngestionAgent_output")
        ingestion_dict = ingestion_output if isinstance(ingestion_output, dict) else (
            ingestion_output.model_dump() if hasattr(ingestion_output, "model_dump") else {}
        )

        if mode == "llm":
            return self._llm_recommend(goal, expertise_level, mining_dict, ingestion_dict)

        self._ensure_kb_loaded()

        # 1. Direct answer for specific factual questions — no RAG dump.
        qa_answer = self._qa_agent.try_answer(goal, mining_dict, ingestion_dict)
        if qa_answer is not None:
            sources = [qa_answer.source] if qa_answer.source else []
            return {
                "recommendations": [
                    {
                        "insight": "Computed from your data",
                        # A direct factual answer is a computed number — there
                        # is no methodology excerpt to expand or compress, so
                        # all three registers are deliberately identical here.
                        # Paraphrasing a statistic per audience would change
                        # the wording without changing the information, which
                        # is presentation, not adaptation.
                        "text_technical": qa_answer.text,
                        "text_analyst": qa_answer.text,
                        "text_plain": qa_answer.text,
                        "confidence": 1.0,
                        "sources": sources,
                    }
                ],
                "rag_sources": sources,
                "prior_runs": self._prior_runs(context),
                "message": "Answered directly from computed statistics.",
            }

        # 2. Finding-led recommendations for broader goals.
        query = self._build_query(context)
        if not query.strip():
            return {
                "recommendations": [],
                "rag_sources": [],
                "prior_runs": self._prior_runs(context),
                "message": "No goal or upstream findings to ground recommendations in.",
            }

        recommendations: list[dict[str, Any]] = []
        seen_sources: set[str] = set()

        # 2a. One recommendation per concrete pattern MiningAgent found —
        # led by the specific finding, not a generic chunk. The orchestrator's
        # post-Mining reflection (OrchestratorAgent._reflect_on_mining) can
        # adjust this list before it's used: a target-completeness warning is
        # prepended so it leads (patterns are processed in order, and this
        # still grounds naturally via RAG, e.g. against missing_values.md),
        # and the attribution pattern is dropped entirely when the fitted
        # model scored too low to trust — already flagged in the ReAct log,
        # so it should not also be presented as a top finding here.
        directives = context.get("directives", {}) or {}
        patterns = list(mining_dict.get("patterns") or [])
        if directives.get("attribution_trusted") is False:
            patterns = [p for p in patterns if ATTRIBUTION_PATTERN_MARKER not in p]
        target_quality_warning = directives.get("target_quality_warning")
        if target_quality_warning:
            patterns.insert(0, target_quality_warning)

        for pattern in patterns:
            if len(recommendations) >= MAX_RECOMMENDATIONS:
                break
            hits = self._vector_store.retrieve(f"{goal} {pattern}", top_k=3)
            hit = next((h for h in hits if h["score"] >= MIN_CONFIDENCE and h["source"] not in seen_sources), None)
            if hit is None:
                continue
            seen_sources.add(hit["source"])
            recommendations.append(
                {
                    "insight": hit["metadata"].get("title", hit["source"]),
                    # Blank line between the finding and the chunk — joining
                    # with a plain space would put the chunk's own leading
                    # "# Heading" mid-line, where Markdown can't recognize
                    # it as a heading anymore.
                    "text_technical": f"**{pattern}**\n\n{_clean_technical(hit['text'])}",
                    "text_analyst": (
                        f"{pattern} {_analyst(hit['text'], hit['metadata'].get('title', ''))}"
                    ),
                    "text_plain": f"{pattern} {_simplify(hit['text'])}",
                    "confidence": round(min(max(hit["score"], 0.0), 1.0), 3),
                    "sources": [hit["source"]],
                }
            )

        # 2b. Fill any remaining slots with a broad goal-only retrieval
        # (no specific finding to lead with — the original behavior).
        if len(recommendations) < MAX_RECOMMENDATIONS:
            hits = self._vector_store.retrieve(query, top_k=TOP_K_RETRIEVAL)
            for hit in hits:
                if len(recommendations) >= MAX_RECOMMENDATIONS:
                    break
                if hit["score"] < MIN_CONFIDENCE or hit["source"] in seen_sources:
                    continue
                seen_sources.add(hit["source"])
                recommendations.append(
                    {
                        "insight": hit["metadata"].get("title", hit["source"]),
                        "text_technical": _clean_technical(hit["text"]),
                        "text_analyst": _analyst(hit["text"], hit["metadata"].get("title", "")),
                        "text_plain": _simplify(hit["text"]),
                        "confidence": round(min(max(hit["score"], 0.0), 1.0), 3),
                        "sources": [hit["source"]],
                    }
                )

        rag_sources = sorted({s for rec in recommendations for s in rec["sources"]})

        return {
            "recommendations": recommendations,
            "rag_sources": rag_sources,
            # Surfaced for display, deliberately separate from `rag_sources`:
            # a prior run is context the user can recognise, not retrievable
            # methodology, and FR-03's citation promise covers only the latter.
            "prior_runs": self._prior_runs(context),
            "message": f"Generated {len(recommendations)} RAG-grounded recommendation(s).",
        }

    # ── LLM mode ─────────────────────────────────────────────────────────

    def _llm_recommend(
        self,
        goal: str,
        expertise_level: str,
        mining_dict: dict[str, Any],
        ingestion_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Freeform response from Gemini, grounded only in already-computed
        aggregate stats (never raw rows). Never raises — a missing key or a
        failed request degrades to a single explanatory recommendation
        instead of breaking the whole pipeline step."""
        if not self._llm_client.is_configured:
            text = (
                "LLM mode isn't set up yet — it needs a Gemini API key configured "
                "on the server (GEMINI_API_KEY). Switch back to RAG mode for now."
            )
            return {
                "recommendations": [
                    {
                        "insight": "LLM mode unavailable",
                        "text_technical": text,
                        "text_plain": text,
                        "confidence": 0.0,
                        "sources": [],
                    }
                ],
                "rag_sources": [],
                "message": "LLM mode not configured.",
            }

        prompt = self._build_llm_prompt(goal, expertise_level, mining_dict, ingestion_dict)
        try:
            text = self._llm_client.generate(prompt)
        except LLMError as exc:
            logger.warning("LLM recommendation failed: %s", exc)
            text = f"The LLM request failed ({exc}). Try again, or switch to RAG mode."

        return {
            "recommendations": [
                {
                    "insight": "LLM-generated response",
                    "text_technical": text,
                    "text_plain": text,
                    "confidence": 1.0,
                    "sources": [],
                }
            ],
            "rag_sources": [],
            "message": "Generated via Gemini (not grounded in the knowledge base).",
        }

    def _build_llm_prompt(
        self,
        goal: str,
        expertise_level: str,
        mining_dict: dict[str, Any],
        ingestion_dict: dict[str, Any],
    ) -> str:
        """Compose a prompt from already-computed aggregate statistics —
        never raw rows, since this leaves the server as part of the LLM
        call."""
        statistics = mining_dict.get("statistics") or {}
        data_quality = mining_dict.get("data_quality") or {}
        patterns = mining_dict.get("patterns") or []
        correlations = mining_dict.get("correlations") or {}
        warnings = ingestion_dict.get("warnings") or []

        column_lines = []
        for name, stat in statistics.items():
            dq = data_quality.get(name, {})
            missing_pct = round(100 - dq.get("completeness_pct", 100), 1)
            if stat.get("type") == "numeric":
                column_lines.append(
                    f"- {name} (numeric): mean={stat.get('mean')}, min={stat.get('min')}, "
                    f"max={stat.get('max')}, {missing_pct}% missing"
                )
            else:
                column_lines.append(f"- {name} (categorical): {missing_pct}% missing")

        top_correlations = []
        seen_pairs: set[frozenset[str]] = set()
        for col_a, row in correlations.items():
            for col_b, r in row.items():
                if col_a == col_b or r is None:
                    continue
                pair = frozenset((col_a, col_b))
                if pair in seen_pairs or abs(r) < 0.5:
                    continue
                seen_pairs.add(pair)
                top_correlations.append(f"- {col_a} vs {col_b}: r={r:.2f}")

        sections = [
            "You are a data analysis assistant helping a user explore a dataset they've uploaded.",
            f"\nColumns:\n" + ("\n".join(column_lines) or "(no column profile available)"),
        ]
        if patterns:
            sections.append("\nNotable patterns already detected:\n" + "\n".join(f"- {p}" for p in patterns))
        if top_correlations:
            sections.append("\nStrong correlations (|r| >= 0.5):\n" + "\n".join(top_correlations))
        if warnings:
            sections.append("\nData quality warnings:\n" + "\n".join(f"- {w}" for w in warnings))
        sections.append(
            f'\nUser\'s question: "{goal}"\n\n'
            f"Answer conversationally, referencing the specific numbers above where relevant. "
            f"Calibrate depth for a {expertise_level} audience. Keep it under ~200 words."
        )
        return "\n".join(sections)
