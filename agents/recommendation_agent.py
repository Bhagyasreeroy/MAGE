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

Wiring (future milestones):
    - Replace deterministic paraphrasing with an LLM call (e.g. Gemini)
      once an LLM provider/API key is configured, for richer synthesis
      across multiple retrieved chunks rather than a single top chunk
      per recommendation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

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


def _strip_markdown_structure(text: str) -> str:
    """Drop markdown tables and headings from a retrieved knowledge-base chunk.

    Chunks contain markdown tables (``| Goal | Chart |``) and ``|---|---|``
    separator rows. Inlined into a recommendation sentence these flatten into
    unreadable runs of cell text ("numeric Line chart ... Choropleth ..."), so
    we remove whole table lines rather than just stripping the pipe characters.
    Prose before/after the table is preserved.
    """
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Separator rows like |---|:--:| or a bare horizontal rule.
        if stripped and set(stripped) <= set("-|:= "):
            continue
        # Table rows: two or more pipe cell separators.
        if stripped.count("|") >= 2:
            continue
        kept.append(line)
    joined = "\n".join(kept)
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
        - ``MiningAgent_output``        : statistical profile + patterns
        - ``IngestionAgent_output``     : data-quality warnings / row count
        - ``VisualizationAgent_output`` : selected chart specs

    Output keys produced:
        - ``recommendations`` : list of recommendation dicts
        - ``rag_sources``     : list of unique knowledge-base sources cited
    """

    def __init__(self, vector_store: VectorStore | None = None, qa_agent: QAAgent | None = None) -> None:
        self._vector_store = vector_store or VectorStore()
        self._qa_agent = qa_agent or QAAgent()
        self._kb_loaded = False

    def _ensure_kb_loaded(self) -> None:
        """Lazily populate the vector store from the knowledge base on first use."""
        if self._kb_loaded:
            return

        self._vector_store.initialize()
        already_populated = self._vector_store.retrieve("eda methodology", top_k=1)
        if not already_populated:
            chunks = KnowledgeBaseLoader().load_all()
            if chunks:
                self._vector_store.add_documents(
                    [c["text"] for c in chunks],
                    metadata=[{**c["metadata"], "source": c["source"]} for c in chunks],
                )
        self._kb_loaded = True

    def _build_query(self, context: dict[str, Any]) -> str:
        """Compose a goal-conditioned retrieval query from goal + upstream findings."""
        parts = [str(context.get("goal", ""))]

        mining_output = context.get("MiningAgent_output")
        patterns = _get(mining_output, "patterns", []) or []
        parts.extend(str(p) for p in patterns)

        ingestion_output = context.get("IngestionAgent_output")
        warnings = _get(ingestion_output, "warnings", []) or []
        parts.extend(str(w) for w in warnings)

        return " ".join(p for p in parts if p)

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        goal = context.get("goal", "")
        expertise_level = context.get("expertise_level", "intermediate")
        logger.info("RecommendationAgent.run() | goal=%r expertise=%s", goal, expertise_level)

        self._ensure_kb_loaded()

        mining_output = context.get("MiningAgent_output")
        mining_dict = mining_output if isinstance(mining_output, dict) else (
            mining_output.model_dump() if hasattr(mining_output, "model_dump") else {}
        )
        ingestion_output = context.get("IngestionAgent_output")
        ingestion_dict = ingestion_output if isinstance(ingestion_output, dict) else (
            ingestion_output.model_dump() if hasattr(ingestion_output, "model_dump") else {}
        )

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
                "message": "Answered directly from computed statistics.",
            }

        # 2. Finding-led recommendations for broader goals.
        query = self._build_query(context)
        if not query.strip():
            return {
                "recommendations": [],
                "rag_sources": [],
                "message": "No goal or upstream findings to ground recommendations in.",
            }

        recommendations: list[dict[str, Any]] = []
        seen_sources: set[str] = set()

        # 2a. One recommendation per concrete pattern MiningAgent found —
        # led by the specific finding, not a generic chunk.
        patterns = mining_dict.get("patterns") or []
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
            "message": f"Generated {len(recommendations)} RAG-grounded recommendation(s).",
        }
