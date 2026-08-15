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

Recommendation text is produced in two registers:
    - ``text_technical`` : dense, for analysts / data scientists.
    - ``text_plain``     : plain-language paraphrase, for beginners.

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


def _simplify(text: str) -> str:
    """Plain-language: strip Markdown, return the first couple of complete sentences."""
    cleaned = _strip_markdown_structure(text)
    cleaned = _MARKDOWN_STRIP_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    sentences = _drop_leading_fragment([s for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()])
    plain = " ".join(sentences[:2]).strip()
    return plain or cleaned[:200]


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
                        "text_technical": qa_answer.text,
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
