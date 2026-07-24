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

_MARKDOWN_STRIP_RE = re.compile(r"[#*`|>]|^-\s+", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\s+")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from `obj`, whether it's a dict or a Pydantic/attr object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _drop_leading_fragment(sentences: list[str]) -> list[str]:
    """Overlapping chunks can start mid-sentence (lowercase, no leading
    capital). Drop that leading fragment so text starts clean — used for
    both registers, not just the plain-language one."""
    if len(sentences) > 1 and sentences[0][:1].islower():
        return sentences[1:]
    return sentences


def _clean_technical(text: str) -> str:
    """Verbatim chunk text, minus a truncated leading sentence fragment."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(_drop_leading_fragment(sentences)).strip() or text.strip()


def _simplify(text: str) -> str:
    """Strip Markdown syntax and return the first couple of complete sentences."""
    cleaned = _MARKDOWN_STRIP_RE.sub(" ", text)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    sentences = _drop_leading_fragment(re.split(r"(?<=[.!?])\s+", cleaned))
    plain = " ".join(sentences[:2]).strip()
    return plain or cleaned[:200]


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
