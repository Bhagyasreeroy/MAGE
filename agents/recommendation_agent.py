"""
agents/recommendation_agent.py
───────────────────────────────
RecommendationAgent — RAG-grounded, expertise-adapted EDA recommendations.

Builds a retrieval query from the user's goal plus any upstream findings
(mining patterns, ingestion quality warnings), retrieves grounded
methodology chunks from the RAG knowledge base via VectorStore, and turns
each retrieved chunk into a ranked recommendation with a citation and a
confidence score derived from retrieval similarity.

Recommendation text is produced in two registers:
    - ``text_technical`` : the source methodology passage, verbatim —
      appropriate for analysts / data scientists.
    - ``text_plain``     : a lightly simplified paraphrase (markdown
      stripped, first key sentences) — appropriate for beginners.

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


def _simplify(text: str) -> str:
    """Strip Markdown syntax and return the first couple of complete sentences."""
    cleaned = _MARKDOWN_STRIP_RE.sub(" ", text)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)

    # Overlapping chunks can start mid-sentence (lowercase, no leading
    # capital). Drop that leading fragment so the paraphrase starts clean.
    if len(sentences) > 1 and sentences[0][:1].islower():
        sentences = sentences[1:]

    plain = " ".join(sentences[:2]).strip()
    return plain or cleaned[:200]


class RecommendationAgent:
    """
    Generates RAG-grounded, expertise-adapted EDA recommendations.

    Input context keys consumed:
        - ``goal``                      : analytical goal
        - ``expertise_level``           : adapts language complexity
        - ``MiningAgent_output``        : statistical profile + patterns
        - ``IngestionAgent_output``     : data-quality warnings
        - ``VisualizationAgent_output`` : selected chart specs

    Output keys produced:
        - ``recommendations`` : list of recommendation dicts
        - ``rag_sources``     : list of unique knowledge-base sources cited
    """

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self._vector_store = vector_store or VectorStore()
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
        """
        Generate RAG-grounded recommendations for the current EDA context.

        Parameters
        ----------
        context : dict, optional
            Shared pipeline context forwarded by the Orchestrator.

        Returns
        -------
        dict
            Recommendation payload with grounded suggestions and citations.
        """
        context = context or {}
        goal = context.get("goal", "")
        expertise_level = context.get("expertise_level", "intermediate")
        logger.info("RecommendationAgent.run() | goal=%r expertise=%s", goal, expertise_level)

        self._ensure_kb_loaded()

        query = self._build_query(context)
        if not query.strip():
            return {
                "recommendations": [],
                "rag_sources": [],
                "message": "No goal or upstream findings to ground recommendations in.",
            }

        retrieved = self._vector_store.retrieve(query, top_k=TOP_K_RETRIEVAL)

        recommendations: list[dict[str, Any]] = []
        seen_sources: set[str] = set()

        for chunk in retrieved:
            if chunk["score"] < MIN_CONFIDENCE:
                continue
            source = chunk["source"]
            # One recommendation per source document — avoids multiple
            # near-duplicate chunks from the same doc crowding the list.
            if source in seen_sources:
                continue
            seen_sources.add(source)

            title = chunk["metadata"].get("title", source)
            recommendations.append(
                {
                    "insight": title,
                    "text_technical": chunk["text"].strip(),
                    "text_plain": _simplify(chunk["text"]),
                    "confidence": round(min(max(chunk["score"], 0.0), 1.0), 3),
                    "sources": [source],
                }
            )
            if len(recommendations) >= MAX_RECOMMENDATIONS:
                break

        rag_sources = sorted({s for rec in recommendations for s in rec["sources"]})

        return {
            "recommendations": recommendations,
            "rag_sources": rag_sources,
            "message": f"Generated {len(recommendations)} RAG-grounded recommendation(s).",
        }
