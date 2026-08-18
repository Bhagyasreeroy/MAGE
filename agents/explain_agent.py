"""
agents/explain_agent.py
─────────────────────────
ExplainAgent — LLM-synthesized, RAG-grounded deep explanations.

Answers "explain this further" for a specific, already-computed finding
(a recommendation's own text, a feature-importance result, a correlation).
Always retrieves from the knowledge base first; the LLM's job is narrower
than RecommendationAgent's LLM mode — synthesize across the retrieved
chunks into one coherent explanation, not reason freely. Unlike LLM chat
mode, this is never ungrounded: with no LLM configured (or a failed call)
it still returns the same single-chunk excerpt RecommendationAgent has
always surfaced, just without the narrative synthesis on top.

This is the "richer synthesis across multiple retrieved chunks rather
than a single top chunk" extension recommendation_agent.py's own
docstring flagged as a future milestone.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.llm_client import GeminiClient, LLMError
from agents.recommendation_agent import _clean_technical
from rag.knowledge_loader import KnowledgeBaseLoader
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

TOP_K_RETRIEVAL = 3
MIN_CONFIDENCE = 0.15


class ExplainAgent:
    """Retrieve grounding for a finding, then optionally have the LLM
    synthesize a narrative explanation across the retrieved chunks."""

    def __init__(self, vector_store: VectorStore | None = None, llm_client: GeminiClient | None = None) -> None:
        self._vector_store = vector_store or VectorStore()
        self._llm_client = llm_client or GeminiClient()
        self._kb_loaded = False

    def _ensure_kb_loaded(self) -> None:
        """Lazily populate the vector store — mirrors
        RecommendationAgent._ensure_kb_loaded so both agents share the same
        underlying KB without duplicating a shared abstraction for two
        call sites."""
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

    def explain(self, finding: str, goal: str = "") -> dict[str, Any]:
        """Returns {"explanation": str, "sources": list[str], "synthesized": bool}.
        synthesized=False means the LLM wasn't used (unconfigured, failed,
        or nothing worth citing) — the raw retrieved excerpt is returned
        instead, so this never raises and never returns nothing grounded."""
        self._ensure_kb_loaded()

        query = f"{goal} {finding}".strip()
        hits = self._vector_store.retrieve(query, top_k=TOP_K_RETRIEVAL)
        grounded = [h for h in hits if h["score"] >= MIN_CONFIDENCE]

        if not grounded:
            return {
                "explanation": "Nothing in the knowledge base is closely related to this finding.",
                "sources": [],
                "synthesized": False,
            }

        sources = sorted({h["source"] for h in grounded})

        if self._llm_client.is_configured:
            prompt = self._build_prompt(finding, goal, grounded)
            try:
                text = self._llm_client.generate(prompt)
                return {"explanation": text, "sources": sources, "synthesized": True}
            except LLMError as exc:
                logger.warning("Explain synthesis failed, falling back to raw excerpt: %s", exc)

        # Fallback: today's RecommendationAgent behavior — the single
        # most relevant chunk, cleaned up but not LLM-synthesized.
        top = grounded[0]
        return {
            "explanation": _clean_technical(top["text"]),
            "sources": [top["source"]],
            "synthesized": False,
        }

    @staticmethod
    def _build_prompt(finding: str, goal: str, grounded: list[dict[str, Any]]) -> str:
        excerpts = "\n\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in grounded)
        goal_line = f'The user\'s broader goal is: "{goal}"\n\n' if goal else ""
        return (
            "You explain EDA findings to a data analyst, grounded strictly in the "
            "methodology excerpts provided — do not introduce claims beyond them.\n\n"
            f"Finding: \"{finding}\"\n\n"
            f"{goal_line}"
            f"Relevant methodology excerpts:\n{excerpts}\n\n"
            "Synthesize a single coherent explanation of why this finding matters and "
            "what to do about it, drawing on all the excerpts above rather than just "
            "one. Keep it under ~200 words. Do not repeat the excerpts verbatim — "
            "synthesize them into your own explanation."
        )
