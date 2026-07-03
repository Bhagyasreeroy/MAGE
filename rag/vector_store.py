"""
rag/vector_store.py
────────────────────
VectorStore — unified abstraction over FAISS and ChromaDB.

This module exposes a single VectorStore class that hides the underlying
vector database implementation.  The backend to use is controlled by the
VECTOR_STORE_BACKEND environment variable ("chroma" | "faiss").

Responsibilities:
    • Initialize the chosen vector store (create collection / load index).
    • Add documents with their pre-computed embeddings.
    • Retrieve the top-k most similar documents for a query embedding.
    • Persist / reload the index across service restarts.

Wiring (future milestones):
    - initialize() will create a real FAISS index or a ChromaDB collection.
    - add_documents() will call embed_text() from rag/embeddings.py and
      upsert the resulting vectors.
    - retrieve() will call embed_text() on the query and run ANN search.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.embeddings import embed_text

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Unified abstraction over FAISS and ChromaDB vector databases.

    Usage (future)::

        vs = VectorStore(backend="chroma")
        vs.initialize()
        vs.add_documents(["EDA best practice 1", "EDA best practice 2"])
        results = vs.retrieve("How to handle missing values?", top_k=3)
    """

    def __init__(self, backend: str = "chroma") -> None:
        """
        Parameters
        ----------
        backend : str
            Vector store backend to use. One of "chroma" | "faiss".
        """
        self.backend = backend
        self._store: Any = None  # Will hold the real client after initialize()
        logger.info("VectorStore created with backend=%s", backend)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Create or connect to the vector store.

        TODO (chroma):  chromadb.Client() → create_collection(name="mage_kb")
        TODO (faiss):   faiss.IndexFlatL2(embedding_dim) and load if exists
        """
        logger.info("[STUB] VectorStore.initialize() — no real store created yet.")
        self._store = {}  # Placeholder

    # ── Documents ─────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[str], metadata: list[dict] | None = None) -> None:
        """
        Embed and upsert documents into the vector store.

        Parameters
        ----------
        documents : list[str]
            Raw text documents to add to the knowledge base.
        metadata : list[dict], optional
            Per-document metadata dictionaries (e.g. source file, section).

        TODO: Call embed_text() per document, then upsert to _store.
        """
        logger.info("[STUB] add_documents() called with %d documents.", len(documents))
        for doc in documents:
            _ = embed_text(doc)  # Placeholder call — result not stored yet

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most relevant documents for a query.

        Parameters
        ----------
        query : str
            Natural-language query to search the knowledge base with.
        top_k : int
            Number of documents to return.

        Returns
        -------
        list[dict]
            Each dict contains at minimum: {"text": str, "score": float, "source": str}.

        TODO: embed_text(query) → ANN search in _store → return real results.
        """
        logger.info("[STUB] retrieve() called | query=%r top_k=%d", query, top_k)
        _ = embed_text(query)  # Placeholder

        # Dummy results — replace with real ANN search results
        return [
            {
                "text": f"[DUMMY] Retrieved document {i + 1} for query: '{query}'",
                "score": 1.0 - i * 0.1,
                "source": f"knowledge_base/doc_{i + 1}.md",
            }
            for i in range(top_k)
        ]
