"""
rag/vector_store.py
────────────────────
VectorStore — unified abstraction over FAISS and ChromaDB.

This module exposes a single VectorStore class that hides the underlying
vector database implementation. The backend to use is controlled by the
``backend`` constructor argument, defaulting to the VECTOR_STORE_BACKEND
environment variable ("chroma" | "faiss").

Responsibilities:
    • Initialize the chosen vector store (create collection / load index).
    • Add documents with embeddings computed via rag/embeddings.py.
    • Retrieve the top-k most similar documents for a query.
    • Persist / reload the index across service restarts.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from rag.embeddings import EMBEDDING_DIM, embed_batch, embed_text

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")
DEFAULT_FAISS_PATH = os.environ.get("FAISS_INDEX_PATH", "./data/faiss_index")
COLLECTION_NAME = "mage_kb"


class VectorStore:
    """
    Unified abstraction over FAISS and ChromaDB vector databases.

    Usage::

        vs = VectorStore(backend="chroma")
        vs.initialize()
        vs.add_documents(["EDA best practice 1", "EDA best practice 2"])
        results = vs.retrieve("How to handle missing values?", top_k=3)
    """

    def __init__(self, backend: str | None = None, persist_path: str | None = None) -> None:
        """
        Parameters
        ----------
        backend : str, optional
            Vector store backend to use. One of "chroma" | "faiss".
            Defaults to the VECTOR_STORE_BACKEND env var, or "chroma".
        persist_path : str, optional
            Directory to persist the index/collection to. Defaults to
            CHROMA_DB_PATH / FAISS_INDEX_PATH depending on backend.
        """
        self.backend = backend or os.environ.get("VECTOR_STORE_BACKEND", "chroma")
        if self.backend not in ("chroma", "faiss"):
            raise ValueError(f"Unsupported vector store backend: {self.backend!r}")

        self.persist_path = persist_path or (
            DEFAULT_CHROMA_PATH if self.backend == "chroma" else DEFAULT_FAISS_PATH
        )
        self._store: Any = None  # chroma Collection, or the FAISS index
        self._docs: list[str] = []  # FAISS-only: parallel document store
        self._metadatas: list[dict[str, Any]] = []  # FAISS-only
        logger.info("VectorStore created with backend=%s path=%s", self.backend, self.persist_path)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create or connect to the vector store."""
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        if self.backend == "chroma":
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_path)
            self._store = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Chroma collection %r ready (%d existing documents).",
                COLLECTION_NAME,
                self._store.count(),
            )
        else:
            import faiss

            index_file = Path(self.persist_path) / "index.faiss"
            docstore_file = Path(self.persist_path) / "docstore.json"

            if index_file.exists() and docstore_file.exists():
                self._store = faiss.read_index(str(index_file))
                payload = json.loads(docstore_file.read_text(encoding="utf-8"))
                self._docs = payload["docs"]
                self._metadatas = payload["metadatas"]
                logger.info("Loaded FAISS index with %d existing documents.", len(self._docs))
            else:
                # Inner product over L2-normalized vectors == cosine similarity.
                self._store = faiss.IndexFlatIP(EMBEDDING_DIM)
                self._docs = []
                self._metadatas = []
                logger.info("Created new FAISS IndexFlatIP(dim=%d).", EMBEDDING_DIM)

    # ── Documents ─────────────────────────────────────────────────────────

    def add_documents(self, documents: list[str], metadata: list[dict] | None = None) -> None:
        """
        Embed and upsert documents into the vector store.

        Parameters
        ----------
        documents : list[str]
            Raw text documents to add to the knowledge base.
        metadata : list[dict], optional
            Per-document metadata dictionaries (e.g. source file, section).
        """
        if not documents:
            return
        if self._store is None:
            self.initialize()

        metadata = metadata or [{} for _ in documents]
        if len(metadata) != len(documents):
            raise ValueError("metadata length must match documents length.")

        embeddings = embed_batch(documents)

        if self.backend == "chroma":
            ids = [str(uuid.uuid4()) for _ in documents]
            # Chroma metadata values must be str/int/float/bool — coerce, and
            # never pass an empty dict (chroma rejects metadata with no keys).
            safe_metadata = [
                {k: v for k, v in m.items() if v is not None and v != ""} or {"source": "unknown"}
                for m in metadata
            ]
            self._store.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=safe_metadata,
            )
        else:
            normalized = self._normalize(embeddings)
            self._store.add(normalized)
            self._docs.extend(documents)
            self._metadatas.extend(metadata)
            self._persist_faiss()

        logger.info("Added %d document(s) to the %s vector store.", len(documents), self.backend)

    # ── Retrieval ─────────────────────────────────────────────────────────

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
            Each dict contains: {"text": str, "score": float, "source": str,
            "metadata": dict}.
        """
        if self._store is None:
            self.initialize()

        if self.backend == "chroma":
            if self._store.count() == 0:
                return []
            query_embedding = embed_text(query)
            result = self._store.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, self._store.count()),
            )
            documents = result["documents"][0]
            metadatas = result["metadatas"][0]
            distances = result["distances"][0]
            return [
                {
                    "text": doc,
                    # Chroma cosine "distance" is 1 - cosine_similarity.
                    "score": 1.0 - dist,
                    "source": meta.get("source", "unknown"),
                    "metadata": meta,
                }
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]
        else:
            if not self._docs:
                return []
            query_embedding = self._normalize(embed_text(query).reshape(1, -1))
            k = min(top_k, len(self._docs))
            scores, indices = self._store.search(query_embedding, k)
            return [
                {
                    "text": self._docs[idx],
                    "score": float(scores[0][pos]),
                    "source": self._metadatas[idx].get("source", "unknown"),
                    "metadata": self._metadatas[idx],
                }
                for pos, idx in enumerate(indices[0])
                if idx != -1
            ]

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows so FAISS inner product == cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def _persist_faiss(self) -> None:
        import faiss

        Path(self.persist_path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._store, str(Path(self.persist_path) / "index.faiss"))
        (Path(self.persist_path) / "docstore.json").write_text(
            json.dumps({"docs": self._docs, "metadatas": self._metadatas}),
            encoding="utf-8",
        )
