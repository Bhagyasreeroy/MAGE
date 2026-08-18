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

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from rag.embeddings import EMBEDDING_DIM, embed_batch, embed_text

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_PATH = os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")
DEFAULT_FAISS_PATH = os.environ.get("FAISS_INDEX_PATH", "./data/faiss_index")
COLLECTION_NAME = "mage_kb"


def _first_wins(doc_id: str, seen: set[str]) -> bool:
    """Guard against duplicates *within* one batch: chroma rejects a call that
    repeats an id, and FAISS would happily store the chunk twice."""
    if doc_id in seen:
        return True
    seen.add(doc_id)
    return False


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
        self._ids: list[str] = []  # FAISS-only: parallel content-hash ids
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
                # Indexes written before ids existed carry none; derive them
                # so an older on-disk index still de-duplicates correctly.
                self._ids = payload.get("ids") or [
                    self.document_id(d, m) for d, m in zip(self._docs, self._metadatas)
                ]
                logger.info("Loaded FAISS index with %d existing documents.", len(self._docs))
            else:
                # Inner product over L2-normalized vectors == cosine similarity.
                self._store = faiss.IndexFlatIP(EMBEDDING_DIM)
                self._docs = []
                self._metadatas = []
                self._ids = []
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

        # Content-addressed ids make this operation idempotent (NFR-02). Skip
        # anything already indexed *before* embedding: embedding is the
        # expensive half, so filtering after it would still re-encode the whole
        # corpus on every startup for no benefit.
        ids = [self.document_id(t, m) for t, m in zip(documents, metadata)]
        known = self.existing_ids(ids)

        fresh = [
            (i, t, m)
            for i, t, m in zip(ids, documents, metadata)
            if i not in known and not _first_wins(i, known)
        ]
        if not fresh:
            logger.info("No new documents to index — all %d already present.", len(documents))
            return

        new_ids = [f[0] for f in fresh]
        new_docs = [f[1] for f in fresh]
        new_meta = [f[2] for f in fresh]
        embeddings = embed_batch(new_docs)

        if self.backend == "chroma":
            # Chroma metadata values must be str/int/float/bool — coerce, and
            # never pass an empty dict (chroma rejects metadata with no keys).
            safe_metadata = [
                {k: v for k, v in m.items() if v is not None and v != ""} or {"source": "unknown"}
                for m in new_meta
            ]
            # upsert, not add: a re-run with identical ids must not raise and
            # must not duplicate.
            self._store.upsert(
                ids=new_ids,
                embeddings=embeddings.tolist(),
                documents=new_docs,
                metadatas=safe_metadata,
            )
        else:
            normalized = self._normalize(embeddings)
            self._store.add(normalized)
            self._docs.extend(new_docs)
            self._metadatas.extend(new_meta)
            self._ids.extend(new_ids)
            self._persist_faiss()

        logger.info(
            "Indexed %d new document(s) into the %s vector store (%d already present).",
            len(new_docs), self.backend, len(documents) - len(new_docs),
        )

    # ── Identity and incremental indexing (NFR-02) ────────────────────────

    @staticmethod
    def document_id(text: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Stable content-addressed id for one chunk.

        Identity is (source, text). Source is part of it because two knowledge
        base documents may legitimately quote the same sentence, and each still
        needs to be citable in its own right — collapsing them would silently
        drop one document's claim to the finding.

        Because the id is derived rather than random, re-indexing an unchanged
        corpus is a no-op and a changed chunk is a new document.
        """
        source = str((metadata or {}).get("source", ""))
        digest = hashlib.sha256(f"{source}\x00{text}".encode("utf-8")).hexdigest()
        return digest[:32]

    def existing_ids(self, ids: list[str]) -> set[str]:
        """Which of `ids` the store already holds."""
        if not ids:
            return set()
        if self._store is None:
            self.initialize()
        if self.backend == "chroma":
            try:
                found = self._store.get(ids=ids, include=[])
                return set(found.get("ids") or [])
            except Exception as exc:  # noqa: BLE001 - treat as "nothing known"
                logger.warning("Could not read existing ids from chroma: %s", exc)
                return set()
        return set(self._ids) & set(ids)

    def count(self) -> int:
        """Number of documents currently indexed."""
        if self._store is None:
            self.initialize()
        if self.backend == "chroma":
            try:
                return int(self._store.count())
            except Exception:  # noqa: BLE001
                return 0
        return len(self._docs)

    def migrate_legacy_ids(self) -> int:
        """
        Re-key rows that predate content-addressed ids, returning how many.

        Before NFR-02 every row was keyed by a fresh ``uuid4``. Such an id can
        never equal a derived one, so the first sync after the change would
        re-add the whole corpus and every retrieval would return each
        methodology twice. Rather than requiring anyone to rebuild their index
        by hand, drop the mis-keyed rows here and let the normal sync re-insert
        them under the right id.

        Chroma only: FAISS ids are derived on load from the persisted docstore,
        so that backend has nothing to migrate.
        """
        if self.backend != "chroma":
            return 0
        if self._store is None:
            self.initialize()
        try:
            stored = self._store.get(include=["documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001 - a store we cannot read is left alone
            logger.warning("Could not inspect the store for legacy ids: %s", exc)
            return 0

        stale = [
            stored_id
            for stored_id, text, meta in zip(
                stored.get("ids") or [], stored.get("documents") or [], stored.get("metadatas") or []
            )
            if stored_id != self.document_id(text, meta or {})
        ]
        if stale:
            self._store.delete(ids=stale)
            logger.info("Migrated %d legacy vector-store row(s) to content-addressed ids.", len(stale))
        return len(stale)

    def sync_documents(
        self, documents: list[str], metadata: list[dict[str, Any]] | None = None
    ) -> int:
        """
        Index whatever is missing and report how many that was.

        The agent-facing entry point for NFR-02. Callers previously guarded on
        *"is the store non-empty"*, which meant a knowledge-base document added
        after the first run was never indexed at all. Calling this
        unconditionally is correct and cheap: unchanged corpora cost one id
        lookup and no embedding.
        """
        self.migrate_legacy_ids()
        before = self.count()
        self.add_documents(documents, metadata)
        return max(0, self.count() - before)

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
            json.dumps({"docs": self._docs, "metadatas": self._metadatas, "ids": self._ids}),
            encoding="utf-8",
        )
