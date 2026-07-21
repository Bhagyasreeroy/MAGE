"""Tests for rag/vector_store.py."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rag.vector_store import VectorStore

DOCS = [
    "Handle missing values with median imputation for skewed numeric columns.",
    "Use the IQR method to detect univariate outliers in a numeric column.",
    "K-Means clustering requires the number of clusters k to be specified in advance.",
]
METADATA = [
    {"source": "docs/missing_values.md", "title": "Missing Values"},
    {"source": "docs/outliers.md", "title": "Outliers"},
    {"source": "docs/clustering.md", "title": "Clustering"},
]


class TestVectorStoreInit:
    def test_defaults_to_chroma(self, tmp_path) -> None:
        vs = VectorStore(persist_path=str(tmp_path))
        assert vs.backend == "chroma"

    def test_rejects_unknown_backend(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="Unsupported vector store backend"):
            VectorStore(backend="pinecone", persist_path=str(tmp_path))


@pytest.mark.parametrize("backend", ["chroma", "faiss"])
class TestVectorStoreBackends:
    def test_retrieve_before_any_documents_returns_empty(self, backend, tmp_path) -> None:
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        assert vs.retrieve("anything", top_k=3) == []

    def test_add_and_retrieve_returns_most_relevant_document(self, backend, tmp_path) -> None:
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        vs.add_documents(DOCS, metadata=METADATA)

        results = vs.retrieve("how do I choose the number of clusters?", top_k=1)

        assert len(results) == 1
        assert results[0]["source"] == "docs/clustering.md"
        assert 0.0 <= results[0]["score"] <= 1.0

    def test_top_k_limits_result_count(self, backend, tmp_path) -> None:
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        vs.add_documents(DOCS, metadata=METADATA)

        results = vs.retrieve("data analysis methodology", top_k=2)
        assert len(results) == 2

    def test_results_are_sorted_by_descending_score(self, backend, tmp_path) -> None:
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        vs.add_documents(DOCS, metadata=METADATA)

        results = vs.retrieve("outlier detection with IQR", top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_persists_and_reloads_across_instances(self, backend, tmp_path) -> None:
        path = str(tmp_path / backend)
        vs1 = VectorStore(backend=backend, persist_path=path)
        vs1.add_documents(DOCS, metadata=METADATA)

        vs2 = VectorStore(backend=backend, persist_path=path)
        results = vs2.retrieve("clustering algorithm choice", top_k=1)

        assert len(results) == 1
        assert results[0]["source"] == "docs/clustering.md"
