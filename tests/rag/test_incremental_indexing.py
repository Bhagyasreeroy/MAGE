"""
tests/rag/test_incremental_indexing.py
───────────────────────────────────────
NFR-02 — incremental vector indexing, without a full rebuild.

This was recorded as "unverified" but is in fact a defect, in two halves:

  1. ``add_documents`` minted a fresh ``uuid4()`` per document. The docstring
     promised an upsert; the behaviour was an unconditional insert, so
     re-adding the same chunk duplicated it.
  2. Both agents guarded indexing on *"does the store return anything"* — i.e.
     is it non-empty. Once populated, nothing was ever indexed again.

Together those mean **adding a fifteenth knowledge-base document silently never
indexes it**, which quietly undercuts the reported "citation quality improves
with corpus size" result: the corpus cannot actually grow.

The contract these tests pin: a document's identity is its *content*, so
re-indexing an unchanged corpus is a no-op, and a new document costs only its
own chunks.
"""

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
META = [
    {"source": "kb/missing_values.md", "title": "Missing Values"},
    {"source": "kb/outlier_detection.md", "title": "Outliers"},
    {"source": "kb/clustering.md", "title": "Clustering"},
]


@pytest.mark.parametrize("backend", ["chroma", "faiss"])
class TestContentAddressedIds:
    def test_same_text_and_source_yields_the_same_id(self, backend, tmp_path) -> None:
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        a = vs.document_id(DOCS[0], META[0])
        b = vs.document_id(DOCS[0], dict(META[0]))
        assert a == b

    def test_different_text_yields_a_different_id(self, backend, tmp_path) -> None:
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        assert vs.document_id(DOCS[0], META[0]) != vs.document_id(DOCS[1], META[0])

    def test_same_text_from_a_different_source_is_a_different_document(self, backend, tmp_path) -> None:
        """Two documents can legitimately quote the same sentence; each still
        needs its own citation, so source is part of the identity."""
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        assert vs.document_id(DOCS[0], META[0]) != vs.document_id(DOCS[0], META[1])


@pytest.mark.parametrize("backend", ["chroma", "faiss"])
class TestReindexingIsIdempotent:
    def test_adding_the_same_corpus_twice_does_not_grow_the_store(self, backend, tmp_path) -> None:
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        vs.add_documents(DOCS, META)
        first = vs.count()
        vs.add_documents(DOCS, META)
        assert vs.count() == first == len(DOCS)

    def test_retrieval_returns_one_hit_not_duplicates(self, backend, tmp_path) -> None:
        """The user-visible symptom of the bug: the same methodology cited
        three times because it was indexed three times."""
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        for _ in range(3):
            vs.add_documents(DOCS, META)
        hits = vs.retrieve("how do I handle missing values", top_k=5)
        texts = [h["text"] for h in hits]
        assert len(texts) == len(set(texts)), f"duplicate documents retrieved: {texts}"

    def test_a_new_document_costs_only_itself(self, backend, tmp_path) -> None:
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        vs.add_documents(DOCS, META)
        before = vs.count()
        vs.add_documents(
            DOCS + ["Isolation Forest isolates anomalies with random splits."],
            META + [{"source": "kb/anomaly.md", "title": "Anomalies"}],
        )
        assert vs.count() == before + 1

    def test_the_new_document_is_retrievable(self, backend, tmp_path) -> None:
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        vs.add_documents(DOCS, META)
        vs.add_documents(
            ["Isolation Forest isolates anomalies with random splits."],
            [{"source": "kb/anomaly.md", "title": "Anomalies"}],
        )
        hits = vs.retrieve("isolation forest anomalies", top_k=3)
        assert any("Isolation Forest" in h["text"] for h in hits)


@pytest.mark.parametrize("backend", ["chroma", "faiss"])
class TestOnlyNewDocumentsAreEmbedded:
    def test_reindexing_an_unchanged_corpus_embeds_nothing(self, backend, tmp_path, monkeypatch) -> None:
        """Skipping the duplicate *write* but still paying to embed it would
        make every startup re-encode the whole corpus for no reason."""
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        vs.add_documents(DOCS, META)

        import rag.vector_store as vs_mod

        calls: list[int] = []
        real = vs_mod.embed_batch
        monkeypatch.setattr(vs_mod, "embed_batch", lambda texts: (calls.append(len(texts)), real(texts))[1])

        vs.add_documents(DOCS, META)
        assert sum(calls) == 0, f"re-embedded {sum(calls)} unchanged document(s)"

    def test_a_mixed_batch_embeds_only_the_new_one(self, backend, tmp_path, monkeypatch) -> None:
        vs = VectorStore(backend=backend, persist_path=str(tmp_path / backend))
        vs.add_documents(DOCS, META)

        import rag.vector_store as vs_mod

        calls: list[int] = []
        real = vs_mod.embed_batch
        monkeypatch.setattr(vs_mod, "embed_batch", lambda texts: (calls.append(len(texts)), real(texts))[1])

        vs.add_documents(DOCS + ["A brand new methodology note."], META + [{"source": "kb/new.md"}])
        assert sum(calls) == 1


@pytest.mark.parametrize("backend", ["chroma", "faiss"])
class TestIndexSurvivesRestart:
    def test_a_reopened_store_recognises_existing_documents(self, backend, tmp_path) -> None:
        """The whole point of NFR-02: a restart must not re-index the corpus."""
        path = str(tmp_path / backend)
        first = VectorStore(backend=backend, persist_path=path)
        first.add_documents(DOCS, META)
        count = first.count()

        reopened = VectorStore(backend=backend, persist_path=path)
        reopened.initialize()
        assert reopened.count() == count
        reopened.add_documents(DOCS, META)
        assert reopened.count() == count, "re-indexed a corpus that was already persisted"


class TestKnowledgeBaseSync:
    """The agent-facing path: `sync` indexes what is missing and nothing else."""

    def test_sync_reports_what_it_added(self, tmp_path) -> None:
        vs = VectorStore(backend="chroma", persist_path=str(tmp_path / "sync"))
        added = vs.sync_documents(DOCS, META)
        assert added == len(DOCS)

    def test_sync_of_an_unchanged_corpus_adds_nothing(self, tmp_path) -> None:
        vs = VectorStore(backend="chroma", persist_path=str(tmp_path / "sync"))
        vs.sync_documents(DOCS, META)
        assert vs.sync_documents(DOCS, META) == 0

    def test_sync_picks_up_a_document_added_later(self, tmp_path) -> None:
        """The regression this whole ticket exists for: a fifteenth KB document
        used to be invisible because the store was already non-empty."""
        vs = VectorStore(backend="chroma", persist_path=str(tmp_path / "sync"))
        vs.sync_documents(DOCS, META)
        added = vs.sync_documents(
            DOCS + ["Cramer's V measures association between two categorical variables."],
            META + [{"source": "kb/correlation_analysis.md", "title": "Correlation"}],
        )
        assert added == 1
        hits = vs.retrieve("association between categorical variables", top_k=3)
        assert any("Cramer" in h["text"] for h in hits)


class TestAgentsPickUpNewKnowledge:
    """
    The second half of the defect. Even with an idempotent store, both agents
    guarded on *"does the store return anything"*, so once populated they never
    indexed again — a new knowledge-base document stayed invisible.
    """

    def _store(self, tmp_path):
        return VectorStore(backend="chroma", persist_path=str(tmp_path / "agent"))

    def test_agent_picks_up_a_knowledge_base_file_added_after_first_use(
        self, tmp_path, monkeypatch
    ) -> None:
        """
        The exact regression. Drive it through the agent's own guard and a real
        knowledge-base directory, because the guard is what was broken — a
        store-level test would pass even with the bug still in place.
        """
        import agents.recommendation_agent as ra
        from rag.knowledge_loader import KnowledgeBaseLoader
        from agents.recommendation_agent import RecommendationAgent

        kb = tmp_path / "kb"
        kb.mkdir()
        (kb / "clustering.md").write_text(
            "---\ntitle: Clustering\n---\n\n"
            "Select k with the silhouette score; the elbow method is a weaker heuristic.\n",
            encoding="utf-8",
        )
        # Patch the symbol the agent imported, not the module constant:
        # DEFAULT_KB_DIR is bound as a default argument at def time, so
        # rebinding it on the module has no effect on new instances.
        monkeypatch.setattr(ra, "KnowledgeBaseLoader", lambda: KnowledgeBaseLoader(kb_dir=kb))

        vs = self._store(tmp_path)
        agent = RecommendationAgent(vector_store=vs)
        agent._ensure_kb_loaded()
        first = vs.count()
        assert first > 0

        # A fifteenth document lands in the corpus.
        (kb / "correlation_analysis.md").write_text(
            "---\ntitle: Correlation\n---\n\n"
            "Cramer's V measures association between two categorical variables.\n",
            encoding="utf-8",
        )

        # A fresh agent (a restart, in effect) must index it.
        RecommendationAgent(vector_store=vs)._ensure_kb_loaded()
        assert vs.count() > first, "a knowledge-base document added later was never indexed"

        hits = vs.retrieve("association between two categorical variables", top_k=3)
        assert any("Cramer" in h["text"] for h in hits)

    def test_ensure_kb_loaded_is_idempotent_across_agents(self, tmp_path) -> None:
        """Two agents sharing one store must not double-index the corpus."""
        from agents.explain_agent import ExplainAgent
        from agents.recommendation_agent import RecommendationAgent

        vs = self._store(tmp_path)
        RecommendationAgent(vector_store=vs)._ensure_kb_loaded()
        after_first = vs.count()
        ExplainAgent(vector_store=vs)._ensure_kb_loaded()
        assert vs.count() == after_first
        assert after_first > 0


class TestLegacyIndexMigration:
    """
    Stores written before content-addressed ids hold rows keyed by ``uuid4``.
    Those ids can never match a derived one, so the first sync after the change
    would re-add the entire corpus — doubling it, and returning the same
    methodology twice in every retrieval.

    Observed for real: the project's own store went 54 -> 108 with 54 duplicate
    texts on the first run after the change. Self-heal instead, so neither this
    repo nor a teammate's has to be rebuilt by hand.
    """

    def _legacy_store(self, tmp_path):
        """A store populated the old way: random ids, same content."""
        import uuid as _uuid

        from rag.embeddings import embed_batch

        vs = VectorStore(backend="chroma", persist_path=str(tmp_path / "legacy"))
        vs.initialize()
        vs._store.add(
            ids=[str(_uuid.uuid4()) for _ in DOCS],
            embeddings=embed_batch(DOCS).tolist(),
            documents=list(DOCS),
            metadatas=[dict(m) for m in META],
        )
        return vs

    def test_legacy_rows_are_not_duplicated_on_sync(self, tmp_path) -> None:
        vs = self._legacy_store(tmp_path)
        assert vs.count() == len(DOCS)
        vs.sync_documents(DOCS, META)
        assert vs.count() == len(DOCS), "legacy rows were re-added instead of recognised"

    def test_no_duplicate_texts_survive(self, tmp_path) -> None:
        vs = self._legacy_store(tmp_path)
        vs.sync_documents(DOCS, META)
        texts = vs._store.get(include=["documents"])["documents"]
        assert len(texts) == len(set(texts))

    def test_content_is_still_retrievable_after_migration(self, tmp_path) -> None:
        """Migrating must not lose the corpus — the citation chain depends on it."""
        vs = self._legacy_store(tmp_path)
        vs.sync_documents(DOCS, META)
        hits = vs.retrieve("how do I handle missing values", top_k=3)
        assert hits and any("median imputation" in h["text"] for h in hits)

    def test_migration_is_idempotent(self, tmp_path) -> None:
        vs = self._legacy_store(tmp_path)
        vs.sync_documents(DOCS, META)
        n = vs.count()
        vs.sync_documents(DOCS, META)
        assert vs.count() == n
