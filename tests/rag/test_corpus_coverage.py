"""
tests/rag/test_corpus_coverage.py
──────────────────────────────────
Coverage contract for the knowledge base (M5).

The system's grounding claim (FR-03) is only as strong as the corpus behind
it: a recommendation can cite a source only if a relevant one is retrievable
above `MIN_CONFIDENCE`. With five documents, goals outside those five topics
retrieved weakly or got filtered out entirely, so the claim held in form but
thinned in substance.

The important tests here are in :class:`TestComputationsHaveMethodology`.
**Every computation the pipeline actually performs should have methodology it
can cite** — otherwise the system runs a statistic it cannot justify. That is
the specific gap this file was written to close, and it is checked against
`planner.py` directly, so adding a computation without adding methodology for
it will fail the suite rather than quietly degrade grounding.

Written test-first: the coverage assertions failed against the original
five-document corpus.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.recommendation_agent import MIN_CONFIDENCE
from rag.knowledge_loader import KnowledgeBaseLoader
from rag.vector_store import VectorStore

# Minimum corpus size. Not arbitrary: below roughly this many documents the
# retrieval floor filters out most goals that stray from the original topics.
MIN_DOCUMENTS = 12

# Every computation token the planner can emit, paired with a query a user
# might plausibly phrase and the document that should ground it.
#
# The expected source matters more than the score. `MIN_CONFIDENCE` is 0.15,
# which is low enough that almost any query clears it against almost any
# document — before this corpus expansion, "how do I interpret SHAP values"
# retrieved `clustering.md` at 0.202. That is a *false citation*: FR-03 is
# satisfied in form while the cited methodology has nothing to do with the
# recommendation. Asserting the retrieved source is what makes this test
# discriminating rather than vacuous.
COMPUTATION_QUERIES: dict[str, tuple[str, set[str]]] = {
    "class_balance": (
        "how do I handle an imbalanced target class distribution",
        {"class_imbalance.md"},
    ),
    "feature_importance": (
        "which features matter most for predicting my target",
        {"feature_attribution.md", "dimensionality_reduction.md"},
    ),
    "shap_attribution": (
        "how do I interpret SHAP values to explain a model",
        {"feature_attribution.md"},
    ),
    "correlation": (
        "how do I measure the relationship between two variables",
        {"correlation_analysis.md"},
    ),
    "linearity_check": (
        "how do I check regression assumptions and residuals",
        {"regression_diagnostics.md"},
    ),
    "standardize": (
        "should I scale or standardize features before modelling",
        {"feature_engineering.md"},
    ),
    "kmeans": ("how do I choose the number of clusters", {"clustering.md"}),
    "dbscan": (
        "how do I cluster data with irregular shapes and noise",
        {"clustering.md"},
    ),
    "silhouette": ("how do I evaluate cluster quality", {"clustering.md"}),
    "iqr_outliers": (
        "how do I detect outliers in a numeric column",
        {"outlier_detection.md"},
    ),
    "isolation_forest": (
        "how do I detect multivariate anomalies",
        {"outlier_detection.md"},
    ),
    "distribution_tails": (
        "how do I interpret skew and heavy tails",
        {"distribution_profiling.md"},
    ),
    "descriptive_profile": (
        "what summary statistics should I look at first",
        {"distribution_profiling.md"},
    ),
    "missingness": ("how should I handle missing values", {"missing_values.md"}),
    "distribution": (
        "how do I profile the distribution of a column",
        {"distribution_profiling.md"},
    ),
}

# Topics with no coverage at all in the original five-document corpus.
TOPIC_QUERIES: list[tuple[str, set[str]]] = [
    ("how do I choose a classification model", {"classification_models.md"}),
    ("how do I check regression assumptions and residuals", {"regression_diagnostics.md"}),
    ("what is data leakage and how do I avoid it", {"data_leakage.md"}),
    ("how should I split data into train and test sets", {"train_test_methodology.md"}),
    ("how do I reduce dimensionality with PCA", {"dimensionality_reduction.md"}),
    ("how do I engineer useful features", {"feature_engineering.md"}),
    (
        "how do I analyse a time series for trend and seasonality",
        {"time_series_analysis.md"},
    ),
    ("how do I interpret SHAP values", {"feature_attribution.md"}),
]


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    return KnowledgeBaseLoader().load_all()


@pytest.fixture(scope="module")
def store() -> VectorStore:
    """A populated in-memory index over the real corpus."""
    vector_store = VectorStore()
    vector_store.initialize()
    loaded = KnowledgeBaseLoader().load_all()
    if not vector_store.retrieve("eda methodology", top_k=1):
        vector_store.add_documents(
            [c["text"] for c in loaded],
            metadata=[{**c["metadata"], "source": c["source"]} for c in loaded],
        )
    return vector_store


# RecommendationAgent retrieves the top 3 per finding and takes the first hit
# above the floor whose source it has not already cited, so appearing anywhere
# in the top 3 means a document genuinely can ground a recommendation. Checking
# the top 3 rather than only the first result matches how retrieval is actually
# consumed; it is not a relaxation to make the assertions pass.
RETRIEVAL_WINDOW = 3


def _assert_grounds(store: VectorStore, query: str, expected: set[str]) -> None:
    """The expected methodology must be retrievable, above the confidence floor."""
    hits = store.retrieve(query, top_k=RETRIEVAL_WINDOW)
    ranked = [(str(h["source"]).rsplit("/", 1)[-1], h["score"]) for h in hits]

    match = next(((s, sc) for s, sc in ranked if s in expected), None)
    assert match is not None, (
        f"{query!r} did not retrieve any of {sorted(expected)} in its top "
        f"{RETRIEVAL_WINDOW}; got {[s for s, _ in ranked]} — a recommendation "
        f"on this topic would cite unrelated methodology"
    )

    source, score = match
    assert score >= MIN_CONFIDENCE, (
        f"{query!r} retrieved {source!r} but only at {score:.3f}, below the "
        f"{MIN_CONFIDENCE} floor, so it would be filtered out before citation"
    )


class TestCorpusSize:
    def test_corpus_meets_the_minimum_document_count(self, chunks: list[dict]) -> None:
        sources = {c["source"] for c in chunks}
        assert len(sources) >= MIN_DOCUMENTS, f"only {len(sources)} documents"

    def test_corpus_produces_a_reasonable_number_of_chunks(self, chunks: list[dict]) -> None:
        assert len(chunks) >= MIN_DOCUMENTS * 2


class TestEveryDocumentIsWellFormed:
    def test_every_chunk_declares_a_title(self, chunks: list[dict]) -> None:
        for chunk in chunks:
            assert chunk["metadata"].get("title"), f"{chunk['source']} has no title"

    def test_every_chunk_declares_a_doc_type(self, chunks: list[dict]) -> None:
        for chunk in chunks:
            assert chunk["metadata"].get("doc_type"), f"{chunk['source']} has no doc_type"

    def test_every_chunk_declares_a_section(self, chunks: list[dict]) -> None:
        for chunk in chunks:
            assert chunk["metadata"].get("section"), f"{chunk['source']} has no section"

    def test_titles_are_unique_per_source(self, chunks: list[dict]) -> None:
        """Citations name the title, so two documents sharing one is ambiguous."""
        by_source = {c["source"]: c["metadata"]["title"] for c in chunks}
        assert len(set(by_source.values())) == len(by_source)

    def test_no_chunk_is_trivially_short(self, chunks: list[dict]) -> None:
        for chunk in chunks:
            assert len(chunk["text"]) > 80, f"{chunk['source']} has a near-empty chunk"


class TestComputationsHaveMethodology:
    """
    The core contract: MAGE should not run a statistic it cannot justify.

    Each computation the planner can emit must have retrievable methodology,
    so a recommendation mentioning it can carry a citation.
    """

    @pytest.mark.parametrize(
        "computation,query,expected",
        [(name, q, exp) for name, (q, exp) in sorted(COMPUTATION_QUERIES.items())],
    )
    def test_computation_is_grounded_in_the_right_document(
        self, store: VectorStore, computation: str, query: str, expected: set[str]
    ) -> None:
        _assert_grounds(store, query, expected)

    def test_every_planner_computation_is_covered_by_this_file(self) -> None:
        """Adding a computation without methodology should fail here, not silently."""
        from agents.planner import _MINING_DIRECTIVES

        planned = {
            token
            for directives in _MINING_DIRECTIVES.values()
            for token in directives["computations"]
        }
        missing = planned - set(COMPUTATION_QUERIES)
        assert not missing, f"computations with no coverage query: {sorted(missing)}"


class TestTopicsBeyondTheOriginalFive:
    """
    Topics the original corpus could not ground at all.

    These are the additions that make the corpus useful past its first five
    subjects — chosen to match what the pipeline computes and what a user
    doing supervised work will actually ask.
    """

    @pytest.mark.parametrize("query,expected", TOPIC_QUERIES)
    def test_topic_is_grounded_in_the_right_document(
        self, store: VectorStore, query: str, expected: set[str]
    ) -> None:
        _assert_grounds(store, query, expected)
