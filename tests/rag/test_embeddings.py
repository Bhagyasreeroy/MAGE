"""Tests for rag/embeddings.py."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rag.embeddings import EMBEDDING_DIM, embed_batch, embed_text


class TestEmbedText:
    def test_returns_correct_shape_and_dtype(self) -> None:
        vec = embed_text("Handle missing values with median imputation.")
        assert vec.shape == (EMBEDDING_DIM,)
        assert vec.dtype == np.float32

    def test_is_deterministic(self) -> None:
        text = "Outlier detection via IQR."
        assert np.array_equal(embed_text(text), embed_text(text))

    def test_similar_texts_are_closer_than_unrelated_ones(self) -> None:
        a = embed_text("How should I impute missing values in a numeric column?")
        b = embed_text("What is the best strategy for handling null values in data?")
        c = embed_text("How do I bake a chocolate cake?")

        def cosine(x: np.ndarray, y: np.ndarray) -> float:
            return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

        assert cosine(a, b) > cosine(a, c)


class TestEmbedBatch:
    def test_empty_list_returns_empty_array(self) -> None:
        result = embed_batch([])
        assert result.shape == (0, EMBEDDING_DIM)

    def test_batch_shape(self) -> None:
        result = embed_batch(["one", "two", "three"])
        assert result.shape == (3, EMBEDDING_DIM)

    def test_batch_matches_individual_embeddings(self) -> None:
        texts = ["missing values", "outlier detection"]
        batch = embed_batch(texts)
        individual = np.stack([embed_text(t) for t in texts])
        assert np.allclose(batch, individual, atol=1e-5)
