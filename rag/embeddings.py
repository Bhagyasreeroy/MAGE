"""
rag/embeddings.py
──────────────────
Embedding utilities for the MAGE RAG pipeline.

This module provides the embed_text() function which converts raw text into
a dense vector representation suitable for storage in and retrieval from the
vector store.

Current state: STUB — returns a zero vector of dimension 1536 (OpenAI
ada-002 default) so the rest of the pipeline can be exercised end-to-end
without a live embedding model.

Wiring (future milestones):
    - Replace the stub with a real embedding call:
        Option A: openai.Embedding.create() with text-embedding-ada-002
        Option B: sentence-transformers (local, no API cost)
        Option C: langchain Embeddings interface for swappable backends
    - Add batch embedding support for efficiency.
    - Cache embeddings to avoid redundant API calls.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Embedding dimensionality.  Must match the model used and the vector store index.
EMBEDDING_DIM = 1536  # OpenAI ada-002 default; adjust for other models


def embed_text(text: str) -> np.ndarray:
    """
    Convert a text string into a dense embedding vector.

    Parameters
    ----------
    text : str
        The input text to embed.

    Returns
    -------
    np.ndarray
        A 1-D float32 array of shape (EMBEDDING_DIM,).

    TODO:
        Replace the zero-vector stub with a real embedding model call:

        .. code-block:: python

            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
    """
    logger.debug("[STUB] embed_text() called — returning zero vector for: %r", text[:80])
    return np.zeros(EMBEDDING_DIM, dtype=np.float32)


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of texts into a 2-D array of shape (N, EMBEDDING_DIM).

    Parameters
    ----------
    texts : list[str]
        List of input texts to embed.

    Returns
    -------
    np.ndarray
        Shape (len(texts), EMBEDDING_DIM), float32.

    TODO: Replace stub with batched API call for efficiency.
    """
    logger.debug("[STUB] embed_batch() called with %d texts.", len(texts))
    return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
