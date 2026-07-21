"""
rag/embeddings.py
──────────────────
Embedding utilities for the MAGE RAG pipeline.

Uses a local sentence-transformers model (no API key required) so the RAG
pipeline works out of the box in development. The model is loaded lazily
and cached at module scope — the first call pays the load cost, every
subsequent call reuses the same in-memory model.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# all-MiniLM-L6-v2: 384-dim, ~80MB, strong quality/speed tradeoff for
# short methodology passages. Swap via EMBEDDING_MODEL_NAME if a larger
# model is preferred in production.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Lazily load and cache the sentence-transformers model (thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %r...", EMBEDDING_MODEL_NAME)
                _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


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
    """
    return embed_batch([text])[0]


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
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)
