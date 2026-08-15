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


def warm_up() -> bool:
    """
    Load the model now instead of on first use.

    The lazy load costs roughly ten seconds. Left to happen on demand, that
    cost lands on whichever request needs an embedding first after a restart —
    which, in the live analysis stream, shows up as a long stall part-way
    through a run. Calling this at application startup moves it to boot time,
    where nothing is waiting on it.

    Blocking, so callers on an event loop should hand it to a worker thread.

    Returns
    -------
    bool
        True if the model is loaded and ready; False if loading failed, in
        which case the normal lazy path will simply try again on first use.
    """
    try:
        _get_model()
    except Exception:  # noqa: BLE001 - warm-up is an optimisation, never fatal
        logger.warning("Embedding model warm-up failed; will load on first use.", exc_info=True)
        return False
    return True


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
