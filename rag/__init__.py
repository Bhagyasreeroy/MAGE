"""
rag/
────
MAGE Retrieval-Augmented Generation pipeline.

Exports:
    VectorStore       — FAISS / ChromaDB abstraction
    embed_text        — placeholder embedding function
    KnowledgeBaseLoader — domain knowledge document loader
"""

from rag.vector_store import VectorStore
from rag.embeddings import embed_text
from rag.knowledge_loader import KnowledgeBaseLoader

__all__ = ["VectorStore", "embed_text", "KnowledgeBaseLoader"]
