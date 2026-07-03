"""
rag/knowledge_loader.py
────────────────────────
KnowledgeBaseLoader — domain knowledge document ingestion for the RAG pipeline.

Responsibilities:
    • Discover and load documents from the knowledge base directory
      (Markdown files, PDFs, web pages, structured JSON knowledge cards).
    • Split documents into appropriately-sized chunks for embedding.
    • Attach source metadata (file path, section, URL) to each chunk so
      the RecommendationAgent can cite its sources.
    • Upsert processed chunks into the VectorStore via add_documents().

Wiring (future milestones):
    - Use langchain DocumentLoaders (TextLoader, PyPDFLoader, WebBaseLoader).
    - Use langchain RecursiveCharacterTextSplitter for chunking.
    - Call VectorStore.add_documents() after chunking.
    - Maintain a manifest file to avoid re-embedding unchanged documents.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default knowledge base directory (relative to the project root).
DEFAULT_KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"


class KnowledgeBaseLoader:
    """
    Discovers, chunks, and upserts domain knowledge documents into the RAG
    vector store.

    Usage (future)::

        loader = KnowledgeBaseLoader(kb_dir="data/knowledge_base")
        loader.load_all()
    """

    def __init__(self, kb_dir: str | Path = DEFAULT_KB_DIR) -> None:
        self.kb_dir = Path(kb_dir)

    def load_all(self) -> list[dict[str, Any]]:
        """
        Discover all documents in kb_dir, chunk them, and return the chunks.

        Returns
        -------
        list[dict]
            Each dict: {"text": str, "source": str, "metadata": dict}

        TODO:
            - Walk self.kb_dir for .md / .pdf / .txt / .json files.
            - Dispatch to the appropriate DocumentLoader per file type.
            - Chunk with RecursiveCharacterTextSplitter.
            - Upsert to VectorStore.
        """
        logger.info("[STUB] KnowledgeBaseLoader.load_all() — no real docs loaded.")
        return [
            {
                "text": "[DUMMY] EDA best practice: always profile data before modelling.",
                "source": "knowledge_base/eda_best_practices.md",
                "metadata": {"section": "overview"},
            }
        ]

    def load_file(self, path: str | Path) -> list[dict[str, Any]]:
        """
        Load and chunk a single document.

        Parameters
        ----------
        path : str | Path
            Path to the document file.

        Returns
        -------
        list[dict]
            Chunked document entries.

        TODO: Implement per-file-type loading and chunking.
        """
        logger.info("[STUB] load_file() called for path=%s", path)
        return []
