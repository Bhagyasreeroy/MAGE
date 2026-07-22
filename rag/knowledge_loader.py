"""
rag/knowledge_loader.py
────────────────────────
KnowledgeBaseLoader — domain knowledge document ingestion for the RAG pipeline.

Discovers Markdown documents under a knowledge base directory, splits each
into overlapping chunks along paragraph/sentence boundaries, and attaches
source metadata (file path, title, section) so the RecommendationAgent can
cite exactly which document and section grounded each recommendation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default knowledge base directory (relative to the project root).
DEFAULT_KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"

# Chunking parameters. Small values suit short methodology paragraphs;
# the overlap keeps context from being severed mid-idea.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """
    Split a Markdown file's optional `--- key: value ---` frontmatter block
    from its body. Returns (metadata, body). If no frontmatter block is
    present, metadata is empty and body is the full raw text.
    """
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    header, body = match.group(1), match.group(2)
    metadata: dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip()
    return metadata, body


def _is_markdown_table(paragraph: str) -> bool:
    """True if most non-blank lines in the paragraph look like a table row."""
    lines = [line for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return False
    table_lines = sum(1 for line in lines if line.strip().startswith("|"))
    return table_lines / len(lines) >= 0.6


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into overlapping chunks, preferring paragraph boundaries
    (blank lines) and falling back to sentence boundaries when a single
    paragraph exceeds chunk_size.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    def seed_overlap() -> None:
        """Flush `current`, seeding the next chunk with a word-aligned tail
        for context continuity — unless `current` is a table, since a
        trailing slice of a table is not useful leading context."""
        nonlocal current
        tail = ""
        if current and chunk_overlap and not _is_markdown_table(current):
            tail = current[-chunk_overlap:]
            first_space = tail.find(" ")
            tail = tail[first_space + 1 :] if first_space != -1 else ""
        flush()
        current = tail

    for paragraph in paragraphs:
        # Tables are never merged with unrelated preceding prose, and never
        # split mid-row — both produce nonsensical fragments out of context.
        if _is_markdown_table(paragraph):
            if current:
                seed_overlap()
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        seed_overlap()

        if len(paragraph) <= chunk_size:
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
        else:
            # Paragraph itself is too long — split on sentence boundaries.
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            for sentence in sentences:
                candidate = f"{current} {sentence}".strip() if current else sentence
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    flush()
                    current = sentence

    flush()
    return chunks


class KnowledgeBaseLoader:
    """
    Discovers, chunks, and returns domain knowledge documents for the RAG
    vector store.

    Usage::

        loader = KnowledgeBaseLoader(kb_dir="data/knowledge_base")
        chunks = loader.load_all()
        vector_store.add_documents(
            [c["text"] for c in chunks],
            metadata=[c["metadata"] for c in chunks],
        )
    """

    def __init__(
        self,
        kb_dir: str | Path = DEFAULT_KB_DIR,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.kb_dir = Path(kb_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_all(self) -> list[dict[str, Any]]:
        """
        Discover all Markdown documents in kb_dir, chunk them, and return
        the chunks.

        Returns
        -------
        list[dict]
            Each dict: {"text": str, "source": str, "metadata": dict}
        """
        if not self.kb_dir.exists():
            logger.warning("Knowledge base directory does not exist: %s", self.kb_dir)
            return []

        chunks: list[dict[str, Any]] = []
        for path in sorted(self.kb_dir.glob("*.md")):
            chunks.extend(self.load_file(path))

        logger.info("KnowledgeBaseLoader loaded %d chunk(s) from %s", len(chunks), self.kb_dir)
        return chunks

    def load_file(self, path: str | Path) -> list[dict[str, Any]]:
        """
        Load and chunk a single Markdown document.

        Parameters
        ----------
        path : str | Path
            Path to the document file.

        Returns
        -------
        list[dict]
            Chunked document entries, each with text/source/metadata.
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.warning("Knowledge base file not found: %s", file_path)
            return []

        raw = file_path.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(raw)
        title = frontmatter.get("title", file_path.stem.replace("_", " ").title())

        source = str(file_path.relative_to(self.kb_dir)) if file_path.is_relative_to(self.kb_dir) else file_path.name

        text_chunks = _split_text(body, self.chunk_size, self.chunk_overlap)

        return [
            {
                "text": chunk,
                "source": f"knowledge_base/{source}",
                "metadata": {
                    "title": title,
                    "doc_type": frontmatter.get("doc_type", ""),
                    "section": frontmatter.get("section", ""),
                    "chunk_index": i,
                },
            }
            for i, chunk in enumerate(text_chunks)
        ]
