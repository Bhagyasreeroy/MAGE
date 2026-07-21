"""Tests for rag/knowledge_loader.py."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rag.knowledge_loader import KnowledgeBaseLoader


class TestKnowledgeBaseLoader:
    def test_load_all_returns_chunks_from_seed_kb(self) -> None:
        chunks = KnowledgeBaseLoader().load_all()
        assert len(chunks) > 0
        assert all({"text", "source", "metadata"} <= chunk.keys() for chunk in chunks)

    def test_chunks_carry_title_metadata(self) -> None:
        chunks = KnowledgeBaseLoader().load_all()
        titles = {chunk["metadata"]["title"] for chunk in chunks}
        assert "Handling Missing Values in Exploratory Data Analysis" in titles

    def test_missing_kb_dir_returns_empty_list(self, tmp_path) -> None:
        loader = KnowledgeBaseLoader(kb_dir=tmp_path / "does_not_exist")
        assert loader.load_all() == []

    def test_load_file_parses_frontmatter(self, tmp_path) -> None:
        doc = tmp_path / "sample.md"
        doc.write_text(
            "---\n"
            "title: Sample Doc\n"
            "doc_type: methodology\n"
            "section: testing\n"
            "---\n"
            "This is the body of the sample document, used to verify chunking.\n",
            encoding="utf-8",
        )
        loader = KnowledgeBaseLoader(kb_dir=tmp_path)
        chunks = loader.load_file(doc)

        assert len(chunks) == 1
        assert chunks[0]["metadata"]["title"] == "Sample Doc"
        assert chunks[0]["metadata"]["doc_type"] == "methodology"
        assert chunks[0]["source"] == "knowledge_base/sample.md"
        assert "sample document" in chunks[0]["text"]

    def test_load_file_without_frontmatter_falls_back_to_filename(self, tmp_path) -> None:
        doc = tmp_path / "no_frontmatter.md"
        doc.write_text("Just plain content, no frontmatter block here.\n", encoding="utf-8")
        loader = KnowledgeBaseLoader(kb_dir=tmp_path)
        chunks = loader.load_file(doc)

        assert len(chunks) == 1
        assert chunks[0]["metadata"]["title"] == "No Frontmatter"

    def test_long_document_splits_into_multiple_chunks_without_mid_word_breaks(self, tmp_path) -> None:
        long_paragraph = " ".join(f"Sentence number {i} about outlier detection methods." for i in range(60))
        doc = tmp_path / "long.md"
        doc.write_text(long_paragraph, encoding="utf-8")

        loader = KnowledgeBaseLoader(kb_dir=tmp_path, chunk_size=200, chunk_overlap=40)
        chunks = loader.load_file(doc)

        assert len(chunks) > 1
        for chunk in chunks:
            text = chunk["text"]
            assert text == text.strip()
            # No chunk should start with a lowercase continuation of a
            # word cut off mid-token by the overlap window.
            assert text[0] != " "

    def test_missing_file_returns_empty_list(self, tmp_path) -> None:
        loader = KnowledgeBaseLoader(kb_dir=tmp_path)
        assert loader.load_file(tmp_path / "nope.md") == []
