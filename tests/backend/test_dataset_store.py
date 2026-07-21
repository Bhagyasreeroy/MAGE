"""Tests for backend/core/dataset_store.py."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.core.dataset_store import MAX_ENTRIES, DatasetSessionStore


class TestDatasetSessionStore:
    def test_put_then_get_roundtrips_content(self) -> None:
        store = DatasetSessionStore()
        dataset_id = store.put("sales.csv", b"a,b\n1,2\n")

        stored = store.get(dataset_id)
        assert stored is not None
        assert stored.filename == "sales.csv"
        assert stored.file.read() == b"a,b\n1,2\n"

    def test_unknown_id_returns_none(self) -> None:
        store = DatasetSessionStore()
        assert store.get("does-not-exist") is None

    def test_each_put_gets_a_distinct_id(self) -> None:
        store = DatasetSessionStore()
        id1 = store.put("a.csv", b"1")
        id2 = store.put("b.csv", b"2")
        assert id1 != id2

    def test_get_returns_independently_readable_stream_each_time(self) -> None:
        store = DatasetSessionStore()
        dataset_id = store.put("a.csv", b"hello")

        first = store.get(dataset_id)
        second = store.get(dataset_id)
        assert first.file.read() == b"hello"
        assert second.file.read() == b"hello", "a second get() must not be exhausted by the first read"

    def test_evicts_oldest_entry_beyond_max_entries(self) -> None:
        store = DatasetSessionStore()
        ids = [store.put(f"{i}.csv", b"x") for i in range(MAX_ENTRIES + 5)]

        assert store.get(ids[0]) is None, "oldest entries should have been evicted"
        assert store.get(ids[-1]) is not None, "most recent entry should still be present"
