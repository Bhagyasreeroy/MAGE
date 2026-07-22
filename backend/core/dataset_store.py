"""
core/dataset_store.py
──────────────────────
In-memory, session-scoped store for uploaded datasets.

No database table exists yet for datasets/analysis runs (see AGENTS.md /
architecture notes), so this is a minimal process-local stand-in: it lets
a follow-up request reference the same uploaded file (by id) instead of
re-uploading it, which is what makes the results-page chat "interactive"
rather than goal-only. Not persisted across backend restarts, not shared
across processes — a real dataset table is the natural next step.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass

#: Cap on concurrently held datasets — evicts oldest on overflow so a long
#: dev session can't grow this dict unbounded.
MAX_ENTRIES = 50


@dataclass
class StoredFile:
    """Duck-types the parts of FastAPI's UploadFile that the ingestion
    engine actually uses (.filename, .file), so a stored dataset can be
    passed straight back through the same ingestion code path."""

    filename: str
    file: io.BytesIO


class DatasetSessionStore:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, bytes]] = {}
        self._order: list[str] = []

    def put(self, filename: str, content: bytes) -> str:
        """Store raw file bytes and return a new id to reference them by."""
        dataset_id = str(uuid.uuid4())
        self._entries[dataset_id] = (filename, content)
        self._order.append(dataset_id)
        while len(self._order) > MAX_ENTRIES:
            oldest = self._order.pop(0)
            self._entries.pop(oldest, None)
        return dataset_id

    def get(self, dataset_id: str) -> StoredFile | None:
        """Return a fresh, independently-readable file-like object, or None."""
        entry = self._entries.get(dataset_id)
        if entry is None:
            return None
        filename, content = entry
        return StoredFile(filename=filename, file=io.BytesIO(content))


#: Process-wide singleton — mirrors the pattern used elsewhere in backend/services.
dataset_store = DatasetSessionStore()
