"""
tests/backend/test_spreadsheet_editing.py
──────────────────────────────────────────
Integration tests for what the spreadsheet grid actually sends.

Two things are pinned here that the unit tests in
tests/data_pipeline/test_cell_edit_types.py cannot reach.

**The status code.** Editing a numeric cell used to return a 500: pandas raised
TypeError on the string-into-int64 assignment, and TypeError is not the
ProcessingError the router maps to a 400, so it escaped as a server fault. The
grid showed "failed to fetch, error 500" with nothing to act on. A 500 for a
value the user typed is the bug, and only an HTTP test can tell a 400 from it.

**The pagination contract.** `row_index` is absolute — an index into the whole
dataset, not into the page on screen. The grid used to send a page-relative
index with the current offset added, so an edit made on page 1 and saved from
page 2 landed on a different row. The backend was always right about this; the
tests below say so out loud, because it is the invariant the client has to hold
up its end of.

Requires PostgreSQL.
"""

from __future__ import annotations

import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app

client = TestClient(app)

# Ingestion types this as int64 / float64 / str / bool / str — CSV dates load
# as text, so there is deliberately no datetime column here. The datetime
# branch of the coercion is covered in
# tests/data_pipeline/test_cell_edit_types.py, against a frame that actually
# has one; asserting datetime behaviour through this path would only be
# asserting that ingestion does not infer dates.
TYPED_CSV = (
    b"units,price,region,active,day\n"
    b"1,1.5,East,true,2024-01-01\n"
    b"2,2.5,West,false,2024-01-02\n"
    b"3,3.5,East,true,2024-01-03\n"
)

# Long enough to page through at the grid's 25 rows per page.
PAGED_CSV = b"idx,label\n" + b"".join(f"{i},row-{i}\n".encode() for i in range(60))


def _auth() -> dict[str, str]:
    email = f"sheet-{uuid.uuid4().hex[:12]}@example.com"
    client.post("/auth/register", json={"email": email, "full_name": "Sheet", "password": "Passw0rd!"})
    token = client.post("/auth/login", json={"email": email, "password": "Passw0rd!"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _ingest(headers, csv: bytes) -> str:
    res = client.post("/analysis/ingest", headers=headers, files={"file": ("d.csv", csv, "text/csv")})
    assert res.status_code == 200, res.text
    return res.json()["dataset_id"]


def _edit(headers, dataset_id: str, edits: list[dict]):
    return client.post(
        f"/analysis/datasets/{dataset_id}/transform",
        headers=headers,
        json={"ops": [{"type": "edit_cells", "edits": edits}]},
    )


@pytest.fixture(scope="module")
def headers() -> dict[str, str]:
    return _auth()


class TestEditingATypedCellSucceeds:
    @pytest.mark.parametrize(
        ("column", "value"),
        [("units", "42"), ("price", "9.25"), ("region", "North"), ("active", "false"), ("day", "2025-05-05")],
    )
    def test_every_column_type_saves(self, headers, column, value) -> None:
        dataset_id = _ingest(headers, TYPED_CSV)
        res = _edit(headers, dataset_id, [{"row_index": 0, "column": column, "value": value}])
        assert res.status_code == 201, res.text

    def test_the_new_value_is_readable_back(self, headers) -> None:
        dataset_id = _ingest(headers, TYPED_CSV)
        new_id = _edit(headers, dataset_id, [{"row_index": 0, "column": "units", "value": "42"}]).json()["id"]
        preview = client.get(f"/analysis/datasets/{new_id}/preview", headers=headers).json()
        assert preview["rows"][0][0] == 42

    def test_clearing_a_numeric_cell_saves(self, headers) -> None:
        dataset_id = _ingest(headers, TYPED_CSV)
        new_id = _edit(headers, dataset_id, [{"row_index": 0, "column": "units", "value": ""}]).json()["id"]
        preview = client.get(f"/analysis/datasets/{new_id}/preview", headers=headers).json()
        assert preview["rows"][0][0] is None


class TestAnImpossibleValueIsA400:
    """Not a 500. The distinction is the whole point — one is something the
    user can fix, the other sends whoever reads the logs somewhere else."""

    @pytest.mark.parametrize(
        ("column", "value"),
        [("units", "abc"), ("price", "not a number"), ("active", "maybe")],
    )
    def test_a_bad_value_is_rejected_not_crashed(self, headers, column, value) -> None:
        dataset_id = _ingest(headers, TYPED_CSV)
        res = _edit(headers, dataset_id, [{"row_index": 0, "column": column, "value": value}])
        assert res.status_code == 400, res.text

    def test_the_error_says_which_column(self, headers) -> None:
        dataset_id = _ingest(headers, TYPED_CSV)
        res = _edit(headers, dataset_id, [{"row_index": 0, "column": "units", "value": "abc"}])
        assert "units" in res.json()["detail"]

    def test_a_rejected_edit_creates_no_version(self, headers) -> None:
        dataset_id = _ingest(headers, TYPED_CSV)
        before = client.get(f"/analysis/datasets/{dataset_id}", headers=headers).json()["root_id"]
        _edit(headers, dataset_id, [{"row_index": 0, "column": "units", "value": "abc"}])
        versions = client.get(f"/analysis/datasets/{before}/versions", headers=headers).json()
        assert len(versions) == 1


class TestRowIndexIsAbsolute:
    def test_an_index_beyond_the_first_page_addresses_that_row(self, headers) -> None:
        """Row 30 is on page 2 of the grid. It is still row 30 here."""
        dataset_id = _ingest(headers, PAGED_CSV)
        new_id = _edit(headers, dataset_id, [{"row_index": 30, "column": "label", "value": "EDITED"}]).json()["id"]
        page2 = client.get(
            f"/analysis/datasets/{new_id}/preview?offset=25&limit=25", headers=headers
        ).json()
        assert page2["rows"][5][1] == "EDITED"

    def test_the_first_page_is_untouched_by_it(self, headers) -> None:
        dataset_id = _ingest(headers, PAGED_CSV)
        new_id = _edit(headers, dataset_id, [{"row_index": 30, "column": "label", "value": "EDITED"}]).json()["id"]
        page1 = client.get(f"/analysis/datasets/{new_id}/preview?offset=0&limit=25", headers=headers).json()
        assert all(row[1] != "EDITED" for row in page1["rows"])

    def test_edits_from_several_pages_save_together(self, headers) -> None:
        """One save carries every pending edit, each against its own row —
        which is only true because the indices are absolute."""
        dataset_id = _ingest(headers, PAGED_CSV)
        new_id = _edit(
            headers,
            dataset_id,
            [
                {"row_index": 2, "column": "label", "value": "FIRST"},
                {"row_index": 30, "column": "label", "value": "SECOND"},
                {"row_index": 57, "column": "label", "value": "THIRD"},
            ],
        ).json()["id"]

        def cell(row: int) -> str:
            page = client.get(
                f"/analysis/datasets/{new_id}/preview?offset={row}&limit=1", headers=headers
            ).json()
            return page["rows"][0][1]

        assert (cell(2), cell(30), cell(57)) == ("FIRST", "SECOND", "THIRD")


class TestPreviewPagesActuallyDiffer:
    """The grid showed page 1's rows no matter which page was selected. That
    was a client bug, but it is worth having the server side stated: each page
    is a genuinely different slice."""

    def test_consecutive_pages_return_different_rows(self, headers) -> None:
        dataset_id = _ingest(headers, PAGED_CSV)
        page1 = client.get(f"/analysis/datasets/{dataset_id}/preview?offset=0&limit=25", headers=headers).json()
        page2 = client.get(f"/analysis/datasets/{dataset_id}/preview?offset=25&limit=25", headers=headers).json()
        assert page1["rows"] != page2["rows"]
        assert page1["rows"][0][0] == 0
        assert page2["rows"][0][0] == 25

    def test_every_page_reports_the_same_total(self, headers) -> None:
        dataset_id = _ingest(headers, PAGED_CSV)
        for offset in (0, 25, 50):
            page = client.get(
                f"/analysis/datasets/{dataset_id}/preview?offset={offset}&limit=25", headers=headers
            ).json()
            assert page["total_rows"] == 60
            assert page["offset"] == offset
