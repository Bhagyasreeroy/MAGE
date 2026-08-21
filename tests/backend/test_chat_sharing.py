"""
tests/backend/test_chat_sharing.py
──────────────────────────────────────
Integration tests for conversation threading (root_run_id) and public
sharing: /analysis/history/{run_id}/thread, /analysis/history/{run_id}/share,
and the public /analysis/shared/{root_run_id}.

Runs against the configured DATABASE_URL (Postgres); each test creates its
own user and deletes it (cascade) when done.
"""

from __future__ import annotations

import os
import sys
import uuid

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app

client = TestClient(app)


def _unique_email() -> str:
    return f"share-test-{uuid.uuid4().hex[:12]}@example.com"


def _auth_headers() -> dict[str, str]:
    email = _unique_email()
    client.post("/auth/register", json={"email": email, "full_name": "Share Test", "password": "Passw0rd!"})
    token = client.post("/auth/login", json={"email": email, "password": "Passw0rd!"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _run(headers: dict[str, str], goal: str, root_run_id: str | None = None) -> dict:
    data = {"goal": goal, "expertise_level": "intermediate"}
    if root_run_id:
        data["root_run_id"] = root_run_id
    return client.post("/analysis/run", headers=headers, data=data).json()


class TestConversationThreading:
    def test_a_fresh_run_is_its_own_root(self) -> None:
        headers = _auth_headers()
        run = _run(headers, "Summarise this dataset")
        thread = client.get(f"/analysis/history/{run['run_id']}/thread", headers=headers).json()

        assert len(thread) == 1
        assert thread[0]["run_id"] == run["run_id"]
        client.delete("/auth/me", headers=headers)

    def test_a_followup_joins_the_original_thread(self) -> None:
        headers = _auth_headers()
        first = _run(headers, "Summarise this dataset")
        second = _run(headers, "What about outliers?", root_run_id=first["run_id"])

        thread = client.get(f"/analysis/history/{first['run_id']}/thread", headers=headers).json()
        assert [r["run_id"] for r in thread] == [first["run_id"], second["run_id"]]

        # Fetching the thread from the follow-up's own id resolves to the
        # same conversation — not just from the root's id.
        thread_via_followup = client.get(
            f"/analysis/history/{second['run_id']}/thread", headers=headers
        ).json()
        assert [r["run_id"] for r in thread_via_followup] == [first["run_id"], second["run_id"]]
        client.delete("/auth/me", headers=headers)

    def test_a_root_run_id_belonging_to_another_user_is_ignored(self) -> None:
        headers_a = _auth_headers()
        headers_b = _auth_headers()
        run_a = _run(headers_a, "User A's private analysis")

        # User B tries to attach a follow-up to A's run — should silently
        # start B's own fresh conversation instead of erroring or merging.
        run_b = _run(headers_b, "Trying to piggyback", root_run_id=run_a["run_id"])
        thread_b = client.get(f"/analysis/history/{run_b['run_id']}/thread", headers=headers_b).json()

        assert len(thread_b) == 1
        assert thread_b[0]["run_id"] == run_b["run_id"]

        thread_a = client.get(f"/analysis/history/{run_a['run_id']}/thread", headers=headers_a).json()
        assert len(thread_a) == 1

        client.delete("/auth/me", headers=headers_a)
        client.delete("/auth/me", headers=headers_b)

    def test_thread_404s_for_another_user(self) -> None:
        headers_a = _auth_headers()
        headers_b = _auth_headers()
        run_a = _run(headers_a, "Private thread")

        res = client.get(f"/analysis/history/{run_a['run_id']}/thread", headers=headers_b)
        assert res.status_code == 404

        client.delete("/auth/me", headers=headers_a)
        client.delete("/auth/me", headers=headers_b)


class TestShareToggle:
    def test_new_conversation_starts_unshared(self) -> None:
        headers = _auth_headers()
        run = _run(headers, "Fresh analysis")
        status_res = client.get(f"/analysis/history/{run['run_id']}/share", headers=headers).json()
        assert status_res == {"is_shared": False, "share_id": run["run_id"]}
        client.delete("/auth/me", headers=headers)

    def test_sharing_and_unsharing_toggles_status(self) -> None:
        headers = _auth_headers()
        run = _run(headers, "Analysis to share")

        shared = client.post(f"/analysis/history/{run['run_id']}/share", headers=headers).json()
        assert shared == {"is_shared": True, "share_id": run["run_id"]}
        assert client.get(f"/analysis/history/{run['run_id']}/share", headers=headers).json()["is_shared"] is True

        unshared = client.delete(f"/analysis/history/{run['run_id']}/share", headers=headers).json()
        assert unshared == {"is_shared": False, "share_id": run["run_id"]}
        assert client.get(f"/analysis/history/{run['run_id']}/share", headers=headers).json()["is_shared"] is False

        client.delete("/auth/me", headers=headers)

    def test_sharing_from_a_followup_shares_the_whole_thread(self) -> None:
        headers = _auth_headers()
        first = _run(headers, "Original question")
        second = _run(headers, "Follow-up question", root_run_id=first["run_id"])

        # Share via the follow-up's own id.
        shared = client.post(f"/analysis/history/{second['run_id']}/share", headers=headers).json()
        assert shared["share_id"] == first["run_id"]

        status_via_root = client.get(f"/analysis/history/{first['run_id']}/share", headers=headers).json()
        assert status_via_root["is_shared"] is True

        client.delete("/auth/me", headers=headers)

    def test_share_endpoints_require_auth(self) -> None:
        run_id = str(uuid.uuid4())
        assert client.get(f"/analysis/history/{run_id}/share").status_code == 401
        assert client.post(f"/analysis/history/{run_id}/share").status_code == 401
        assert client.delete(f"/analysis/history/{run_id}/share").status_code == 401

    def test_share_endpoints_404_for_another_users_run(self) -> None:
        headers_a = _auth_headers()
        headers_b = _auth_headers()
        run_a = _run(headers_a, "A's analysis")

        assert client.get(f"/analysis/history/{run_a['run_id']}/share", headers=headers_b).status_code == 404
        assert client.post(f"/analysis/history/{run_a['run_id']}/share", headers=headers_b).status_code == 404
        assert client.delete(f"/analysis/history/{run_a['run_id']}/share", headers=headers_b).status_code == 404

        client.delete("/auth/me", headers=headers_a)
        client.delete("/auth/me", headers=headers_b)


class TestPublicSharedView:
    def test_unshared_run_returns_404(self) -> None:
        headers = _auth_headers()
        run = _run(headers, "Never shared")

        res = client.get(f"/analysis/shared/{run['run_id']}")
        assert res.status_code == 404

        client.delete("/auth/me", headers=headers)

    def test_nonexistent_run_returns_404(self) -> None:
        res = client.get(f"/analysis/shared/{uuid.uuid4()}")
        assert res.status_code == 404

    def test_shared_run_is_viewable_with_no_auth(self) -> None:
        headers = _auth_headers()
        run = _run(headers, "Public analysis")
        client.post(f"/analysis/history/{run['run_id']}/share", headers=headers)

        res = client.get(f"/analysis/shared/{run['run_id']}")
        assert res.status_code == 200
        body = res.json()
        assert len(body) == 1
        assert body[0]["goal"] == "Public analysis"

        client.delete("/auth/me", headers=headers)

    def test_shared_view_includes_the_whole_thread_in_order(self) -> None:
        headers = _auth_headers()
        first = _run(headers, "First question")
        second = _run(headers, "Second question", root_run_id=first["run_id"])
        client.post(f"/analysis/history/{first['run_id']}/share", headers=headers)

        res = client.get(f"/analysis/shared/{first['run_id']}")
        assert res.status_code == 200
        body = res.json()
        assert [r["goal"] for r in body] == ["First question", "Second question"]
        assert [r["run_id"] for r in body] == [first["run_id"], second["run_id"]]

        client.delete("/auth/me", headers=headers)

    def test_shared_view_omits_dataset_id(self) -> None:
        headers = _auth_headers()
        sample_csv = b"a,b\n1,2\n3,4\n"
        run = client.post(
            "/analysis/run",
            headers=headers,
            data={"goal": "Analysis with a dataset", "expertise_level": "intermediate"},
            files={"file": ("data.csv", sample_csv, "text/csv")},
        ).json()
        assert run["dataset_id"]
        client.post(f"/analysis/history/{run['run_id']}/share", headers=headers)

        body = client.get(f"/analysis/shared/{run['run_id']}").json()
        assert body[0]["dataset_id"] is None

        client.delete("/auth/me", headers=headers)

    def test_a_non_root_run_id_is_not_directly_viewable(self) -> None:
        """Even once shared, the share id must be the thread's root — a
        follow-up's own id is not itself a valid public URL."""
        headers = _auth_headers()
        first = _run(headers, "Root question")
        second = _run(headers, "Follow-up question", root_run_id=first["run_id"])
        client.post(f"/analysis/history/{first['run_id']}/share", headers=headers)

        assert client.get(f"/analysis/shared/{first['run_id']}").status_code == 200
        assert client.get(f"/analysis/shared/{second['run_id']}").status_code == 404

        client.delete("/auth/me", headers=headers)

    def test_unsharing_makes_the_link_404_again(self) -> None:
        headers = _auth_headers()
        run = _run(headers, "Toggle test")
        client.post(f"/analysis/history/{run['run_id']}/share", headers=headers)
        assert client.get(f"/analysis/shared/{run['run_id']}").status_code == 200

        client.delete(f"/analysis/history/{run['run_id']}/share", headers=headers)
        assert client.get(f"/analysis/shared/{run['run_id']}").status_code == 404

        client.delete("/auth/me", headers=headers)
