"""
tests/backend/test_conversation_wiring.py
──────────────────────────────────────────
End-to-end tests that chat history reaches the pipeline over HTTP.

The agent-side behaviour is covered in tests/agents/test_followup_chat.py. This
checks the wiring around it: the field is accepted, it changes the answer, and
the two ways it can go wrong are reported as the client's mistake rather than
ours.

The second of those is the reason this file exists separately. `conversation`
is parsed inside the handler, because the endpoint is multipart and cannot take
a body model — and an uncaught ValidationError there is a 500. A validation
failure being reported as a server fault is the kind of bug that sends the next
person reading logs in entirely the wrong direction.

Written test-first. Requires PostgreSQL.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import app

client = TestClient(app)

SAMPLE_CSV = (
    b"units,unit_price,revenue\n"
    + b"".join(
        f"{i % 9 + 1},{10 + i % 7},{(i % 9 + 1) * (10 + i % 7)}\n".encode() for i in range(60)
    )
)

OPENING_GOAL = "Analyse this dataset and tell me what stands out"
FOLLOW_UP = "What is the right way to treat absent readings before modelling?"


def _auth() -> dict[str, str]:
    email = f"convo-{uuid.uuid4().hex[:12]}@example.com"
    client.post(
        "/auth/register",
        json={"email": email, "full_name": "Conversation Test", "password": "Passw0rd!"},
    )
    token = client.post(
        "/auth/login", json={"email": email, "password": "Passw0rd!"}
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _post(headers, goal, dataset_id=None, conversation=None, raw_conversation=None):
    data = {"goal": goal, "expertise_level": "intermediate"}
    files = None
    if dataset_id:
        data["dataset_id"] = dataset_id
    else:
        files = {"file": ("d.csv", SAMPLE_CSV, "text/csv")}
    if conversation is not None:
        data["conversation"] = json.dumps(conversation)
    if raw_conversation is not None:
        data["conversation"] = raw_conversation
    return client.post("/analysis/run", headers=headers, data=data, files=files)


@pytest.fixture(scope="module")
def headers() -> dict[str, str]:
    return _auth()


@pytest.fixture(scope="module")
def opening(headers) -> dict:
    """The first run of a conversation — the one a follow-up must not repeat."""
    response = _post(headers, OPENING_GOAL)
    assert response.status_code == 200, response.text
    return response.json()


def _history(opening: dict) -> list[dict]:
    return [
        {"role": "user", "content": opening["goal"], "sources": []},
        {
            "role": "assistant",
            "content": opening["summary"] or "…",
            "sources": opening["rag_sources"],
        },
    ]


class TestTheFieldIsAcceptedAndUsed:
    def test_a_run_without_the_field_still_works(self, opening: dict) -> None:
        """Absent means "first turn", not "malformed" — every existing client
        sends nothing."""
        assert opening["recommendations"]

    def test_a_follow_up_does_not_repeat_the_opening_answer(
        self, headers, opening: dict
    ) -> None:
        response = _post(
            headers,
            FOLLOW_UP,
            dataset_id=opening["dataset_id"],
            conversation=_history(opening),
        )
        assert response.status_code == 200, response.text
        assert response.json()["rag_sources"] != opening["rag_sources"]

    def test_a_short_elliptical_follow_up_is_accepted(self, headers, opening: dict) -> None:
        """"Why?" is under the first-turn goal floor. With a conversation to
        attach it to it is a complete question, and refusing it would refuse
        the shape of question this whole field exists to support."""
        response = _post(
            headers, "Why?", dataset_id=opening["dataset_id"], conversation=_history(opening)
        )
        assert response.status_code == 200, response.text
        assert response.json()["recommendations"]

    def test_the_stored_goal_is_what_the_user_typed(self, headers, opening: dict) -> None:
        """Resolution is for machines. History should record the question they
        asked, not our expansion of it."""
        response = _post(
            headers, "Why?", dataset_id=opening["dataset_id"], conversation=_history(opening)
        )
        assert response.json()["goal"] == "Why?"


class TestBadInputIsTheClientsFault:
    def test_a_short_goal_on_a_first_turn_is_rejected_as_a_422(self, headers) -> None:
        response = _post(headers, "Why?")
        assert response.status_code == 422, response.text

    def test_malformed_json_is_a_422(self, headers, opening: dict) -> None:
        response = _post(
            headers,
            FOLLOW_UP,
            dataset_id=opening["dataset_id"],
            raw_conversation="{not json",
        )
        assert response.status_code == 422, response.text
        assert "conversation" in response.json()["detail"]

    def test_a_json_object_instead_of_an_array_is_a_422(self, headers, opening: dict) -> None:
        response = _post(
            headers,
            FOLLOW_UP,
            dataset_id=opening["dataset_id"],
            raw_conversation='{"role": "user", "content": "hi"}',
        )
        assert response.status_code == 422, response.text

    def test_an_unknown_role_is_a_422(self, headers, opening: dict) -> None:
        response = _post(
            headers,
            FOLLOW_UP,
            dataset_id=opening["dataset_id"],
            conversation=[{"role": "wizard", "content": "abracadabra"}],
        )
        assert response.status_code == 422, response.text

    def test_an_empty_string_means_no_history(self, headers, opening: dict) -> None:
        """A client that always sets the field, and has nothing to put in it,
        is on the first turn — not in error."""
        response = _post(
            headers, FOLLOW_UP, dataset_id=opening["dataset_id"], raw_conversation=""
        )
        assert response.status_code == 200, response.text
