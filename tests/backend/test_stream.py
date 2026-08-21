"""
tests/backend/test_stream.py
──────────────────────────────
Integration tests for the WS /analysis/stream endpoint — the live
Reason/Act/Observe trail (M7 live dashboard + M8 WebSocket).

Covers three things:

  • **Auth parity.** The socket takes its token as a query parameter, because
    the browser WebSocket API cannot set headers. That transport change must
    not become an authentication loophole, so the rejection cases here mirror
    the ones ``get_current_user`` enforces on the REST endpoints.
  • **Protocol.** Ordered step frames, then exactly one terminal frame.
  • **Equivalence with POST /analysis/run.** Streaming must be a different view
    of the same pipeline, not a second implementation that can drift.

Runs against the configured DATABASE_URL (Postgres); each test creates its own
user so runs stay isolated.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.core.security import create_access_token, create_refresh_token
from backend.main import app
from backend.routers.analysis import WS_BAD_REQUEST, WS_UNAUTHORIZED

client = TestClient(app)

SAMPLE_CSV = (
    b"order_id,region,units,revenue,churned\n"
    b"1,East,3,45.0,0\n2,West,9,120.5,1\n3,East,5,75.0,0\n"
    b"4,North,4,60.0,1\n5,South,8,110.0,0\n6,West,2,30.0,1\n"
)

# The four agents the planner schedules for every task type.
EXPECTED_AGENTS = [
    "IngestionAgent",
    "MiningAgent",
    "VisualizationAgent",
    "RecommendationAgent",
]


def _unique_email() -> str:
    return f"stream-{uuid.uuid4().hex[:12]}@example.com"


def _register() -> tuple[str, str]:
    """Create a user; return (access_token, user_id)."""
    email = _unique_email()
    client.post(
        "/auth/register",
        json={"email": email, "full_name": "Stream Test", "password": "Passw0rd!"},
    )
    body = client.post("/auth/login", json={"email": email, "password": "Passw0rd!"}).json()
    return body["access_token"], body.get("user", {}).get("id", "")


def _upload(token: str) -> str:
    """Persist a dataset via the REST ingest endpoint; return its id."""
    res = client.post(
        "/analysis/ingest",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("stream.csv", SAMPLE_CSV, "text/csv")},
    )
    assert res.status_code == 200, res.text
    return res.json()["dataset_id"]


def _drain(ws) -> list[dict]:
    """Read frames until the terminal message, returning everything received."""
    frames: list[dict] = []
    while True:
        message = ws.receive_json()
        frames.append(message)
        if message["type"] in ("complete", "error"):
            return frames


@pytest.fixture(scope="module")
def token() -> str:
    return _register()[0]


@pytest.fixture(scope="module")
def dataset_id(token: str) -> str:
    return _upload(token)


class TestStreamAuthentication:
    """The query-param token must be held to the same standard as the header."""

    def test_missing_token_is_rejected(self) -> None:
        with client.websocket_connect("/analysis/stream") as ws:
            ws.receive()  # the close frame
        # Reaching here without an exception means the server closed it cleanly.

    def test_missing_token_uses_the_unauthorized_close_code(self) -> None:
        with client.websocket_connect("/analysis/stream") as ws:
            message = ws.receive()
        assert message["type"] == "websocket.close"
        assert message["code"] == WS_UNAUTHORIZED

    def test_garbage_token_is_rejected(self) -> None:
        with client.websocket_connect("/analysis/stream?token=not-a-jwt") as ws:
            message = ws.receive()
        assert message["code"] == WS_UNAUTHORIZED

    def test_refresh_token_is_rejected(self) -> None:
        """A refresh token must not be usable as an access token here either."""
        refresh = create_refresh_token("some-user-id")
        with client.websocket_connect(f"/analysis/stream?token={refresh}") as ws:
            message = ws.receive()
        assert message["code"] == WS_UNAUTHORIZED

    def test_wellformed_token_for_unknown_user_is_rejected(self) -> None:
        """Valid signature is not enough — the subject must resolve to a live user."""
        orphan = create_access_token(str(uuid.uuid4()))
        with client.websocket_connect(f"/analysis/stream?token={orphan}") as ws:
            message = ws.receive()
        assert message["code"] == WS_UNAUTHORIZED

    def test_valid_token_is_accepted(self, token: str) -> None:
        with client.websocket_connect(f"/analysis/stream?token={token}") as ws:
            ws.send_json({"goal": "Summarise this dataset", "expertise_level": "intermediate"})
            assert ws.receive_json()["type"] == "accepted"


class TestStreamRequestValidation:
    def test_empty_goal_is_rejected(self, token: str) -> None:
        with client.websocket_connect(f"/analysis/stream?token={token}") as ws:
            ws.send_json({"goal": "   "})
            message = ws.receive()
        assert message["code"] == WS_BAD_REQUEST

    def test_missing_goal_is_rejected(self, token: str) -> None:
        with client.websocket_connect(f"/analysis/stream?token={token}") as ws:
            ws.send_json({"expertise_level": "beginner"})
            message = ws.receive()
        assert message["code"] == WS_BAD_REQUEST

    def test_unknown_expertise_level_is_rejected(self, token: str) -> None:
        with client.websocket_connect(f"/analysis/stream?token={token}") as ws:
            ws.send_json({"goal": "Summarise this", "expertise_level": "wizard"})
            message = ws.receive()
        assert message["code"] == WS_BAD_REQUEST

    def test_non_json_start_message_is_rejected(self, token: str) -> None:
        with client.websocket_connect(f"/analysis/stream?token={token}") as ws:
            ws.send_text("this is not json")
            message = ws.receive()
        assert message["code"] == WS_BAD_REQUEST


@pytest.fixture(scope="module")
def frames(token: str, dataset_id: str) -> list[dict]:
    """One full streamed run, shared across the protocol assertions."""
    with client.websocket_connect(f"/analysis/stream?token={token}") as ws:
        ws.send_json(
            {
                "goal": "Find natural groupings and segments in this data",
                "expertise_level": "intermediate",
                "dataset_id": dataset_id,
            }
        )
        assert ws.receive_json()["type"] == "accepted"
        return _drain(ws)


@pytest.fixture(scope="module")
def streamed(token: str, dataset_id: str) -> dict:
    """The terminal frame of a streamed run, for the persistence assertions."""
    with client.websocket_connect(f"/analysis/stream?token={token}") as ws:
        ws.send_json(
            {
                "goal": "Detect unusual or anomalous records",
                "expertise_level": "intermediate",
                "dataset_id": dataset_id,
            }
        )
        ws.receive_json()
        return _drain(ws)[-1]


class TestStreamProtocol:
    def test_run_completes_successfully(self, frames: list[dict]) -> None:
        assert frames[-1]["type"] == "complete", frames[-1].get("detail")

    def test_one_step_frame_per_planned_agent(self, frames: list[dict]) -> None:
        steps = [f for f in frames if f["type"] == "step"]
        assert len(steps) == len(EXPECTED_AGENTS)

    def test_steps_arrive_in_pipeline_order(self, frames: list[dict]) -> None:
        steps = [f for f in frames if f["type"] == "step"]
        assert [s["agent_name"] for s in steps] == EXPECTED_AGENTS

    def test_step_indices_are_sequential(self, frames: list[dict]) -> None:
        steps = [f for f in frames if f["type"] == "step"]
        assert [s["index"] for s in steps] == list(range(len(steps)))

    def test_steps_carry_the_reason_act_observe_trail(self, frames: list[dict]) -> None:
        """FR-06 — the live view must show the explainability trail, not just names."""
        for step in (f for f in frames if f["type"] == "step"):
            assert step["reasoning"]
            assert step["observation"]
            assert step["action"]
            assert step["status"] in ("success", "error", "skipped")
            assert isinstance(step["latency_ms"], int)

    def test_step_frames_omit_the_heavy_output_payload(self, frames: list[dict]) -> None:
        """Kept lean deliberately; the full payload rides on the complete frame."""
        assert all("output" not in f for f in frames if f["type"] == "step")

    def test_exactly_one_terminal_frame(self, frames: list[dict]) -> None:
        terminal = [f for f in frames if f["type"] in ("complete", "error")]
        assert len(terminal) == 1
        assert frames[-1] is terminal[0]

    def test_complete_frame_carries_the_persisted_run_id(self, frames: list[dict]) -> None:
        assert frames[-1]["run_id"]

    def test_complete_frame_carries_the_full_result(self, frames: list[dict]) -> None:
        result = frames[-1]["result"]
        assert result["summary"]
        assert result["recommendations"]
        assert len(result["steps"]) == len(EXPECTED_AGENTS)

    def test_full_output_is_present_in_the_complete_frame(self, frames: list[dict]) -> None:
        """What the step frames omit must still be recoverable at the end."""
        mining = next(
            s for s in frames[-1]["result"]["steps"] if s["agent_name"] == "MiningAgent"
        )
        assert mining["output"]["computations_run"]


class TestStreamedRunIsPersistedAndEquivalent:
    """Streaming must be a view of the same pipeline, not a parallel one."""

    def test_run_is_saved_to_history(self, streamed: dict, token: str) -> None:
        res = client.get(
            f"/analysis/history/{streamed['run_id']}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        assert len(res.json()["steps"]) == len(EXPECTED_AGENTS)

    def test_streamed_run_is_exportable_like_any_other(self, streamed: dict, token: str) -> None:
        res = client.get(
            f"/analysis/history/{streamed['run_id']}/export/pdf",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        assert res.content.startswith(b"%PDF")

    def test_goal_conditioning_survives_the_streaming_path(
        self, token: str, dataset_id: str
    ) -> None:
        """FR-02 over the socket: two goals, same dataset, different computations."""
        def computations_for(goal: str) -> set[str]:
            with client.websocket_connect(f"/analysis/stream?token={token}") as ws:
                ws.send_json({"goal": goal, "dataset_id": dataset_id})
                ws.receive_json()
                final = _drain(ws)[-1]
            mining = next(
                s for s in final["result"]["steps"] if s["agent_name"] == "MiningAgent"
            )
            return set(mining["output"]["computations_run"])

        clustering = computations_for("Find natural clusters and segments in this data")
        anomaly = computations_for("Detect fraudulent and anomalous transactions")

        assert clustering and anomaly
        assert clustering != anomaly


class TestStreamOwnership:
    def test_another_users_dataset_is_not_streamed(self, dataset_id: str) -> None:
        """
        A second user naming the first user's dataset_id must not receive its
        data. `get_dataset` is user-scoped, so the id resolves to nothing.

        That used to mean the pipeline ran anyway with no source and returned a
        completed report built from goal-only retrieval. It is now refused
        outright (MissingDataSourceError → a bad-request close), which is the
        same security property reached more honestly: the caller cannot tell
        "this dataset is not yours" from "no such dataset", and gets no report
        either way.
        """
        other_token, _ = _register()
        with client.websocket_connect(f"/analysis/stream?token={other_token}") as ws:
            ws.send_json({"goal": "Summarise this dataset", "dataset_id": dataset_id})
            ws.receive_json()
            frames = _drain(ws)

        assert frames[-1]["type"] == "error"
        assert "dataset" in frames[-1]["detail"].lower()

        # The point of the test: none of the owner's data came back, in any
        # frame — not the column names, not the values.
        blob = json.dumps(frames)
        for leaked in ("order_id", "churned", "120.5", "South"):
            assert leaked not in blob, f"{leaked!r} leaked to a non-owner"
