"""
tests/agents/test_llm_retry.py
───────────────────────────────
NFR-03 — retry with exponential backoff on the LLM call.

This requirement used to be recorded as "moot until an LLM exists". An LLM now
exists (Gemini, 17 Aug), which moved it from *not applicable* to *unmet*:
``GeminiClient.generate`` raised ``LLMError`` on the first failure of any kind.

That is the wrong default for the failure modes a hosted model actually has.
A 503 or a read timeout is usually a moment of capacity pressure, and the
project already hit exactly this — ``gemini-2.5-flash`` was pinned because the
``-latest`` alias was *observed returning 503s*. Retrying those costs a second
and usually succeeds.

The opposite half matters just as much: a 400 (malformed prompt) or a 403 (bad
key) will fail identically three times. Retrying them wastes the user's wait and
tells them nothing new, so they must fail fast.

``httpx.post`` is called at module scope precisely so it can be faked here —
the same pattern ``test_llm_client.py`` and the REST ingestion tests already use.
"""

from __future__ import annotations

import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents import llm_client as lc
from agents.llm_client import GeminiClient, LLMError


class FakeResponse:
    def __init__(self, status_code: int, text: str = "", payload: dict | None = None) -> None:
        self.status_code = status_code
        self.text = text or "{}"
        self._payload = payload if payload is not None else _ok_payload("fine")

    def json(self) -> dict:
        return self._payload


def _ok_payload(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


@pytest.fixture
def client() -> GeminiClient:
    return GeminiClient(api_key="test-key")


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Record the backoff schedule instead of living through it."""
    slept: list[float] = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: slept.append(s))
    return slept


def _responder(monkeypatch, outcomes):
    """Serve `outcomes` in order; each is a FakeResponse or an exception."""
    calls: list[int] = []

    def fake_post(*args, **kwargs):
        i = len(calls)
        calls.append(i)
        outcome = outcomes[min(i, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


# ── Transient failures are retried ───────────────────────────────────────────


class TestTransientFailuresRetry:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_a_retryable_status_is_retried_and_can_succeed(self, client, monkeypatch, status) -> None:
        calls = _responder(monkeypatch, [FakeResponse(status, "busy"), FakeResponse(200)])
        assert client.generate("hello") == "fine"
        assert len(calls) == 2

    def test_a_timeout_is_retried(self, client, monkeypatch) -> None:
        calls = _responder(monkeypatch, [httpx.ReadTimeout("too slow"), FakeResponse(200)])
        assert client.generate("hello") == "fine"
        assert len(calls) == 2

    def test_a_connection_error_is_retried(self, client, monkeypatch) -> None:
        calls = _responder(monkeypatch, [httpx.ConnectError("refused"), FakeResponse(200)])
        assert client.generate("hello") == "fine"
        assert len(calls) == 2

    def test_it_gives_up_after_the_attempt_cap(self, client, monkeypatch) -> None:
        calls = _responder(monkeypatch, [FakeResponse(503, "still busy")])
        with pytest.raises(LLMError, match="503"):
            client.generate("hello")
        assert len(calls) == lc.MAX_ATTEMPTS

    def test_the_final_error_still_names_the_cause(self, client, monkeypatch) -> None:
        """A caller that has waited through three attempts deserves the reason."""
        _responder(monkeypatch, [httpx.ReadTimeout("too slow")])
        with pytest.raises(LLMError) as err:
            client.generate("hello")
        assert "too slow" in str(err.value) or "timeout" in str(err.value).lower()


# ── Permanent failures fail fast ─────────────────────────────────────────────


class TestPermanentFailuresDoNotRetry:
    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_a_client_error_is_not_retried(self, client, monkeypatch, status) -> None:
        """These fail identically three times — retrying only triples the wait."""
        calls = _responder(monkeypatch, [FakeResponse(status, "bad request")])
        with pytest.raises(LLMError, match=str(status)):
            client.generate("hello")
        assert len(calls) == 1

    def test_a_malformed_success_body_is_not_retried(self, client, monkeypatch) -> None:
        calls = _responder(monkeypatch, [FakeResponse(200, payload={"nope": True})])
        with pytest.raises(LLMError, match="Unexpected"):
            client.generate("hello")
        assert len(calls) == 1

    def test_an_unconfigured_client_does_not_call_out_at_all(self, monkeypatch) -> None:
        calls = _responder(monkeypatch, [FakeResponse(200)])
        with pytest.raises(LLMError, match="GEMINI_API_KEY"):
            GeminiClient(api_key="").generate("hello")
        assert len(calls) == 0


# ── The backoff itself ───────────────────────────────────────────────────────


class TestBackoffSchedule:
    def test_waits_grow_between_attempts(self, client, monkeypatch, _no_real_sleeping) -> None:
        _responder(monkeypatch, [FakeResponse(503)])
        with pytest.raises(LLMError):
            client.generate("hello")
        waits = _no_real_sleeping
        assert len(waits) == lc.MAX_ATTEMPTS - 1, "should sleep between attempts, not after the last"
        assert waits == sorted(waits), f"backoff must not shrink: {waits}"
        assert waits[-1] > waits[0]

    def test_a_success_never_sleeps(self, client, monkeypatch, _no_real_sleeping) -> None:
        _responder(monkeypatch, [FakeResponse(200)])
        client.generate("hello")
        assert _no_real_sleeping == []

    def test_total_wait_is_bounded(self, client, monkeypatch, _no_real_sleeping) -> None:
        """A user is waiting on this — the ceiling has to be defensible."""
        _responder(monkeypatch, [FakeResponse(503)])
        with pytest.raises(LLMError):
            client.generate("hello")
        assert sum(_no_real_sleeping) <= lc.MAX_TOTAL_BACKOFF_SECONDS
