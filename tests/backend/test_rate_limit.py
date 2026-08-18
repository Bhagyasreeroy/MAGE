"""
tests/backend/test_rate_limit.py
─────────────────────────────────
M8 — application-level rate limiting.

``backend/routers/auth.py`` used to concede in a docstring that rate limiting
was "recommended via a reverse proxy in production", which is a way of saying
the application does not do it. These tests make it do it.

Scope is deliberately narrow — the endpoints where an unauthenticated or cheap
request costs something real:

  * ``/auth/login`` and ``/auth/register`` — credential stuffing
  * ``/analysis/run`` — CPU, and the whole pipeline
  * ``/analysis/explain`` and ``.../query/nl`` — they spend Gemini quota

Reads stay unlimited: throttling a user for paging through their own history
would be a worse bug than the one being fixed.

**Keying matters more than the number.** Limiting purely by IP would let one
user behind a shared NAT — a university lab, which is exactly where this will be
demoed — lock out everyone around them. The limiter keys on the authenticated
user where there is one and falls back to IP only for anonymous requests.

The limit is configurable so the other ~720 tests, which hammer these endpoints,
do not become timing-dependent. That configurability is itself tested: a
misconfigured "unlimited" default would make every other test here vacuous.
"""

from __future__ import annotations

import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.core.config import settings
from backend.core.rate_limit import limiter, rate_limit_key
from backend.main import app

client = TestClient(app)


def _register(email: str | None = None) -> tuple[str, dict]:
    email = email or f"rl-{uuid.uuid4().hex[:8]}@example.com"
    client.post("/auth/register", json={"email": email, "password": "Str0ng!Pass123", "full_name": "RL"})
    res = client.post("/auth/login", json={"email": email, "password": "Str0ng!Pass123"})
    token = res.json().get("access_token", "")
    return email, {"Authorization": f"Bearer {token}"}


@pytest.fixture
def tight_limit(monkeypatch):
    """Squeeze the limit down for one test, then let it go back."""
    monkeypatch.setattr(settings, "rate_limit_auth", "3/minute")
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    limiter.reset()
    yield 3
    limiter.reset()


class TestTheLimitIsEnforced:
    def test_exceeding_the_login_limit_returns_429(self, tight_limit) -> None:
        email, _ = _register()
        codes = [
            client.post("/auth/login", json={"email": email, "password": "wrong"}).status_code
            for _ in range(tight_limit + 2)
        ]
        assert 429 in codes, f"never rate limited: {codes}"

    def test_requests_within_the_limit_are_untouched(self, tight_limit) -> None:
        email, _ = _register()
        # _register() spends two requests of the same bucket (register + login),
        # so reset before measuring or the budget is already half gone.
        limiter.reset()
        codes = [
            client.post("/auth/login", json={"email": email, "password": "Str0ng!Pass123"}).status_code
            for _ in range(tight_limit)
        ]
        assert 429 not in codes, f"limited too early: {codes}"

    def test_the_429_explains_itself(self, tight_limit) -> None:
        """A bare 429 with no body is indistinguishable from a broken server."""
        email, _ = _register()
        last = None
        for _ in range(tight_limit + 3):
            last = client.post("/auth/login", json={"email": email, "password": "wrong"})
        assert last.status_code == 429
        assert "rate" in last.text.lower() or "many" in last.text.lower()


class TestReadsAreNotLimited:
    def test_history_can_be_paged_freely(self, tight_limit) -> None:
        _, headers = _register()
        codes = [client.get("/analysis/history", headers=headers).status_code for _ in range(tight_limit + 4)]
        assert 429 not in codes, "throttled a read endpoint"

    def test_health_is_never_limited(self, tight_limit) -> None:
        codes = [client.get("/health").status_code for _ in range(tight_limit + 4)]
        assert set(codes) == {200}


class TestKeyingIsPerUserNotPerHost:
    def test_the_key_prefers_the_authenticated_user(self) -> None:
        """Two users on one NAT must not share a bucket."""
        _, h1 = _register()
        _, h2 = _register()

        class Req:
            def __init__(self, headers):
                self.headers = headers
                self.client = type("C", (), {"host": "10.0.0.1"})()
                self.scope = {"type": "http"}

        k1, k2 = rate_limit_key(Req(h1)), rate_limit_key(Req(h2))
        assert k1 != k2, "distinct users collapsed onto one rate-limit key"

    def test_anonymous_requests_fall_back_to_the_client_host(self) -> None:
        class Req:
            headers: dict = {}
            client = type("C", (), {"host": "203.0.113.9"})()
            scope = {"type": "http"}

        assert "203.0.113.9" in rate_limit_key(Req())

    def test_anonymous_credential_endpoints_share_a_host_bucket(self, tight_limit) -> None:
        """
        Deliberate, and worth stating rather than papering over: login and
        register have no authenticated identity to key on — establishing one is
        the point of the call — so they necessarily fall back to the host.

        That is the correct trade-off for exactly the endpoint being brute
        forced: an attacker must not be able to reset their own budget just by
        varying the email. The cost is that users behind one NAT share this
        bucket, which is why the limit is a generous 20/minute by default and
        why every *authenticated* endpoint keys per user instead.
        """
        email_a, _ = _register()
        email_b, _ = _register()
        limiter.reset()
        for _ in range(tight_limit + 2):
            client.post("/auth/login", json={"email": email_a, "password": "wrong"})
        res = client.post("/auth/login", json={"email": email_b, "password": "Str0ng!Pass123"})
        assert res.status_code == 429, "a shared host bucket is the intended behaviour here"


class TestConfiguration:
    def test_limits_are_configurable(self) -> None:
        assert hasattr(settings, "rate_limit_enabled")
        assert hasattr(settings, "rate_limit_auth")
        assert hasattr(settings, "rate_limit_analysis")

    def test_disabling_the_limiter_lets_everything_through(self, monkeypatch) -> None:
        """The escape hatch the rest of the suite relies on."""
        monkeypatch.setattr(settings, "rate_limit_enabled", False)
        monkeypatch.setattr(settings, "rate_limit_auth", "2/minute")
        limiter.reset()
        email, _ = _register()
        codes = [
            client.post("/auth/login", json={"email": email, "password": "wrong"}).status_code
            for _ in range(6)
        ]
        assert 429 not in codes
        limiter.reset()

    def test_the_default_is_not_effectively_unlimited(self) -> None:
        """Guards against the limiter being 'on' but set so high it never fires,
        which would make every other test in this file pass vacuously."""
        import re

        n = int(re.match(r"\s*(\d+)", settings.rate_limit_auth).group(1))
        assert n <= 60, f"default auth limit {settings.rate_limit_auth} is not a real limit"
