"""
tests/conftest.py
──────────────────
Shared test configuration.

**Rate limiting is off by default under test, and on only where it is the thing
being tested.**

The suite makes hundreds of calls to `/auth/register`, `/auth/login` and
`/analysis/run` in a few seconds. Those are exactly the endpoints M8 throttles,
and the credential endpoints necessarily key on the host (there is no
authenticated identity yet — establishing one is the point of the call), so the
whole suite shares a single bucket. Left on, the limiter exhausts it partway
through and roughly 25 unrelated tests start failing with 429s that say nothing
about the code under test.

Worse, they fail *positionally*: each file passes alone and fails in the full
run, which is the most expensive kind of flake to chase.

`tests/backend/test_rate_limit.py` opts back in explicitly via its own fixture,
so the feature is still covered — and that file also asserts the production
default is a real limit, so switching it off here cannot quietly make the
production configuration permissive.
"""

from __future__ import annotations

import pytest

from backend.core.config import settings


@pytest.fixture(autouse=True)
def _disable_rate_limiting_by_default(request):
    """Off for every test except those that explicitly exercise the limiter."""
    if "test_rate_limit" in str(getattr(request.node, "fspath", "")):
        yield
        return

    original = getattr(settings, "rate_limit_enabled", True)
    settings.rate_limit_enabled = False
    try:
        yield
    finally:
        settings.rate_limit_enabled = original
