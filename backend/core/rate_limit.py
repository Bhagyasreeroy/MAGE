"""
core/rate_limit.py
───────────────────
Application-level rate limiting (M8).

Previously this was only "recommended via a reverse proxy in production", which
in practice meant nothing enforced it. This module supplies a `slowapi` limiter
wired into the app, applied narrowly to the endpoints where a cheap request
costs something real: credential endpoints (stuffing), the analysis run (CPU and
the whole pipeline), and the two LLM-backed endpoints (they spend Gemini quota).

Reads are deliberately left alone. Throttling someone paging through their own
history would be a worse bug than the one this fixes.

**Keying is the part worth reviewing.** slowapi's stock key is the remote
address. That would put every user behind one NAT — a university lab, which is
exactly where this gets demoed — into a single shared bucket, so one person
running a few analyses would lock out the room. The key below prefers the
authenticated identity and falls back to the host only when there isn't one.
"""

from __future__ import annotations

import logging

from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.core.config import settings
from backend.core.security import decode_token

logger = logging.getLogger(__name__)


def rate_limit_key(request) -> str:
    """
    Bucket by authenticated user, falling back to client host.

    The token is only *decoded*, never trusted for authorisation — a forged or
    expired token simply falls through to host-based limiting, which is the
    conservative direction: an attacker cannot widen their own allowance by
    presenting garbage, they can only land in the shared anonymous bucket.
    """
    auth = ""
    try:
        auth = request.headers.get("Authorization", "") or ""
    except Exception:  # noqa: BLE001 - a header-less scope is anonymous
        auth = ""

    if auth.startswith("Bearer "):
        try:
            payload = decode_token(auth.split(" ", 1)[1])
            subject = payload.get("sub")
            if subject:
                return f"user:{subject}"
        except Exception:  # noqa: BLE001 - unreadable token → anonymous
            pass

    try:
        return f"host:{get_remote_address(request)}"
    except Exception:  # noqa: BLE001
        host = getattr(getattr(request, "client", None), "host", None)
        return f"host:{host or 'unknown'}"


def _enabled() -> bool:
    """Read at call time, not import time, so tests can toggle it."""
    return bool(getattr(settings, "rate_limit_enabled", True))


limiter = Limiter(
    key_func=rate_limit_key,
    enabled=True,  # the per-request check below is the real switch
    # Off deliberately: with headers enabled the *decorator* insists every
    # limited endpoint also declare a `response: Response` parameter so it can
    # inject X-RateLimit-*, which would mean reshaping five handler signatures
    # for a header. SlowAPIMiddleware still adds them where it can, and the 429
    # body states the limit in words, which is what a caller actually needs.
    headers_enabled=False,
)


def auth_limit() -> str:
    return settings.rate_limit_auth


def analysis_limit() -> str:
    return settings.rate_limit_analysis


def llm_limit() -> str:
    return settings.rate_limit_llm


def limit_exempt_when_disabled(request) -> bool:
    """slowapi `exempt_when` hook — bypass entirely when switched off."""
    return not _enabled()
