"""
agents/llm_client.py
──────────────────────
GeminiClient — thin wrapper around Google's Gemini REST API.

MAGE's recommendations have always been fully deterministic (retrieved
methodology text composed with computed findings — see
agents/recommendation_agent.py). This is the project's first real LLM
call, used only when a user explicitly opts into "LLM mode" in the
follow-up chat; the RAG path is untouched.

Uses plain httpx (already a dependency, used the same way for REST
ingestion in data_pipeline/ingestion.py) rather than the official Google
SDK, so no new dependency is needed. The module-level ``httpx.post`` call
is what lets tests fake this cleanly via
``monkeypatch.setattr(httpx, "post", ...)`` — the same pattern already
used for ingestion's outbound HTTP calls.
"""

from __future__ import annotations

import logging
import random
import time

import httpx

from backend.core.config import settings

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
# Gemini model availability shifts over time (gemini-2.0-flash was retired
# entirely; the "-latest" alias was observed returning 503s while this
# pinned version responded normally) — if this starts 404ing, check
# GET https://generativelanguage.googleapis.com/v1beta/models?key=... for
# what's currently live on the configured key.
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT = 20.0

# NFR-03 — bounded exponential backoff.
#
# Retry only what a retry can plausibly fix. A 503 or a read timeout is usually
# momentary capacity pressure, and this project has already met it: 2.5-flash is
# pinned because the "-latest" alias was observed returning 503s. A 400 or 403
# will fail identically three times, so retrying those only triples the wait a
# user spends before being told the same thing.
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0
# Jitter keeps several concurrent analyses from retrying in lockstep and
# re-creating the burst that rate-limited them.
BACKOFF_JITTER_SECONDS = 0.25
# Ceiling on the *sleeping* portion, asserted by the tests: someone is waiting
# on this call, so the worst case has to be defensible rather than emergent.
MAX_TOTAL_BACKOFF_SECONDS = 10.0
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class LLMError(Exception):
    """Raised for a failed or misconfigured LLM call."""


class GeminiClient:
    """Minimal client for Gemini's generateContent REST endpoint."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, timeout: float = DEFAULT_TIMEOUT) -> str:
        """Send `prompt` to Gemini and return the generated text.

        Raises
        ------
        LLMError
            If no API key is configured, the request fails, or the
            response doesn't contain the expected generated-text shape.
        """
        if not self.is_configured:
            raise LLMError("No Gemini API key configured (set GEMINI_API_KEY).")

        url = f"{GEMINI_API_BASE}/{self.model}:generateContent"
        last_error: str = "no attempt was made"

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = httpx.post(
                    url,
                    params={"key": self.api_key},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=timeout,
                )
            except httpx.HTTPError as exc:
                # Transport-level: timeout, connection refused, DNS. Always
                # worth one more try.
                last_error = f"Gemini request failed: {exc}"
            else:
                if response.status_code == 200:
                    try:
                        body = response.json()
                        return body["candidates"][0]["content"]["parts"][0]["text"]
                    except (KeyError, IndexError, ValueError) as exc:
                        # A 200 with an unexpected shape is not transient —
                        # the same prompt will produce the same shape.
                        raise LLMError(f"Unexpected Gemini response shape: {exc}") from exc

                detail = f"Gemini returned HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in RETRYABLE_STATUS:
                    raise LLMError(detail)
                last_error = detail

            if attempt < MAX_ATTEMPTS:
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Gemini attempt %d/%d failed (%s); retrying in %.2fs.",
                    attempt, MAX_ATTEMPTS, last_error[:120], delay,
                )
                time.sleep(delay)

        raise LLMError(f"Gemini failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}")

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff with jitter, clamped so the total stays bounded."""
        base = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        delay = base + random.uniform(0, BACKOFF_JITTER_SECONDS)
        # Budget the remaining sleeps so the sum can never exceed the ceiling.
        remaining_sleeps = max(1, MAX_ATTEMPTS - attempt)
        return min(delay, MAX_TOTAL_BACKOFF_SECONDS / remaining_sleeps)
