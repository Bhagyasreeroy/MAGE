"""
agents/llm_client.py
──────────────────────
GeminiClient — thin wrapper around Google's Gemini REST API.

MAGE's recommendations have always been fully deterministic (retrieved
methodology text composed with computed findings — see
agents/recommendation_agent.py). This is the project's first real LLM
call, used only when a user explicitly opts into "LLM mode" in the
follow-up chat; the RAG path is untouched.

It also backs voice input: ``transcribe()`` sends recorded audio to the
same generateContent endpoint, which is why voice-to-text added no new
dependency and no second API key.

Uses plain httpx (already a dependency, used the same way for REST
ingestion in data_pipeline/ingestion.py) rather than the official Google
SDK, so no new dependency is needed. The module-level ``httpx.post`` call
is what lets tests fake this cleanly via
``monkeypatch.setattr(httpx, "post", ...)`` — the same pattern already
used for ingestion's outbound HTTP calls.
"""

from __future__ import annotations

import base64
import logging
import random
import time
from typing import Any

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
# Audio takes materially longer to process than a text prompt of equivalent
# "size", and a 20s ceiling was observed cutting off ~30s clips mid-flight.
TRANSCRIBE_TIMEOUT = 60.0

# Without an explicit "output only the transcript" instruction, Gemini
# prefixes its answer with conversational scaffolding ("Sure! Here is the
# transcription:"), which would land verbatim in the user's goal box.
TRANSCRIBE_PROMPT = (
    "Transcribe the speech in this audio to English text, verbatim. "
    "Output only the transcript itself: no preamble, no explanation, no "
    "quotation marks, no speaker labels, no timestamps. If the audio "
    "contains no intelligible speech, output nothing at all."
)

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
        return self._generate_content([{"text": prompt}], timeout=timeout)

    def transcribe(
        self,
        audio: bytes,
        mime_type: str,
        timeout: float = TRANSCRIBE_TIMEOUT,
    ) -> str:
        """Transcribe spoken audio to English text.

        `audio` is the raw file bytes and `mime_type` must be one Gemini
        accepts natively (wav, mp3, ogg, aac, flac, aiff). Notably *not*
        webm, which is what Chrome's MediaRecorder produces by default —
        the browser re-encodes to wav before upload, so that guarantee is
        the caller's to keep.

        Returns the transcript, stripped. An empty string means Gemini
        heard no intelligible speech; that is a normal outcome for a
        recording of silence, not a failure.

        Raises
        ------
        LLMError
            Same conditions as `generate`.
        """
        parts: list[dict[str, Any]] = [
            {"text": TRANSCRIBE_PROMPT},
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(audio).decode("ascii"),
                }
            },
        ]
        return self._generate_content(parts, timeout=timeout, allow_empty=True).strip()

    def _generate_content(
        self,
        parts: list[dict[str, Any]],
        timeout: float,
        allow_empty: bool = False,
    ) -> str:
        """POST one generateContent request, retrying transient failures.

        Shared by every call shape: `parts` is Gemini's content-part list,
        so a plain text prompt and a prompt-plus-audio request differ only
        in what the caller assembles, never in how failures are handled.
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
                    json={"contents": [{"parts": parts}]},
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
                        return self._extract_text(body, allow_empty=allow_empty)
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
    def _extract_text(body: dict[str, Any], *, allow_empty: bool) -> str:
        """Pull the generated text out of a 200 response body.

        When `allow_empty`, a candidate carrying no text part yields "".
        Transcription needs that: asked to transcribe silence, Gemini
        returns a candidate with no parts at all, and treating that as a
        malformed response would turn "you recorded nothing" into an
        error dialog.
        """
        candidate = body["candidates"][0]
        parts = candidate.get("content", {}).get("parts")
        if not parts:
            if allow_empty:
                return ""
            raise KeyError("parts")
        return parts[0]["text"]

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff with jitter, clamped so the total stays bounded."""
        base = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        delay = base + random.uniform(0, BACKOFF_JITTER_SECONDS)
        # Budget the remaining sleeps so the sum can never exceed the ceiling.
        remaining_sleeps = max(1, MAX_ATTEMPTS - attempt)
        return min(delay, MAX_TOTAL_BACKOFF_SECONDS / remaining_sleeps)
