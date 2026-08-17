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
        try:
            response = httpx.post(
                url,
                params={"key": self.api_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(f"Gemini returned HTTP {response.status_code}: {response.text[:300]}")

        try:
            body = response.json()
            return body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected Gemini response shape: {exc}") from exc
