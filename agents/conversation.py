"""
agents/conversation.py
───────────────────────
Conversation state for the follow-up chat.

Every chat turn re-runs the full pipeline, and until now each one arrived as a
cold start: the request carried a `goal` and nothing else. That is what made
follow-ups feel unheard. Two separate things were missing, and this module
supplies both.

**Reference.** "Why?", "what about revenue?", "explain that" have no meaning on
their own — the subject lives in the previous turn. :func:`resolve` folds the
last question back in so classification and retrieval have something to work
with. The fold is deliberately conservative: only questions that actually *look*
elliptical are rewritten, and the rewrite leads with the current question so the
carried text informs retrieval rather than displacing it.

Resolution is heuristic, not an LLM call. The RAG path is the one that works
with no API key configured, and making conversational follow-up depend on
Gemini would quietly break it for exactly the deployments that chose RAG mode.

**Repetition.** Assistant turns carry the knowledge-base documents they cited,
so the next turn can tell what the user has already been shown and prefer
material they haven't seen.

Everything here is defensive. The payload is client-supplied and the
conversation is *additive* grounding — a malformed turn must degrade the answer
slightly, never fail the run. Compare `RecommendationAgent._prior_runs`, which
takes the same stance for run memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# How much history is read. Retrieval is driven by the last question, not by
# the whole transcript, so a long tail adds payload without adding signal.
MAX_TURNS = 16

# Per-turn truncation. Assistant replies are five recommendation cards flattened
# into prose and can run to several thousand characters; carried into a query
# whole they would swamp it.
MAX_TURN_CHARS = 1200

_USER = "user"
_ASSISTANT = "assistant"
_ROLES = frozenset({_USER, _ASSISTANT})

# A question that opens with one of these is continuing the previous one rather
# than starting a new subject.
_CONTINUATION_OPENERS = (
    "why",
    "why not",
    "and",
    "but",
    "so",
    "then",
    "what about",
    "how about",
    "what if",
    "ok",
    "okay",
    "also",
    "plus",
    "besides",
    "instead",
)

# Bare demonstratives and pronouns. These only refer to something when that
# something was named earlier, which is precisely the case we need to repair.
_ANAPHORA_RE = re.compile(
    r"(?<!\w)(it|its|that|this|those|these|them|they|their|the same|the former|the latter|one)(?!\w)",
    re.IGNORECASE,
)

# Below this, a question is too short to stand alone ("explain more", "in
# detail?", "and revenue?").
_SHORT_QUESTION_WORDS = 5


@dataclass(frozen=True)
class Turn:
    """One chat turn, normalised."""

    role: str
    content: str
    # Knowledge-base documents this turn cited. Assistant turns only; user
    # turns cite nothing.
    sources: tuple[str, ...] = field(default=())

    @property
    def is_user(self) -> bool:
        return self.role == _USER


@dataclass(frozen=True)
class ResolvedGoal:
    """
    A chat turn's goal, with any carried context made explicit.

    `text` is what classification and retrieval should read. `display` — the
    text the user actually typed — is what gets persisted and shown, because
    the run history should record the question they asked, not our expansion
    of it.
    """

    text: str
    display: str
    is_followup: bool
    carried: str = ""


def _clean(value: Any) -> str:
    """Coerce an untrusted value to a bounded, whitespace-normalised string."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:MAX_TURN_CHARS].strip()


def _clean_sources(value: Any) -> tuple[str, ...]:
    """Source paths from an untrusted list, deduplicated, order preserved."""
    if not isinstance(value, list):
        return ()
    seen: dict[str, None] = {}
    for item in value:
        source = _clean(item)
        if source:
            seen.setdefault(source, None)
    return tuple(seen)


def normalize(raw: Any) -> list[Turn]:
    """
    Parse a client-supplied conversation into `Turn`s, discarding anything odd.

    Accepts dicts (the wire form) and `Turn`s alike, so the orchestrator can
    normalise once and hand the result on as ordinary context without callers
    having to care which form they received.
    """
    if isinstance(raw, Turn):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    turns: list[Turn] = []
    for item in raw[-MAX_TURNS:]:
        if isinstance(item, Turn):
            turns.append(item)
            continue
        if not isinstance(item, dict):
            continue
        role = _clean(item.get("role")).lower()
        content = _clean(item.get("content"))
        if role not in _ROLES or not content:
            continue
        turns.append(
            Turn(
                role=role,
                content=content,
                sources=_clean_sources(item.get("sources")) if role == _ASSISTANT else (),
            )
        )
    return turns


def previously_cited(turns: list[Turn]) -> set[str]:
    """Knowledge-base documents already shown to the user in this conversation."""
    return {source for turn in turns if not turn.is_user for source in turn.sources}


def last_question(turns: list[Turn]) -> str:
    """The most recent thing the user asked, or "" if they haven't yet."""
    for turn in reversed(turns):
        if turn.is_user:
            return turn.content
    return ""


def looks_elliptical(goal: str) -> bool:
    """
    True when `goal` cannot be understood without the previous turn.

    Three signals, any of which is enough:

    * it opens with a continuation word ("why…", "what about…", "and…");
    * it leans on a demonstrative or pronoun with nothing to bind it to;
    * it is too short to carry a subject at all.

    Kept deliberately narrow. A false positive costs a little retrieval
    precision — the previous question is appended to a self-contained one — so
    a fully-formed question like "What is the correlation between price and
    revenue?" must not trip any of these, and does not.
    """
    text = goal.strip().lower()
    if not text:
        return False

    stripped = text.rstrip("?!.")
    if any(
        stripped == opener or stripped.startswith(f"{opener} ") or stripped.startswith(f"{opener},")
        for opener in _CONTINUATION_OPENERS
    ):
        return True

    if _ANAPHORA_RE.search(text):
        return True

    return len(text.split()) < _SHORT_QUESTION_WORDS


def resolve(goal: str, turns: list[Turn]) -> ResolvedGoal:
    """
    Work out what this turn is really asking.

    With no history, or with a question that already stands on its own, the
    goal passes through untouched — the first turn of a conversation must
    behave exactly as it did before this module existed.

    Otherwise the previous question is appended. Appended, not prepended: the
    embedding sees both subjects either way, but leading with the current
    question keeps it dominant, the same principle
    `RecommendationAgent._build_query` already applies to run memory.
    """
    display = goal.strip()
    if not turns or not looks_elliptical(display):
        return ResolvedGoal(text=display, display=display, is_followup=False)

    carried = last_question(turns)
    if not carried:
        return ResolvedGoal(text=display, display=display, is_followup=False)

    return ResolvedGoal(
        text=f"{display} {carried}".strip(),
        display=display,
        is_followup=True,
        carried=carried,
    )
