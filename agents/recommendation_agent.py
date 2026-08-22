"""
agents/recommendation_agent.py
───────────────────────────────
RecommendationAgent — RAG-grounded, expertise-adapted EDA recommendations.

Two paths, tried in order:

1. **Direct answer** (QAAgent): if the goal is a specific factual question
   ("which column has the most missing values?", "what's the correlation
   between X and Y?"), answer it directly from MiningAgent's already-computed
   output — no retrieval needed. This is what makes follow-up chat feel
   conversational instead of re-dumping a wall of methodology text for every
   message.

2. **Grounded recommendations** (fallback, for broader goals), built in three
   passes and divided by a ``_Budget``:

   * *Finding-led* — for each concrete pattern MiningAgent found (a specific
     outlier count, a specific correlation, a specific skew), retrieve the one
     most relevant methodology chunk and lead the recommendation with the
     concrete finding, e.g. "'revenue' has 3 outliers (IQR method).
     <methodology excerpt>" — rather than surfacing the raw excerpt alone,
     which reads as generic textbook material disconnected from the user's
     actual data.
   * *Goal-led* — retrieved for the question and nothing else. Findings are a
     property of the dataset and so are identical on every turn against the
     same data; these are the cards that move when the question moves, which
     is why the budget reserves slots for them mid-conversation.
   * *Broad* — the full context query, including this user's prior runs
     (Objective 4), filling whatever the first two passes left.

   Mid-conversation the findings are additionally ranked against the question
   and documents already cited earlier in the chat are held back, so a
   follow-up is answered rather than the opening report repeated. See
   ``agents/conversation.py``.

Recommendation text is produced in three registers (FR-04), one per expertise
level the system offers:
    - ``text_plain``     : plain language, no jargon — **Beginner**.
    - ``text_analyst``   : the finding plus the statistic supporting it, in
      working language but without full methodological framing — **Analyst**.
    - ``text_technical`` : the finding in context with the complete
      methodology excerpt, markdown intact — **Data Scientist**.

The three are deliberately nested rather than independently written: each
level adds detail the one below omits, so a reader moving up a level never
loses information they had. ``ANALYST_SENTENCES`` and ``PLAIN_SENTENCES`` set
where each stops.

**LLM mode** (opt-in, ``context["mode"] == "llm"``): bypasses both paths
above entirely. Instead of retrieval, the dataset's already-computed
features (MiningAgent's stats/patterns, IngestionAgent's warnings — never
raw rows) and the user's goal are handed to Gemini for a freeform,
conversational response. Deliberately not grounded — LLM recommendations
always carry empty ``sources``, which is what keeps them visually distinct
from RAG output (no citation chips) rather than something to "fix" later.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from agents import conversation as conversation_state
from agents.llm_client import GeminiClient, LLMError
from agents.mining_agent import ATTRIBUTION_PATTERN_MARKER
from agents.qa_agent import QAAgent
from rag.embeddings import embed_batch
from rag.knowledge_loader import KnowledgeBaseLoader
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Recommendations to return per run. Keep small — these are meant to be
# the highest-signal, most goal-relevant grounded suggestions, not an
# exhaustive dump of the knowledge base.
TOP_K_RETRIEVAL = 8
MAX_RECOMMENDATIONS = 5

# Chunks below this similarity are considered too weak to ground a
# recommendation in and are dropped rather than surfaced with low confidence.
MIN_CONFIDENCE = 0.15

# Retrieval floor for a *goal-led* card, which is a stronger claim than the one
# `MIN_CONFIDENCE` guards. A finding-led card says "this document relates to
# this statistic from your data"; a goal-led card says "this document answers
# your question", and it is the only kind of card a question with no relevant
# finding can produce. Held to a higher bar accordingly.
#
# Measured, not guessed. Across twelve realistic methodology questions and
# eight questions about the *contents* of a dataset ("tell me about drake
# artist", "which country has the most artists"), the two bands do not
# overlap or even approach each other: methodology bottoms out at 0.337, and
# data-content tops out at 0.246. 0.30 sits in the empty gap between them.
#
# This is what stops the failure that prompted it. The knowledge base holds EDA
# methodology and has nothing whatever to say about an artist; retrieval knew
# that and scored its best document 0.10, but 0.10 cleared `MIN_CONFIDENCE` in
# other passes and a card about SHAP and LIME was served as the answer.
GOAL_LED_MIN_CONFIDENCE = 0.30

# Cosine floor, goal against finding, below which a finding earns no card on a
# follow-up turn. Only applied to follow-ups — see `_patterns_for_goal`.
#
# Set from measured separation on all-MiniLM-L6-v2 rather than guessed. Across
# a representative mining output, an on-topic pairing scores 0.40-0.79
# ("handle the missing values" against a missing-values finding: 0.44; "compare
# the clusters" against a clustering finding: 0.52), while an off-topic one
# sits at 0.02-0.17. The gap between those bands is wide and empty, and 0.20
# is inside it.
PATTERN_RELEVANCE_FLOOR = 0.20


@dataclass(frozen=True)
class _Budget:
    """
    How the ``MAX_RECOMMENDATIONS`` slots are divided between the three passes.

    The split is the fix for a specific failure. Findings are a property of the
    *dataset*, so a pattern-led card is identical on every turn against the
    same data. With patterns free to claim every slot — which they were, since
    a real dataset yields well over five — the goal-led pass never ran and the
    question never reached retrieval except as a prefix on a pattern query.
    Every follow-up came back as a copy of the first answer.

    Reserving slots is what breaks that: whatever the findings are, some of the
    answer is always retrieved for the question that was actually asked.
    """

    # Cards led by a MiningAgent finding, grounded against it.
    pattern_led: int
    # Cards retrieved for the question alone — no finding, nothing from the
    # dataset in the query. These are the ones that change when the question
    # changes.
    goal_led: int
    # Whether to pad any remaining slots from the broad context query.
    #
    # Off for a follow-up, and that is the point. The broad query is mostly
    # MiningAgent's findings, so it scores well against the knowledge base
    # whatever was asked — which meant a question the other two passes had
    # correctly refused to answer still came back with a confident-looking
    # card. Asked "tell me about drake artist", retrieval scored the best
    # available document at 0.10, far below `MIN_CONFIDENCE`, and both passes
    # rightly produced nothing; the broad pass then returned a 0.28 match on
    # SHAP and LIME and it was presented as the answer.
    #
    # Padding is only defensible where the cards are not answering a question.
    # On the opening report it is background grounding. On a follow-up there is
    # nothing to pad *with* that is about what was asked.
    broad_fill: bool


# The opening run is a report: the user asked for analysis and the findings are
# the answer, so patterns keep every slot and behaviour is unchanged from
# before conversations existed.
INITIAL_BUDGET = _Budget(pattern_led=MAX_RECOMMENDATIONS, goal_led=0, broad_fill=True)

# A follow-up is a question. Findings still earn cards — they are what grounds
# the answer in the user's own data — but they no longer crowd out the answer.
FOLLOWUP_BUDGET = _Budget(pattern_led=2, goal_led=3, broad_fill=False)

# How much of the retrieved methodology each register keeps. The technical
# register is uncapped (it keeps the whole excerpt), so these two are what
# separate the three levels — keep them distinct or the registers converge and
# FR-04's "visibly alters output language" stops being true.
PLAIN_SENTENCES = 2
ANALYST_SENTENCES = 4

# Column names carry qualifiers a person never types. "Total Streams (in
# millions)" is "total streams" when someone asks about it.
_PARENTHETICAL_RE = re.compile(r"\([^)]*\)")
_NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")

# Below this a column name matches too much ordinary prose to be evidence that
# the question is about that column.
MIN_COLUMN_NAME_CHARS = 4

_MARKDOWN_STRIP_RE = re.compile(r"[#*`|>]|^-\s+", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\s+")
_HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s*")
_INLINE_MD_RE = re.compile(r"[*`_]")


def _column_key(text: str) -> str:
    """A column name (or a sentence) reduced to comparable words."""
    stripped = _PARENTHETICAL_RE.sub(" ", str(text)).lower()
    return _WHITESPACE_RE.sub(" ", _NON_WORD_RE.sub(" ", stripped)).strip()


def _columns_named(goal: str, columns: list[str]) -> list[str]:
    """Columns the question actually names, longest name first.

    Whole-word matching on the qualifier-stripped name, so "…more total
    streams?" names `Total Streams (in millions)` but "collaborating" does not
    name `Collaborative Streams (in millions)`.
    """
    goal_key = _column_key(goal)
    if not goal_key:
        return []
    named = [
        column
        for column in columns
        if len(_column_key(column)) >= MIN_COLUMN_NAME_CHARS
        and re.search(rf"(?<!\w){re.escape(_column_key(column))}(?!\w)", goal_key)
    ]
    return sorted(named, key=lambda c: len(_column_key(c)), reverse=True)


def _mentions_any_column(pattern: str, columns: list[str]) -> bool:
    """True if `pattern` is about one of `columns`. Findings quote the column
    name verbatim, so this is a substring test on both the raw and reduced
    forms."""
    lowered = pattern.lower()
    reduced = _column_key(pattern)
    return any(
        str(column).lower() in lowered or (_column_key(column) and _column_key(column) in reduced)
        for column in columns
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from `obj`, whether it's a dict or a Pydantic/attr object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _is_separator_row(line: str) -> bool:
    """True for ``|---|:--:|`` separators and bare horizontal rules."""
    return bool(line) and set(line) <= set("-|:= ")


def _split_row(line: str) -> list[str]:
    """Split one markdown table row into stripped cell values."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_block_to_sentences(block: list[str]) -> list[str]:
    """
    Rewrite one markdown table as one labelled sentence per row.

    ``| Goal | Chart |`` / ``| Compare groups | Box plot |`` becomes
    ``Goal: Compare groups; Chart: Box plot.`` — the header row supplies the
    field names rather than being emitted as content.

    This is the point of the whole function. Several knowledge-base documents
    carry their real guidance as a decision table, so dropping tables meant
    those documents contributed only their headings. Inlining the raw cells
    instead produces an unreadable run ("numeric Line chart … Choropleth …"),
    which is why the values are labelled.
    """
    rows = [_split_row(line) for line in block if not _is_separator_row(line)]
    if not rows:
        return []

    headers, data_rows = rows[0], rows[1:]
    if not data_rows:
        # A single row with no body is a header with nothing under it; keep the
        # values rather than inventing labels for them.
        return [" ".join(cell for cell in headers if cell)]

    sentences: list[str] = []
    for cells in data_rows:
        parts: list[str] = []
        for index, cell in enumerate(cells):
            if not cell:
                continue  # an empty cell labelled "Header:" is noise
            label = headers[index] if index < len(headers) else ""
            parts.append(f"{label}: {cell}" if label else cell)
        if parts:
            sentences.append("; ".join(parts) + ".")
    return sentences


def _flatten_markdown_tables(text: str) -> str:
    """Replace every markdown table in `text` with labelled per-row sentences."""
    out: list[str] = []
    block: list[str] = []

    def flush() -> None:
        if block:
            out.extend(_table_block_to_sentences(block))
            block.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.count("|") >= 2:
            block.append(stripped)
        elif _is_separator_row(stripped):
            # Inside a table this is the header separator; outside it is a bare
            # horizontal rule, which carries nothing worth keeping.
            if block:
                block.append(stripped)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def _strip_markdown_structure(text: str) -> str:
    """Convert a retrieved knowledge-base chunk into plain readable prose.

    Tables are rewritten row-by-row into labelled sentences (see
    :func:`_table_block_to_sentences`) rather than discarded; headings and
    inline emphasis markers are removed. Prose before and after a table is
    preserved in place.
    """
    joined = _flatten_markdown_tables(text)
    joined = _HEADING_RE.sub("", joined)      # drop "## Heading" markers
    joined = _INLINE_MD_RE.sub("", joined)    # drop * ` _ emphasis
    return joined


def _drop_leading_fragment(sentences: list[str]) -> list[str]:
    """Overlapping chunks can start mid-sentence (lowercase, no leading
    capital). Drop that leading fragment so text starts clean — used for
    both registers, not just the plain-language one."""
    while len(sentences) > 1 and sentences[0].strip()[:1].islower():
        sentences = sentences[1:]
    return sentences


def _clean_technical(text: str) -> str:
    """Readable chunk prose: markdown tables/headings removed, leading
    sentence fragment dropped, whitespace normalised."""
    cleaned = _strip_markdown_structure(text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", cleaned.strip()) if s.strip()]
    result = " ".join(_drop_leading_fragment(sentences)).strip()
    result = _WHITESPACE_RE.sub(" ", result)
    return result or _WHITESPACE_RE.sub(" ", cleaned).strip() or text.strip()


def _first_sentences(text: str, limit: int) -> str:
    """Strip Markdown and return the first `limit` complete sentences."""
    cleaned = _strip_markdown_structure(text)
    cleaned = _MARKDOWN_STRIP_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    sentences = _drop_leading_fragment([s for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()])
    trimmed = " ".join(sentences[:limit]).strip()
    return trimmed or cleaned[:200]


def _simplify(text: str) -> str:
    """Plain-language register (Beginner): Markdown stripped, first couple of sentences."""
    return _first_sentences(text, PLAIN_SENTENCES)


def _analyst(text: str, title: str) -> str:
    """
    Analyst register: working language, attributed, with the reasoning kept.

    Two things separate this from the plain register:

    * **More of the excerpt** — ``ANALYST_SENTENCES`` rather than
      ``PLAIN_SENTENCES``, so the reasoning behind the advice survives and not
      just its headline. Markdown is still stripped; an analyst wants the
      substance, not the document structure.
    * **Inline attribution** — the methodology is named in the sentence. This
      is not decoration: it is the difference between "handle the missing
      values" and knowing *which* documented procedure is being invoked, which
      is precisely what a working analyst needs in order to check it.

    The attribution also makes the separation *structural* rather than merely
    a matter of length. Several knowledge-base chunks are a single sentence
    long, and against those a purely length-based rule collapses this register
    onto the plain one — which would quietly break FR-04 for exactly the
    recommendations whose source is shortest.
    """
    body = _first_sentences(text, ANALYST_SENTENCES)
    return f"Per {title}: {body}" if title else body


class RecommendationAgent:
    """
    Generates RAG-grounded, expertise-adapted EDA recommendations — or a
    direct computed answer when the goal is a specific factual question.

    Input context keys consumed:
        - ``goal``                      : analytical goal
        - ``expertise_level``           : adapts language complexity
        - ``mode``                      : "rag" (default) | "llm"
        - ``MiningAgent_output``        : statistical profile + patterns
        - ``IngestionAgent_output``     : data-quality warnings / row count
        - ``VisualizationAgent_output`` : selected chart specs

    Output keys produced:
        - ``recommendations`` : list of recommendation dicts
        - ``rag_sources``     : list of unique knowledge-base sources cited
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        qa_agent: QAAgent | None = None,
        llm_client: GeminiClient | None = None,
    ) -> None:
        self._vector_store = vector_store or VectorStore()
        self._qa_agent = qa_agent or QAAgent()
        self._llm_client = llm_client or GeminiClient()
        self._kb_loaded = False

    def _ensure_kb_loaded(self) -> None:
        """Lazily populate the vector store from the knowledge base on first use."""
        if self._kb_loaded:
            return

        self._vector_store.initialize()
        # Sync unconditionally rather than skipping when the store is
        # non-empty. The old "is anything in here?" guard meant a knowledge
        # base document added after the first ever run was never indexed —
        # invisible, with no error. Sync is content-addressed (NFR-02), so an
        # unchanged corpus costs one id lookup and embeds nothing.
        chunks = KnowledgeBaseLoader().load_all()
        if chunks:
            added = self._vector_store.sync_documents(
                [c["text"] for c in chunks],
                metadata=[{**c["metadata"], "source": c["source"]} for c in chunks],
            )
            if added:
                logger.info("Indexed %d new knowledge-base chunk(s).", added)
        self._kb_loaded = True

    @staticmethod
    def _prior_runs(context: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Prior-run memory from the context, defensively normalised.

        Supplied by the service layer, which owns the database. Anything
        malformed is discarded rather than raised on: memory is additive
        grounding, and a bad memory row must not be able to fail an analysis.
        """
        prior = context.get("prior_runs")
        if not isinstance(prior, list):
            return []
        return [run for run in prior if isinstance(run, dict)]

    @staticmethod
    def _retrieval_goal(context: dict[str, Any]) -> str:
        """
        The question as retrieval should read it.

        ``resolved_goal`` is set by the orchestrator when the turn was
        elliptical and had to be resolved against the previous one; otherwise
        the goal is already self-contained and is used as typed.
        """
        return str(context.get("resolved_goal") or context.get("goal", ""))

    def _build_query(self, context: dict[str, Any]) -> str:
        """Compose a goal-conditioned retrieval query from goal + upstream findings."""
        parts = [self._retrieval_goal(context)]

        mining_output = context.get("MiningAgent_output")
        patterns = _get(mining_output, "patterns", []) or []
        parts.extend(str(p) for p in patterns)

        ingestion_output = context.get("IngestionAgent_output")
        warnings = _get(ingestion_output, "warnings", []) or []
        parts.extend(str(w) for w in warnings)

        # What this user previously found on comparable questions (Objective 4).
        # Appended last so the current goal and this run's own findings still
        # dominate the query — history informs retrieval, it does not replace
        # the present question.
        for run in self._prior_runs(context):
            parts.extend(str(f) for f in (run.get("findings") or []))

        return " ".join(p for p in parts if p)

    @staticmethod
    def _patterns_for_goal(goal: str, patterns: list[str], columns: list[str] | None = None) -> list[str]:
        """
        Order findings by how much they have to do with the question, dropping
        the ones that have nothing to do with it.

        Used on follow-up turns only. MiningAgent emits its findings in its own
        order — correlations, then outliers, then skew, and so on — which is
        the right order for a report and the wrong one for an answer. Asked
        about missing values, the user should not be handed a card about a
        correlation simply because correlations are computed first.

        When the question names a column, findings about *that* column come
        first, ahead of embedding order. Cosine similarity cannot reliably tell
        "Total Streams (in millions)" from "Collaborative Streams (in
        millions)" — the names are nearly the same sentence — so asked which
        artists were outliers in Total Streams, the run led with the outliers
        in Collaborative Streams. A named column is a much stronger signal
        about what was asked than the distance between two column names, so it
        is applied on top rather than mixed into the score.

        Returning nothing is a valid answer, and an important one. When the
        data has no finding bearing on the question, the honest response is a
        purely goal-led one — the reserved slots and the broad pass still fill
        the turn. Keeping a best-of-a-bad-lot finding instead produced the
        worst output this agent can produce: a real statistic about clusters
        presented as the reason for guidance about missing values, which reads
        as a causal claim and is not one.

        Embedding failure is not fatal — the original order is returned and the
        turn degrades to the previous behaviour.
        """
        if not patterns or not goal.strip():
            return patterns

        try:
            vectors = embed_batch([goal, *patterns])
        except Exception:  # noqa: BLE001 - relevance is a refinement, not a requirement
            logger.warning("Could not embed patterns for goal relevance; keeping mining order.", exc_info=True)
            return patterns

        goal_vec, pattern_vecs = vectors[0], vectors[1:]
        norms = np.linalg.norm(pattern_vecs, axis=1) * float(np.linalg.norm(goal_vec))
        # A zero-norm embedding would divide by zero; score it 0 instead, which
        # drops it — an unembeddable finding is one we cannot vouch for.
        scores = np.divide(
            pattern_vecs @ goal_vec,
            norms,
            out=np.zeros(len(patterns), dtype=np.float32),
            where=norms > 0,
        )

        ranked = sorted(zip(patterns, scores), key=lambda pair: float(pair[1]), reverse=True)
        relevant = [p for p, score in ranked if float(score) >= PATTERN_RELEVANCE_FLOOR]

        named = _columns_named(goal, list(columns or []))
        if not named:
            return relevant
        # A stable partition, so embedding order still decides within each half.
        about_named = [p for p in relevant if _mentions_any_column(p, named)]
        return about_named + [p for p in relevant if p not in about_named]

    @staticmethod
    def _column_names(mining_dict: dict[str, Any], ingestion_dict: dict[str, Any]) -> list[str]:
        """The dataset's column names, from whichever upstream agent has them."""
        statistics = mining_dict.get("statistics") or {}
        if statistics:
            return [str(name) for name in statistics]
        return [
            str(column.get("name", ""))
            for column in (ingestion_dict.get("column_summary") or [])
            if isinstance(column, dict)
        ]

    @staticmethod
    def _pick(
        hits: list[dict[str, Any]],
        seen: set[str],
        blocked: set[str],
        floor: float = MIN_CONFIDENCE,
    ) -> dict[str, Any] | None:
        """
        The best hit worth using, or None.

        `floor` is the minimum retrieval score worth using, and differs by pass
        — see `GOAL_LED_MIN_CONFIDENCE`.

        `seen` keeps one answer from citing the same document twice. `blocked`
        is the conversational equivalent — documents shown earlier in this chat
        — and is deliberately not applied to every pass; see
        `_grounded_recommendations`. The caller decides what to do when the two
        together exclude everything.
        """
        for hit in hits:
            if hit["score"] < floor:
                continue
            if hit["source"] in seen or hit["source"] in blocked:
                continue
            seen.add(hit["source"])
            return hit
        return None

    @staticmethod
    def _card(hit: dict[str, Any], pattern: str | None) -> dict[str, Any]:
        """Build one recommendation from a retrieved chunk and, optionally, the
        finding that led to it."""
        title = hit["metadata"].get("title", "")
        technical = _clean_technical(hit["text"])
        analyst = _analyst(hit["text"], title)
        plain = _simplify(hit["text"])

        return {
            "insight": hit["metadata"].get("title", hit["source"]),
            # The finding and the guidance, kept apart.
            #
            # The `text_*` fields below concatenate them, because exports,
            # history persistence and the flattened
            # `AnalysisResponse.recommendations` all read those. But a reader
            # needs to see which half is *their data* and which half is *the
            # methodology*, and once concatenated the frontend cannot tell them
            # apart to style them differently. So both forms travel.
            #
            # A goal-led card has no finding — nothing in the data led to it —
            # and says so with None rather than an empty string, so the card
            # renders a heading-and-guidance layout instead of pretending it
            # found something.
            "finding": pattern,
            "guidance_technical": technical,
            "guidance_analyst": analyst,
            "guidance_plain": plain,
            # Blank line between the finding and the chunk — joining with a
            # plain space would put the chunk's own leading "# Heading"
            # mid-line, where Markdown can't recognize it as a heading anymore.
            "text_technical": f"**{pattern}**\n\n{technical}" if pattern else technical,
            "text_analyst": f"{pattern} {analyst}" if pattern else analyst,
            "text_plain": f"{pattern} {plain}" if pattern else plain,
            "confidence": round(min(max(hit["score"], 0.0), 1.0), 3),
            "sources": [hit["source"]],
        }

    def _grounded_recommendations(
        self,
        context: dict[str, Any],
        patterns: list[str],
        budget: _Budget,
        blocked_sources: set[str],
    ) -> list[dict[str, Any]]:
        """
        Build one turn's recommendations in three passes, in priority order.

        1. **Finding-led** — one card per relevant MiningAgent pattern, grounded
           against the question *and* the finding, capped by the budget.
        2. **Goal-led** — retrieved for the question alone. Nothing from the
           dataset enters this query, which is what makes these cards move when
           the question moves.
        3. **Broad** — the full context query (findings, ingestion warnings and
           this user's prior runs, per Objective 4), filling whatever is left.
           Opening report only, per `_Budget.broad_fill`.

        Pass 3 is unreserved on purpose: it is background grounding, and on a
        turn where the first two passes have plenty to say it should yield.

        Returning fewer than `MAX_RECOMMENDATIONS` — or none at all — is a
        valid outcome for a follow-up. A question the knowledge base cannot
        speak to should be answered with silence the caller can report, not
        with the nearest document to a query the question barely influenced.

        `blocked_sources` — documents already cited earlier in the chat —
        applies to passes 2 and 3 only. Those passes are looking for material
        the reader has not seen, so holding back what they have is exactly
        right. Pass 1 is not: there the document is chosen *for* a specific
        finding, and refusing the best one either drops a relevant finding
        outright or pairs it with a worse document. Both were observed. Asked
        about a correlation the opening report had already cited, the run lost
        its correlation card entirely and led with an unrelated finding
        instead.
        """
        retrieval_goal = self._retrieval_goal(context).strip()
        recommendations: list[dict[str, Any]] = []
        seen: set[str] = set()

        # 1. Finding-led.
        pattern_cap = min(budget.pattern_led, MAX_RECOMMENDATIONS)
        for pattern in patterns:
            if len(recommendations) >= pattern_cap:
                break
            hits = self._vector_store.retrieve(f"{retrieval_goal} {pattern}".strip(), top_k=3)
            hit = self._pick(hits, seen, blocked=set())
            if hit is not None:
                recommendations.append(self._card(hit, pattern))

        # 2. Goal-led. One retrieval, drained until the reserved slots are full.
        goal_cap = min(len(recommendations) + budget.goal_led, MAX_RECOMMENDATIONS)
        if retrieval_goal and len(recommendations) < goal_cap:
            hits = self._vector_store.retrieve(retrieval_goal, top_k=TOP_K_RETRIEVAL)
            while len(recommendations) < goal_cap:
                hit = self._pick(hits, seen, blocked_sources, floor=GOAL_LED_MIN_CONFIDENCE)
                if hit is None:
                    break
                recommendations.append(self._card(hit, None))

        # 3. Broad — opening report only; see `_Budget.broad_fill`.
        query = self._build_query(context)
        if budget.broad_fill and len(recommendations) < MAX_RECOMMENDATIONS and query.strip():
            hits = self._vector_store.retrieve(query, top_k=TOP_K_RETRIEVAL)
            while len(recommendations) < MAX_RECOMMENDATIONS:
                hit = self._pick(hits, seen, blocked_sources)
                if hit is None:
                    break
                recommendations.append(self._card(hit, None))

        return recommendations

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        goal = context.get("goal", "")
        expertise_level = context.get("expertise_level", "intermediate")
        mode = context.get("mode", "rag")
        logger.info("RecommendationAgent.run() | goal=%r expertise=%s mode=%s", goal, expertise_level, mode)

        mining_output = context.get("MiningAgent_output")
        mining_dict = mining_output if isinstance(mining_output, dict) else (
            mining_output.model_dump() if hasattr(mining_output, "model_dump") else {}
        )
        ingestion_output = context.get("IngestionAgent_output")
        ingestion_dict = ingestion_output if isinstance(ingestion_output, dict) else (
            ingestion_output.model_dump() if hasattr(ingestion_output, "model_dump") else {}
        )

        if mode == "llm":
            return self._llm_recommend(goal, expertise_level, mining_dict, ingestion_dict)

        self._ensure_kb_loaded()

        # 1. Direct answer for specific factual questions — no RAG dump.
        #
        # The dataframe travels too, for the one handler that answers questions
        # about the table's *contents* rather than its shape ("tell me about
        # Drake"). Nothing upstream can answer those — every statistic
        # describes the shape — and the knowledge base holds methodology, so
        # without this the only honest response is that there isn't one.
        qa_answer = self._qa_agent.try_answer(
            goal, mining_dict, ingestion_dict, dataframe=context.get("dataframe")
        )
        if qa_answer is not None:
            sources = [qa_answer.source] if qa_answer.source else []
            return {
                "recommendations": [
                    {
                        "insight": "Computed from your data",
                        # A direct factual answer is a computed number — there
                        # is no methodology excerpt to expand or compress, so
                        # all three registers are deliberately identical here.
                        # Paraphrasing a statistic per audience would change
                        # the wording without changing the information, which
                        # is presentation, not adaptation.
                        "text_technical": qa_answer.text,
                        "text_analyst": qa_answer.text,
                        "text_plain": qa_answer.text,
                        "confidence": 1.0,
                        "sources": sources,
                    }
                ],
                "rag_sources": sources,
                "prior_runs": self._prior_runs(context),
                "message": "Answered directly from computed statistics.",
            }

        # 2. Grounded recommendations, split between what the data found and
        #    what was asked. The orchestrator's post-Mining reflection
        #    (OrchestratorAgent._reflect_on_mining) can adjust the finding list
        #    before it is used: a target-completeness warning is prepended so
        #    it leads (patterns are processed in order, and this still grounds
        #    naturally via RAG, e.g. against missing_values.md), and the
        #    attribution pattern is dropped entirely when the fitted model
        #    scored too low to trust — already flagged in the ReAct log, so it
        #    should not also be presented as a top finding here.
        directives = context.get("directives", {}) or {}
        patterns = list(mining_dict.get("patterns") or [])
        if directives.get("attribution_trusted") is False:
            patterns = [p for p in patterns if ATTRIBUTION_PATTERN_MARKER not in p]
        target_quality_warning = directives.get("target_quality_warning")

        turns = conversation_state.normalize(context.get("conversation"))
        budget = FOLLOWUP_BUDGET if turns else INITIAL_BUDGET

        # Rank findings against the question, but only mid-conversation. On the
        # opening run the goal is usually broad ("analyse this dataset") and
        # every finding is equally on-topic, so reordering would only trade
        # MiningAgent's deliberate order for embedding noise.
        if turns:
            patterns = self._patterns_for_goal(
                self._retrieval_goal(context), patterns, self._column_names(mining_dict, ingestion_dict)
            )

        # Pinned after ranking: a target-quality warning is an orchestrator
        # directive about whether the analysis can be trusted at all, which
        # outranks the question being asked.
        if target_quality_warning:
            patterns.insert(0, target_quality_warning)

        if not patterns and not self._build_query(context).strip():
            return {
                "recommendations": [],
                "rag_sources": [],
                "prior_runs": self._prior_runs(context),
                "message": "No goal or upstream findings to ground recommendations in.",
            }

        # Documents the user has already been shown in this chat, held back
        # from the passes that look for fresh material — but only as a
        # preference: if excluding them leaves nothing at all to say, a repeat
        # beats an empty answer, so the turn is rebuilt without the
        # restriction.
        already_cited = conversation_state.previously_cited(turns)
        recommendations = self._grounded_recommendations(context, patterns, budget, already_cited)
        if not recommendations and already_cited:
            logger.info("Every grounding document was already cited this chat; allowing repeats.")
            recommendations = self._grounded_recommendations(context, patterns, budget, set())

        rag_sources = sorted({s for rec in recommendations for s in rec["sources"]})

        return {
            "recommendations": recommendations,
            "rag_sources": rag_sources,
            # Surfaced for display, deliberately separate from `rag_sources`:
            # a prior run is context the user can recognise, not retrievable
            # methodology, and FR-03's citation promise covers only the latter.
            "prior_runs": self._prior_runs(context),
            "message": f"Generated {len(recommendations)} RAG-grounded recommendation(s).",
        }

    # ── LLM mode ─────────────────────────────────────────────────────────

    def _llm_recommend(
        self,
        goal: str,
        expertise_level: str,
        mining_dict: dict[str, Any],
        ingestion_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Freeform response from Gemini, grounded only in already-computed
        aggregate stats (never raw rows). Never raises — a missing key or a
        failed request degrades to a single explanatory recommendation
        instead of breaking the whole pipeline step."""
        if not self._llm_client.is_configured:
            text = (
                "LLM mode isn't set up yet — it needs a Gemini API key configured "
                "on the server (GEMINI_API_KEY). Switch back to RAG mode for now."
            )
            return {
                "recommendations": [
                    {
                        "insight": "LLM mode unavailable",
                        "text_technical": text,
                        "text_plain": text,
                        "confidence": 0.0,
                        "sources": [],
                    }
                ],
                "rag_sources": [],
                "message": "LLM mode not configured.",
            }

        prompt = self._build_llm_prompt(goal, expertise_level, mining_dict, ingestion_dict)
        try:
            text = self._llm_client.generate(prompt)
        except LLMError as exc:
            logger.warning("LLM recommendation failed: %s", exc)
            text = f"The LLM request failed ({exc}). Try again, or switch to RAG mode."

        return {
            "recommendations": [
                {
                    "insight": "LLM-generated response",
                    "text_technical": text,
                    "text_plain": text,
                    "confidence": 1.0,
                    "sources": [],
                }
            ],
            "rag_sources": [],
            "message": "Generated via Gemini (not grounded in the knowledge base).",
        }

    def _build_llm_prompt(
        self,
        goal: str,
        expertise_level: str,
        mining_dict: dict[str, Any],
        ingestion_dict: dict[str, Any],
    ) -> str:
        """Compose a prompt from already-computed aggregate statistics —
        never raw rows, since this leaves the server as part of the LLM
        call."""
        statistics = mining_dict.get("statistics") or {}
        data_quality = mining_dict.get("data_quality") or {}
        patterns = mining_dict.get("patterns") or []
        correlations = mining_dict.get("correlations") or {}
        warnings = ingestion_dict.get("warnings") or []

        column_lines = []
        for name, stat in statistics.items():
            dq = data_quality.get(name, {})
            missing_pct = round(100 - dq.get("completeness_pct", 100), 1)
            if stat.get("type") == "numeric":
                column_lines.append(
                    f"- {name} (numeric): mean={stat.get('mean')}, min={stat.get('min')}, "
                    f"max={stat.get('max')}, {missing_pct}% missing"
                )
            else:
                column_lines.append(f"- {name} (categorical): {missing_pct}% missing")

        top_correlations = []
        seen_pairs: set[frozenset[str]] = set()
        for col_a, row in correlations.items():
            for col_b, r in row.items():
                if col_a == col_b or r is None:
                    continue
                pair = frozenset((col_a, col_b))
                if pair in seen_pairs or abs(r) < 0.5:
                    continue
                seen_pairs.add(pair)
                top_correlations.append(f"- {col_a} vs {col_b}: r={r:.2f}")

        sections = [
            "You are a data analysis assistant helping a user explore a dataset they've uploaded.",
            f"\nColumns:\n" + ("\n".join(column_lines) or "(no column profile available)"),
        ]
        if patterns:
            sections.append("\nNotable patterns already detected:\n" + "\n".join(f"- {p}" for p in patterns))
        if top_correlations:
            sections.append("\nStrong correlations (|r| >= 0.5):\n" + "\n".join(top_correlations))
        if warnings:
            sections.append("\nData quality warnings:\n" + "\n".join(f"- {w}" for w in warnings))
        sections.append(
            f'\nUser\'s question: "{goal}"\n\n'
            f"Answer conversationally, referencing the specific numbers above where relevant. "
            f"Calibrate depth for a {expertise_level} audience. Keep it under ~200 words."
        )
        return "\n".join(sections)
