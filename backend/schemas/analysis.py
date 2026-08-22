"""
schemas/analysis.py
────────────────────
Pydantic v2 request / response models for the analysis endpoints.

Models:
    ExpertiseLevel     — Enum for user expertise (beginner / intermediate / expert)
    AnalysisRequest    — Incoming payload: goal + expertise level
    StepResult         — A single agent step result (name + output)
    AnalysisResponse   — Full pipeline response returned to the client
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ExpertiseLevel(str, Enum):
    """User-declared analytical expertise level."""

    beginner = "beginner"
    intermediate = "intermediate"
    expert = "expert"


class RecommendationMode(str, Enum):
    """How RecommendationAgent produces its response.

    "rag" (default) is the original, grounded/cited path — deterministic,
    every claim traces to a knowledge-base source. "llm" opts into a
    freeform Gemini response reasoning over the same computed features,
    with no citations (see agents/recommendation_agent.py).
    """

    rag = "rag"
    llm = "llm"


class TaskType(str, Enum):
    """
    Analytical task type inferred from the user's natural-language goal.

    The orchestrator uses this to build a *conditional* analysis pipeline —
    the task type decides which computations are prioritized and run, not
    merely which results are surfaced (Module 2, FR-02).
    """

    classification = "classification"
    regression = "regression"
    clustering = "clustering"
    anomaly_detection = "anomaly_detection"
    reporting = "reporting"


class GoalClassification(BaseModel):
    """Result of classifying a natural-language goal into a task type."""

    task_type: TaskType = Field(
        ...,
        description="Inferred analytical task type driving the conditional pipeline.",
    )
    target_column: str | None = Field(
        default=None,
        description="Best-guess target/label column, when the goal implies one.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classifier confidence in the inferred task type (0-1).",
    )
    rationale: str = Field(
        default="",
        description="Short human-readable justification for the classification.",
    )


class ReActStep(BaseModel):
    """
    A single Reason-Act-Observe step in the orchestrator loop.

    Field names mirror the `agent_steps` table in the system design so the
    step log is the complete, inspectable explainability trail (FR-06).
    """

    agent_name: str = Field(..., description="Specialist agent invoked in this step.")
    action: str = Field(..., description="The action taken (e.g. 'run mining with directives').")
    reasoning: str = Field(default="", description="Why the orchestrator chose this step now.")
    observation: str = Field(default="", description="Short summary of what the step produced.")
    status: str = Field(..., description="Step execution status ('success' | 'error' | 'skipped').")
    latency_ms: int = Field(default=0, ge=0, description="Wall-clock duration of the step in ms.")
    output: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured agent output payload for downstream consumers.",
    )


# Floor on how much of a question a goal has to be. A first request is
# starting from nothing, so it has to say what it wants. A follow-up is not:
# "why?" is a complete question when there is a previous turn to attach it to,
# and rejecting it would refuse the very shape of question that conversational
# follow-up exists to support.
MIN_GOAL_CHARS = 5
MIN_FOLLOWUP_GOAL_CHARS = 2


class ConversationTurn(BaseModel):
    """
    One prior turn of the follow-up chat, replayed by the client.

    The pipeline is stateless — every chat message is a fresh run — so the
    client sends back what has been said so far. Without it a follow-up like
    "why?" has no subject, and nothing can tell that the answer about to be
    built is the one the user already read.

    `sources` is what makes the second of those possible: an assistant turn
    records which knowledge-base documents it cited, so the next turn can
    prefer material the user has not seen. Empty for user turns.
    """

    role: Literal["user", "assistant"] = Field(..., description="Who produced this turn.")
    content: str = Field(
        ...,
        max_length=4000,
        description="What was said. Assistant turns may be truncated by the client.",
    )
    sources: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Knowledge-base documents this turn cited (assistant turns only).",
    )


class AnalysisRequest(BaseModel):
    """Payload sent by the frontend to trigger an EDA pipeline run."""

    goal: str = Field(
        ...,
        max_length=2000,
        description="Natural-language analytical goal (e.g. 'Find anomalies in sales data').",
        examples=["Identify the top factors driving customer churn."],
    )
    expertise_level: ExpertiseLevel = Field(
        default=ExpertiseLevel.intermediate,
        description="User's analytical expertise level — adapts recommendation verbosity.",
    )
    dataset_name: str | None = Field(
        default=None,
        description="Optional name / identifier of the uploaded dataset.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata forwarded to the pipeline.",
    )
    mode: RecommendationMode = Field(
        default=RecommendationMode.rag,
        description="'rag' (grounded/cited, default) or 'llm' (freeform Gemini response).",
    )
    conversation: list[ConversationTurn] = Field(
        default_factory=list,
        max_length=32,
        description=(
            "Prior turns of this chat, oldest first. Supplied on follow-up "
            "requests so an elliptical question can be resolved against what "
            "came before, and so the answer avoids repeating what was already "
            "shown. Absent on the first request of a conversation."
        ),
    )

    @model_validator(mode="after")
    def _goal_is_long_enough(self) -> "AnalysisRequest":
        """
        Enforce the goal floor, which depends on whether this is a follow-up.

        A validator rather than `min_length` on the field, because the limit is
        not a property of the field alone — it is a property of the request.
        See `MIN_GOAL_CHARS`.
        """
        minimum = MIN_FOLLOWUP_GOAL_CHARS if self.conversation else MIN_GOAL_CHARS
        if len(self.goal.strip()) < minimum:
            raise ValueError(
                f"`goal` must be at least {minimum} characters"
                + ("." if self.conversation else " on the first request of a conversation.")
            )
        return self


class StepResult(BaseModel):
    """Output from a single specialist agent step."""

    agent: str = Field(..., description="Agent name (e.g. 'IngestionAgent').")
    status: str = Field(..., description="Step execution status.")
    output: dict[str, Any] = Field(default_factory=dict, description="Agent output payload.")


class ColumnStats(BaseModel):
    """Column-level statistics produced during ingestion profiling."""

    min: float | None = Field(default=None, description="Minimum numeric value.")
    max: float | None = Field(default=None, description="Maximum numeric value.")
    mean: float | None = Field(default=None, description="Mean numeric value.")
    unique_count: int | None = Field(
        default=None,
        description="Unique-value count for categorical/object columns.",
    )


class ColumnSummary(BaseModel):
    """Summary for a single ingested column."""

    name: str = Field(..., description="Column name.")
    dtype: str = Field(..., description="Pandas dtype string.")
    missing_count: int = Field(..., description="Number of missing values in the column.")
    stats: ColumnStats | None = Field(
        default=None,
        description="Optional profiling stats for the column.",
    )


class IngestionResult(BaseModel):
    """Structured ingestion and profiling output for tabular datasets."""

    row_count: int = Field(..., description="Number of rows in the ingested dataset.")
    column_count: int = Field(..., description="Number of columns in the ingested dataset.")
    column_summary: list[ColumnSummary] = Field(
        default_factory=list,
        description="Per-column summary data.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal ingestion/profile warnings.",
    )
    dataset_id: str | None = Field(
        default=None,
        description="Id of the persisted dataset, for reuse in a later /analysis/run call.",
    )


class SampleDataset(BaseModel):
    """A bundled demo dataset in data/samples/, selectable from New Analysis
    without the user needing the file on their own machine."""

    filename: str = Field(..., description="File name within data/samples/, used to request loading it.")
    title: str = Field(..., description="Human-readable name shown in the picker.")
    description: str = Field(..., description="What the dataset demonstrates (clusters, outliers, etc.).")
    size_kb: float = Field(..., description="File size in kilobytes.")


class KnowledgeSource(BaseModel):
    """A single document in the RAG knowledge base."""

    source: str = Field(..., description="Relative source path, e.g. 'knowledge_base/missing_values.md'.")
    title: str = Field(..., description="Document title.")
    doc_type: str = Field(default="", description="Document type/category, e.g. 'methodology'.")
    chunk_count: int = Field(..., description="Number of chunks this document was split into.")


class RecommendationCard(BaseModel):
    """
    One recommendation with its parts separated.

    A grounded recommendation is two things joined: a finding computed from the
    user's data, and the methodology the knowledge base offers about it. Once
    concatenated into a single string the frontend cannot tell them apart, so
    it cannot lay them out differently — which made a cited, grounded answer
    render as a paragraph of run-on prose beside the LLM's structured one.
    """

    insight: str = Field(
        default="",
        description="Title of the knowledge-base document this is grounded in.",
    )
    finding: str | None = Field(
        default=None,
        description=(
            "The finding computed from the user's data that led here, or None "
            "when the recommendation came from goal-only retrieval and there "
            "is nothing to lead with."
        ),
    )
    guidance: str = Field(
        default="",
        description="The methodology prose, in the reader's FR-04 register.",
    )
    confidence: float = Field(
        default=0.0,
        description="Retrieval confidence for the grounding chunk, 0-1.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Knowledge-base file(s) this recommendation cites.",
    )


class AnalysisResponse(BaseModel):
    """Full EDA pipeline response returned to the client."""

    goal: str = Field(..., description="Echo of the original analytical goal.")
    expertise_level: ExpertiseLevel
    mode: RecommendationMode = Field(
        default=RecommendationMode.rag,
        description="Which path produced these recommendations — 'rag' or 'llm'.",
    )
    task_type: TaskType | None = Field(
        default=None,
        description="Task type inferred from the goal, driving the conditional pipeline.",
    )
    classification: GoalClassification | None = Field(
        default=None,
        description="Full goal-classification detail (task type, target, confidence, rationale).",
    )
    steps: list[ReActStep] = Field(
        default_factory=list,
        description="Ordered Reason-Act-Observe step log — the explainability trail.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="RAG-grounded, expertise-adapted EDA recommendations.",
    )
    recommendation_cards: list[RecommendationCard] = Field(
        default_factory=list,
        description=(
            "The same recommendations with their parts kept apart, so a reader "
            "can see which half came from their data and which from the "
            "knowledge base. `recommendations` above stays the flattened form."
        ),
    )
    rag_sources: list[str] = Field(
        default_factory=list,
        description="Knowledge-base sources used by the RAG retrieval step.",
    )
    summary: str = Field(
        default="",
        description="High-level summary of the analysis.",
    )
    dataset_id: str | None = Field(
        default=None,
        description=(
            "Id referencing the persisted, uploaded dataset. Pass it back on a "
            "follow-up /analysis/run call (instead of re-uploading a file) to "
            "keep querying the same dataset."
        ),
    )
    run_id: str | None = Field(
        default=None,
        description="Id of the persisted analysis run, for GET /analysis/history/{run_id}.",
    )


class AnalysisRunSummary(BaseModel):
    """A lightweight entry in a user's analysis history list."""

    id: str
    goal: str
    expertise_level: ExpertiseLevel
    status: str
    summary: str
    dataset_id: str | None
    created_at: datetime


class ShareStatus(BaseModel):
    """Whether a conversation (a run and everything it's a follow-up to,
    or that follows up on it) is publicly shared, and the id to share."""

    is_shared: bool
    share_id: str = Field(..., description="The conversation's root run id — also its public share id.")


class DatasetSummary(BaseModel):
    """A lightweight entry in a user's dataset list."""

    id: str
    filename: str
    row_count: int | None
    column_count: int | None
    created_at: datetime
    expires_at: datetime | None = Field(
        default=None,
        description="When retention collects this dataset (NFR-04). Null means never.",
    )
    root_id: str = Field(default="", description="Lineage root id — shared by every version of this dataset.")
    parent_id: str | None = Field(default=None, description="The version this one was transformed from, if any.")
    version: int = Field(default=1, description="1 for an original upload; increments per transform.")
    transform_type: str | None = Field(
        default=None, description="'clean' | 'query_save' | None (an original upload)."
    )


class DatasetDetail(DatasetSummary):
    """Full detail for a single dataset version — backs the workbench page."""

    column_summary: list[ColumnSummary] = Field(default_factory=list)
    transform_params: dict[str, Any] | None = Field(
        default=None, description="{'ops': [...]} for 'clean', {'sql': '...'} for 'query_save'."
    )
    report: list[str] | None = Field(
        default=None,
        description="Human-readable summary of what changed — only present on a freshly-created version.",
    )


class PurgeResult(BaseModel):
    """Outcome of a retention sweep (NFR-04)."""

    purged: int = Field(..., description="How many expired datasets were deleted.")


class DatasetPreview(BaseModel):
    """A page of rows from a dataset — backs the spreadsheet grid and the
    query console's result table."""

    columns: list[str] = Field(..., description="Column names, in order.")
    dtypes: list[str] = Field(..., description="Pandas dtype string per column, same order as columns.")
    rows: list[list[Any]] = Field(..., description="Row values, each inner list ordered like columns.")
    total_rows: int = Field(..., description="Total row count of the full dataset (not just this page).")
    offset: int = Field(default=0)
    limit: int = Field(default=50)


class TransformOpRequest(BaseModel):
    """One staged operation — see data_pipeline/processing.py for the
    supported 'type' values and their params."""

    model_config = {"extra": "allow"}

    type: str = Field(..., description="Op type, e.g. 'drop_columns', 'fill_missing', 'edit_cells'.")


class TransformRequest(BaseModel):
    """Body for POST /analysis/datasets/{id}/transform."""

    ops: list[TransformOpRequest] = Field(..., min_length=1)


class QueryRequest(BaseModel):
    """Body for POST /analysis/datasets/{id}/query and .../query/save."""

    sql: str = Field(..., min_length=1, max_length=10000)


class NLQueryRequest(BaseModel):
    """Body for POST /analysis/datasets/{id}/query/nl — a plain-English
    question, translated to SQL by Gemini before running through the same
    validated/sandboxed path as hand-typed SQL."""

    question: str = Field(..., min_length=1, max_length=2000)


class NLQueryResult(BaseModel):
    """Response for the ask-in-English endpoint — the translated SQL is
    always returned alongside the results, so it's never a black box."""

    sql: str = Field(..., description="The SQL Gemini generated from the question.")
    preview: DatasetPreview


class ExplainRequest(BaseModel):
    """Body for POST /analysis/explain — a specific, already-computed
    finding (a recommendation's own text, a feature-importance result)
    the user wants a deeper, RAG-grounded explanation of."""

    finding: str = Field(..., min_length=1, max_length=1000)
    goal: str = Field(default="", max_length=2000)


class ExplainResult(BaseModel):
    """Always grounded — synthesized=False means the LLM wasn't used
    (unconfigured, failed, or nothing to add) and this is the same raw
    retrieved excerpt RecommendationAgent has always surfaced."""

    explanation: str
    sources: list[str]
    synthesized: bool = Field(
        ..., description="True if the LLM synthesized across multiple sources; False if this is a raw excerpt."
    )
