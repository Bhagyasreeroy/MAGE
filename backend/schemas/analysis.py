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

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExpertiseLevel(str, Enum):
    """User-declared analytical expertise level."""

    beginner = "beginner"
    intermediate = "intermediate"
    expert = "expert"


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


class AnalysisRequest(BaseModel):
    """Payload sent by the frontend to trigger an EDA pipeline run."""

    goal: str = Field(
        ...,
        min_length=5,
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


class KnowledgeSource(BaseModel):
    """A single document in the RAG knowledge base."""

    source: str = Field(..., description="Relative source path, e.g. 'knowledge_base/missing_values.md'.")
    title: str = Field(..., description="Document title.")
    doc_type: str = Field(default="", description="Document type/category, e.g. 'methodology'.")
    chunk_count: int = Field(..., description="Number of chunks this document was split into.")


class AnalysisResponse(BaseModel):
    """Full EDA pipeline response returned to the client."""

    goal: str = Field(..., description="Echo of the original analytical goal.")
    expertise_level: ExpertiseLevel
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
    rag_sources: list[str] = Field(
        default_factory=list,
        description="Knowledge-base sources used by the RAG retrieval step.",
    )
    summary: str = Field(
        default="",
        description="High-level summary of the analysis.",
    )
    analysis_id: str | None = Field(
        default=None,
        description=(
            "Id referencing the uploaded dataset for this session. Pass it back "
            "on a follow-up /analysis/run call (instead of re-uploading a file) "
            "to keep querying the same dataset."
        ),
    )
