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


class AnalysisResponse(BaseModel):
    """Full EDA pipeline response returned to the client."""

    goal: str = Field(..., description="Echo of the original analytical goal.")
    expertise_level: ExpertiseLevel
    steps: list[StepResult] = Field(
        default_factory=list,
        description="Ordered results from each specialist agent.",
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
