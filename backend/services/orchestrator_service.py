"""
services/orchestrator_service.py
─────────────────────────────────
Thin service layer that bridges the FastAPI router with the
OrchestratorAgent.  Handles async / sync boundary and maps
domain models to/from Pydantic schemas.
"""

from __future__ import annotations

import logging
import sys
import os

# Allow importing from the monorepo root when running via uvicorn from /backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Any

from fastapi import UploadFile

from backend.schemas.analysis import AnalysisRequest, AnalysisResponse
from agents.orchestrator import OrchestratorAgent

logger = logging.getLogger(__name__)


class OrchestratorService:
    """
    Bridges the HTTP layer with the OrchestratorAgent.

    Responsibilities:
    - Validate/transform the incoming AnalysisRequest.
    - Invoke OrchestratorAgent.run() (synchronously for now; async in later milestone).
    - Map the raw agent output back to an AnalysisResponse.
    """

    def __init__(self) -> None:
        self._agent = OrchestratorAgent()

    async def run(self, request: AnalysisRequest, file: UploadFile | None = None) -> AnalysisResponse:
        """Orchestrate a full MAGE pipeline run for the given request."""
        logger.info(
            "Starting analysis | goal=%r expertise=%s file=%s",
            request.goal,
            request.expertise_level,
            file.filename if file else None,
        )

        data: dict[str, Any] = dict(request.metadata)
        if file is not None:
            data["source"] = file

        raw_result = self._agent.run(
            goal=request.goal,
            expertise_level=request.expertise_level.value,
            data=data,
        )

        return AnalysisResponse(
            goal=request.goal,
            expertise_level=request.expertise_level,
            task_type=raw_result.get("task_type"),
            classification=raw_result.get("classification"),
            steps=raw_result.get("steps", []),
            recommendations=raw_result.get("recommendations", []),
            rag_sources=raw_result.get("rag_sources", []),
            summary=raw_result.get("summary", ""),
        )
