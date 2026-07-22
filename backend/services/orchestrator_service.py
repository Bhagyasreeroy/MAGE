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

from backend.core.dataset_store import dataset_store
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

    async def run(
        self,
        request: AnalysisRequest,
        file: UploadFile | None = None,
        analysis_id: str | None = None,
    ) -> AnalysisResponse:
        """Orchestrate a full MAGE pipeline run for the given request.

        If `file` is given, it's ingested and also cached under a new
        analysis_id so a later call can pass that id instead of the file
        to keep querying the same dataset. If only `analysis_id` is given,
        the previously-uploaded file is looked up and reused.
        """
        data: dict[str, Any] = dict(request.metadata)
        resolved_id = analysis_id

        if file is not None:
            content = await file.read()
            resolved_id = dataset_store.put(file.filename or "dataset", content)
            data["source"] = dataset_store.get(resolved_id)
        elif analysis_id is not None:
            stored = dataset_store.get(analysis_id)
            if stored is not None:
                data["source"] = stored

        logger.info(
            "Starting analysis | goal=%r expertise=%s analysis_id=%s has_source=%s",
            request.goal,
            request.expertise_level,
            resolved_id,
            "source" in data,
        )

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
            analysis_id=resolved_id,
        )
