"""
services/orchestrator_service.py
─────────────────────────────────
Thin service layer that bridges the FastAPI router with the
OrchestratorAgent.  Handles async / sync boundary, dataset/run
persistence, and maps domain models to/from Pydantic schemas.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import os

# Allow importing from the monorepo root when running via uvicorn from /backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from collections.abc import Callable
from typing import Any

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services import analysis_run_service, dataset_service
from backend.schemas.analysis import AnalysisRequest, AnalysisResponse
from agents.orchestrator import OrchestratorAgent

logger = logging.getLogger(__name__)


class OrchestratorService:
    """
    Bridges the HTTP layer with the OrchestratorAgent.

    Responsibilities:
    - Validate/transform the incoming AnalysisRequest.
    - Invoke OrchestratorAgent.run() on a worker thread.
    - Persist the uploaded dataset and the completed run, scoped to the user.
    - Map the raw agent output back to an AnalysisResponse.
    """

    def __init__(self) -> None:
        self._agent = OrchestratorAgent()

    async def run(
        self,
        request: AnalysisRequest,
        db: AsyncSession,
        user_id: str,
        file: UploadFile | None = None,
        dataset_id: str | None = None,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> AnalysisResponse:
        """Orchestrate a full MAGE pipeline run for the given request.

        If `file` is given, it's ingested and persisted as a new Dataset
        row so a later call can pass its id instead of the file to keep
        querying the same dataset. If only `dataset_id` is given, the
        previously-uploaded file is looked up (scoped to `user_id`) and
        reused.

        `on_step`, when supplied, is forwarded to the agent and invoked once
        per completed pipeline step — this is what the WebSocket endpoint uses
        to stream the Reason/Act/Observe trail live. **It is called on the
        worker thread, not the event loop**, so an async caller must marshal
        back itself (e.g. via ``loop.call_soon_threadsafe``).
        """
        data: dict[str, Any] = dict(request.metadata)
        resolved_dataset_id = dataset_id

        if file is not None:
            content = await file.read()
            dataset = await dataset_service.save_dataset(db, user_id, file.filename or "dataset", content)
            resolved_dataset_id = dataset.id
            data["source"] = dataset_service.as_stored_file(dataset)
        elif dataset_id is not None:
            dataset = await dataset_service.get_dataset(db, user_id, dataset_id)
            if dataset is not None:
                data["source"] = dataset_service.as_stored_file(dataset)

        logger.info(
            "Starting analysis | goal=%r expertise=%s dataset_id=%s has_source=%s",
            request.goal,
            request.expertise_level,
            resolved_dataset_id,
            "source" in data,
        )

        # The agent is synchronous and CPU-bound (pandas / sklearn). Run it on a
        # worker thread rather than inline: on the event loop it would block
        # every other request for the duration of the analysis, and streaming
        # would be impossible — `on_step` would fire, but nothing could be sent
        # over the socket until the whole run had already finished.
        raw_result = await asyncio.to_thread(
            self._agent.run,
            goal=request.goal,
            expertise_level=request.expertise_level.value,
            data=data,
            on_step=on_step,
        )

        steps = raw_result.get("steps", [])

        if file is not None and resolved_dataset_id is not None:
            ingestion_output = next(
                (s["output"] for s in steps if s["agent_name"] == "IngestionAgent" and s["status"] == "success"),
                None,
            )
            if ingestion_output is not None:
                await dataset_service.update_dataset_stats(
                    db,
                    resolved_dataset_id,
                    row_count=ingestion_output.get("row_count"),
                    column_count=ingestion_output.get("column_count"),
                )

        recommendations = raw_result.get("recommendations", [])
        rag_sources = raw_result.get("rag_sources", [])
        summary = raw_result.get("summary", "")

        run = await analysis_run_service.save_run(
            db,
            user_id=user_id,
            goal=request.goal,
            expertise_level=request.expertise_level.value,
            steps=steps,
            recommendations=recommendations,
            rag_sources=rag_sources,
            summary=summary,
            dataset_id=resolved_dataset_id,
        )

        return AnalysisResponse(
            goal=request.goal,
            expertise_level=request.expertise_level,
            task_type=raw_result.get("task_type"),
            classification=raw_result.get("classification"),
            steps=steps,
            recommendations=recommendations,
            rag_sources=rag_sources,
            summary=summary,
            dataset_id=resolved_dataset_id,
            run_id=run.id,
        )
