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

from backend.services import analysis_run_service, dataset_service, run_memory_service
from backend.schemas.analysis import AnalysisRequest, AnalysisResponse
from agents.orchestrator import OrchestratorAgent
from agents.planner import INGESTION, MINING

logger = logging.getLogger(__name__)


class MissingDataSourceError(ValueError):
    """
    Raised when a run has no dataset to analyse.

    Its own type, not a bare ValueError, because the routers have to tell it
    apart from a genuine pipeline failure: this one is the caller's mistake
    (422 / a WebSocket bad-request close), everything else is ours (500).
    """


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

    @staticmethod
    async def _remember(
        db: AsyncSession,
        user_id: str,
        goal: str,
        raw_result: dict[str, Any],
        steps: list[dict[str, Any]],
    ) -> None:
        """
        Record this run in run memory, never letting that failure surface.

        The dataset is fingerprinted from the ingested schema rather than the
        uploaded file, so two uploads of the same table are recognised as the
        same data. `run_memory_service.record` already swallows its own
        errors; the extra guard here covers the shape-extraction above it.
        """
        try:
            mining = next(
                (s["output"] for s in steps if s["agent_name"] == MINING), {}
            ) or {}
            ingestion = next(
                (s["output"] for s in steps if s["agent_name"] == INGESTION), {}
            ) or {}
            columns = [
                str(c.get("name", ""))
                for c in (ingestion.get("column_summary") or [])
                if isinstance(c, dict)
            ]
            await run_memory_service.record(
                db,
                user_id=user_id,
                goal=goal,
                task_type=str(raw_result.get("task_type") or ""),
                dataset_fingerprint=run_memory_service.dataset_fingerprint(
                    columns, int(ingestion.get("row_count") or 0)
                ),
                findings=[str(p) for p in (mining.get("patterns") or [])],
            )
        except Exception:  # noqa: BLE001 - memory is additive, never fatal
            logger.warning("Could not record run memory for this run.", exc_info=True)

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

        # No resolvable dataset → refuse, rather than run the pipeline anyway.
        #
        # Without this the run completes and returns a full report. Ingestion
        # fails, Mining and Visualization report *success* at having skipped
        # (they return empty results rather than raising), and
        # RecommendationAgent grounds against the knowledge base using the goal
        # text alone — a path it needs, since the follow-up chat asks
        # dataset-free methodology questions. The output is five real, cited
        # recommendations about the goal's topic, presented as though they were
        # analysis of the user's data. Compare `goal`, whose absence FastAPI
        # already rejects with a 422; this is the same class of missing input.
        #
        # One check covers both ways of arriving here: no file and no
        # dataset_id, and a dataset_id that resolved to nothing (a wrong id, or
        # another user's). The second case was worse than a misleading report —
        # `resolved_dataset_id` kept the unresolvable id and persisting the run
        # hit a foreign-key violation, a 500 from a request that should have
        # been a 422.
        if "source" not in data:
            raise MissingDataSourceError(
                "This run has no dataset to analyse. Upload a file or pass the "
                "dataset_id of one you already uploaded."
                if dataset_id is None
                else f"No dataset found for dataset_id {dataset_id!r}. "
                "Upload a file, or pass the id of a dataset you own."
            )

        logger.info(
            "Starting analysis | goal=%r expertise=%s dataset_id=%s has_source=%s",
            request.goal,
            request.expertise_level,
            resolved_dataset_id,
            "source" in data,
        )

        # Objective 4 — ground this run in the user's own analytical history as
        # well as the knowledge base. Retrieved before the pipeline runs so the
        # RecommendationAgent can use it; scoped to this user by the service.
        data["prior_runs"] = await run_memory_service.retrieve_similar(
            db, user_id=user_id, goal=request.goal
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
            mode=request.mode.value,
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

        # Objective 4 — remember this run so later comparable questions can be
        # grounded in it. Derived material only: the goal, the task type, a
        # hashed dataset fingerprint, and the headline findings.
        await self._remember(db, user_id, request.goal, raw_result, steps)

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
            mode=request.mode,
            task_type=raw_result.get("task_type"),
            classification=raw_result.get("classification"),
            steps=steps,
            recommendations=recommendations,
            rag_sources=rag_sources,
            summary=summary,
            dataset_id=resolved_dataset_id,
            run_id=run.id,
        )
