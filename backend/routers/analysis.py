"""
Analysis endpoints - run the goal-conditioned pipeline and ingest tabular files.

Endpoints:
    POST /analysis/run     - trigger a full MAGE analysis pipeline run
    POST /analysis/ingest  - upload a CSV or XLSX file and profile it
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from agents.ingestion_agent import IngestionAgent
from backend.schemas.analysis import AnalysisRequest, AnalysisResponse, IngestionResult
from backend.services.orchestrator_service import OrchestratorService
from data_pipeline.ingestion import IngestionError

logger = logging.getLogger(__name__)
router = APIRouter()

_orchestrator_service = OrchestratorService()
_ingestion_agent = IngestionAgent()


@router.post(
    "/run",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a goal-conditioned EDA pipeline",
)
async def run_analysis(request: AnalysisRequest) -> AnalysisResponse:
    """
    Trigger the full MAGE pipeline for a given analytical goal.

    - Accepts a natural-language goal and a user expertise level.
    - Delegates to the OrchestratorAgent which runs the ReAct loop.
    - Returns structured EDA recommendations grounded in the RAG layer.

    Note: this route remains on the orchestrator flow; ingestion is exposed
    separately through POST /analysis/ingest.
    """
    try:
        result = await _orchestrator_service.run(request)
        return result
    except Exception as exc:
        logger.exception("Analysis pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {exc}",
        ) from exc


@router.post(
    "/ingest",
    response_model=IngestionResult,
    status_code=status.HTTP_200_OK,
    summary="Ingest and profile a tabular file",
)
async def ingest_file(file: UploadFile = File(...)) -> IngestionResult:
    """Upload a CSV or XLSX file and return a structured ingestion profile."""
    try:
        return _ingestion_agent.run(source=file)
    except IngestionError as exc:
        logger.info("Ingestion failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected ingestion failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc
