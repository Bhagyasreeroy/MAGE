"""
Analysis endpoints - run the goal-conditioned pipeline and ingest tabular files.

Endpoints:
    POST /analysis/run     - trigger a full MAGE analysis pipeline run
    POST /analysis/ingest  - upload a CSV or XLSX file and profile it
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from agents.ingestion_agent import IngestionAgent
from backend.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    ExpertiseLevel,
    IngestionResult,
    KnowledgeSource,
)
from backend.services.orchestrator_service import OrchestratorService
from data_pipeline.ingestion import IngestionError
from rag.knowledge_loader import KnowledgeBaseLoader

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
async def run_analysis(
    goal: str = Form(...),
    expertise_level: ExpertiseLevel = Form(ExpertiseLevel.intermediate),
    file: UploadFile | None = File(None),
    analysis_id: str | None = Form(None),
) -> AnalysisResponse:
    """
    Trigger the full MAGE pipeline for a given analytical goal.

    - Accepts a natural-language goal, a user expertise level, and either
      a dataset file (first request) or an analysis_id from a previous
      response (follow-up requests, to keep querying the same dataset)
      in a single multipart/form-data request.
    - Delegates to the OrchestratorAgent which runs the ReAct loop.
    - Returns structured EDA recommendations grounded in the RAG layer,
      plus the analysis_id to reuse for follow-up calls.
    """
    request = AnalysisRequest(goal=goal, expertise_level=expertise_level)
    try:
        result = await _orchestrator_service.run(request, file=file, analysis_id=analysis_id)
        return result
    except Exception as exc:
        logger.exception("Analysis pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {exc}",
        ) from exc


@router.get(
    "/knowledge-sources",
    response_model=list[KnowledgeSource],
    status_code=status.HTTP_200_OK,
    summary="List documents in the RAG knowledge base",
)
async def list_knowledge_sources() -> list[KnowledgeSource]:
    """Return the real EDA methodology documents the RecommendationAgent grounds on."""
    chunks = KnowledgeBaseLoader().load_all()
    by_source: dict[str, KnowledgeSource] = {}
    for chunk in chunks:
        source = chunk["source"]
        if source not in by_source:
            by_source[source] = KnowledgeSource(
                source=source,
                title=chunk["metadata"].get("title", source),
                doc_type=chunk["metadata"].get("doc_type", ""),
                chunk_count=0,
            )
        by_source[source].chunk_count += 1
    return list(by_source.values())


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
