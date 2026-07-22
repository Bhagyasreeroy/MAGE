"""
Analysis endpoints - run the goal-conditioned pipeline and ingest tabular files.

Endpoints:
    POST /analysis/run              - trigger a full MAGE analysis pipeline run
    POST /analysis/ingest           - upload a CSV or XLSX file and profile it
    GET  /analysis/knowledge-sources - list the RAG knowledge base documents
    GET  /analysis/history          - list the current user's past analysis runs
    GET  /analysis/history/{run_id} - fetch one past run in full
    GET  /analysis/datasets         - list the current user's uploaded datasets

All endpoints except /knowledge-sources require authentication — analysis
runs and datasets are scoped to the authenticated user.
"""

from __future__ import annotations

import io
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from agents.ingestion_agent import IngestionAgent
from backend.core.database import get_db
from backend.core.deps import get_current_user
from backend.models.user import User
from backend.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisRunSummary,
    DatasetSummary,
    ExpertiseLevel,
    IngestionResult,
    KnowledgeSource,
)
from backend.schemas.auth import MessageResponse
from backend.services import analysis_run_service, dataset_service
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
    dataset_id: str | None = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    """
    Trigger the full MAGE pipeline for a given analytical goal.

    - Accepts a natural-language goal, a user expertise level, and either
      a dataset file (first request) or a dataset_id from a previous
      response (follow-up requests, to keep querying the same dataset)
      in a single multipart/form-data request.
    - Delegates to the OrchestratorAgent which runs the ReAct loop.
    - Persists the run to the user's history.
    - Returns structured EDA recommendations grounded in the RAG layer,
      plus dataset_id/run_id to reuse for follow-up calls / history lookups.
    """
    request = AnalysisRequest(goal=goal, expertise_level=expertise_level)
    try:
        result = await _orchestrator_service.run(
            request, db=db, user_id=current_user.id, file=file, dataset_id=dataset_id
        )
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
    summary="Ingest, profile, and persist a tabular file",
)
async def ingest_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> IngestionResult:
    """Upload a file, return a structured ingestion profile, and persist it
    as a Dataset owned by the current user (only once ingestion succeeds)."""
    content = await file.read()
    source = dataset_service.StoredFile(filename=file.filename or "dataset", file=io.BytesIO(content))
    try:
        result = _ingestion_agent.run(source=source)
    except IngestionError as exc:
        logger.info("Ingestion failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected ingestion failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    dataset = await dataset_service.save_dataset(
        db,
        current_user.id,
        file.filename or "dataset",
        content,
        row_count=result.row_count,
        column_count=result.column_count,
    )
    result.dataset_id = dataset.id
    return result


@router.get(
    "/history",
    response_model=list[AnalysisRunSummary],
    status_code=status.HTTP_200_OK,
    summary="List the current user's past analysis runs",
)
async def list_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[AnalysisRunSummary]:
    runs = await analysis_run_service.list_runs(db, current_user.id)
    return [
        AnalysisRunSummary(
            id=r.id,
            goal=r.goal,
            expertise_level=r.expertise_level,
            status=r.status,
            summary=r.summary,
            dataset_id=r.dataset_id,
            created_at=r.created_at,
        )
        for r in runs
    ]


@router.get(
    "/history/{run_id}",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch one past analysis run in full",
)
async def get_history_run(
    run_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    run = await analysis_run_service.get_run(db, current_user.id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis run not found.")
    return AnalysisResponse(
        goal=run.goal,
        expertise_level=run.expertise_level,
        steps=run.steps,
        recommendations=run.recommendations,
        rag_sources=run.rag_sources,
        summary=run.summary,
        dataset_id=run.dataset_id,
        run_id=run.id,
    )


@router.get(
    "/datasets",
    response_model=list[DatasetSummary],
    status_code=status.HTTP_200_OK,
    summary="List the current user's uploaded datasets",
)
async def list_datasets(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[DatasetSummary]:
    datasets = await dataset_service.list_datasets(db, current_user.id)
    return [
        DatasetSummary(
            id=d.id,
            filename=d.filename,
            row_count=d.row_count,
            column_count=d.column_count,
            created_at=d.created_at,
        )
        for d in datasets
    ]


@router.delete(
    "/datasets/{dataset_id}",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete one of the current user's uploaded datasets",
)
async def delete_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    deleted = await dataset_service.delete_dataset(db, current_user.id, dataset_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    return MessageResponse(message="Dataset deleted.")
