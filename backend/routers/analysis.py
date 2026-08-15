"""
Analysis endpoints - run the goal-conditioned pipeline and ingest tabular files.

Endpoints:
    POST /analysis/run              - trigger a full MAGE analysis pipeline run
    POST /analysis/ingest           - upload a CSV or XLSX file and profile it
    GET  /analysis/sample-datasets  - list bundled demo datasets
    POST /analysis/sample-datasets/{filename}/load - ingest a bundled demo dataset
    GET  /analysis/knowledge-sources - list the RAG knowledge base documents
    GET  /analysis/history          - list the current user's past analysis runs
    GET  /analysis/history/{run_id} - fetch one past run in full
    GET  /analysis/history/{run_id}/export/pdf       - download a PDF report
    GET  /analysis/history/{run_id}/export/json      - download the raw run as JSON
    GET  /analysis/history/{run_id}/export/citations - download a BibTeX citation bundle
    GET  /analysis/datasets         - list the current user's uploaded datasets
    GET  /analysis/datasets/{id}                 - full detail for one dataset version
    GET  /analysis/datasets/{id}/preview         - paginated rows (spreadsheet view)
    GET  /analysis/datasets/{root_id}/versions   - every version sharing a lineage root
    POST /analysis/datasets/{id}/transform       - apply cleaning ops / cell edits -> new version
    POST /analysis/datasets/{id}/query           - read-only SQL preview (not persisted)
    POST /analysis/datasets/{id}/query/save      - re-run a query, persist result -> new version
    POST /analysis/datasets/{id}/query/nl        - translate plain English into SQL, then run it
    POST /analysis/explain                       - deeper RAG-grounded explanation of a finding

All endpoints except /knowledge-sources require authentication — analysis
runs and datasets are scoped to the authenticated user.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from agents.explain_agent import ExplainAgent
from agents.ingestion_agent import IngestionAgent
from backend.core.database import get_db
from backend.core.deps import get_current_user
from backend.models.user import User
from backend.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisRunSummary,
    ColumnStats,
    ColumnSummary,
    DatasetDetail,
    DatasetPreview,
    DatasetSummary,
    ExpertiseLevel,
    ExplainRequest,
    ExplainResult,
    IngestionResult,
    KnowledgeSource,
    NLQueryRequest,
    NLQueryResult,
    QueryRequest,
    RecommendationMode,
    SampleDataset,
    TransformRequest,
)
from backend.schemas.auth import MessageResponse
from backend.services import analysis_run_service, dataset_service, export_service, transform_service
from backend.services.orchestrator_service import OrchestratorService
from data_pipeline.ingestion import IngestionError
from data_pipeline.processing import ProcessingError
from rag.knowledge_loader import KnowledgeBaseLoader

logger = logging.getLogger(__name__)
router = APIRouter()

_orchestrator_service = OrchestratorService()
_ingestion_agent = IngestionAgent()
_explain_agent = ExplainAgent()

# Bundled demo datasets a user can load without having the file on their own
# machine. Keyed by filename in data/samples/; anything in that directory but
# not listed here is just generator scratch output, not meant to be surfaced.
SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"
_SAMPLE_DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    # filename -> (title, description)
    "customer_orders.csv": (
        "Customer Orders",
        "60 orders across 2 customer segments — clean correlation (units × price → "
        "revenue), a few injected outliers, and a churn label for classification goals.",
    ),
    "saas_customers.csv": (
        "SaaS Customers",
        "150 SaaS subscription customers across 3 tiers — real cluster structure, "
        "linear MRR correlation, injected anomalies, missing values, and a churn label.",
    ),
}


@router.post(
    "/run",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a goal-conditioned EDA pipeline",
)
async def run_analysis(
    goal: str = Form(...),
    expertise_level: ExpertiseLevel = Form(ExpertiseLevel.intermediate),
    mode: RecommendationMode = Form(RecommendationMode.rag),
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
    - `mode` selects how RecommendationAgent responds: "rag" (default,
      grounded/cited) or "llm" (freeform Gemini response, no citations).
    - Delegates to the OrchestratorAgent which runs the ReAct loop.
    - Persists the run to the user's history.
    - Returns structured EDA recommendations grounded in the RAG layer,
      plus dataset_id/run_id to reuse for follow-up calls / history lookups.
    """
    request = AnalysisRequest(goal=goal, expertise_level=expertise_level, mode=mode)
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
    "/sample-datasets",
    response_model=list[SampleDataset],
    status_code=status.HTTP_200_OK,
    summary="List bundled demo datasets",
)
async def list_sample_datasets() -> list[SampleDataset]:
    """Datasets shipped in the repo for demoing MAGE without a file of your own."""
    out: list[SampleDataset] = []
    for filename, (title, description) in _SAMPLE_DATASET_REGISTRY.items():
        path = SAMPLES_DIR / filename
        if not path.is_file():
            continue
        out.append(
            SampleDataset(
                filename=filename,
                title=title,
                description=description,
                size_kb=round(path.stat().st_size / 1024, 1),
            )
        )
    return out


@router.post(
    "/sample-datasets/{filename}/load",
    response_model=IngestionResult,
    status_code=status.HTTP_200_OK,
    summary="Ingest a bundled demo dataset and save it to your account",
)
async def load_sample_dataset(
    filename: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> IngestionResult:
    """Same ingest-and-persist path as /ingest, sourced from data/samples/
    instead of an upload — so a demo dataset behaves exactly like one the
    user picked from their own machine (own dataset_id, own history)."""
    if filename not in _SAMPLE_DATASET_REGISTRY:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown sample dataset.")
    path = SAMPLES_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample dataset file is missing.")

    content = path.read_bytes()
    source = dataset_service.StoredFile(filename=filename, file=io.BytesIO(content))
    try:
        result = _ingestion_agent.run(source=source)
    except IngestionError as exc:
        logger.info("Sample ingestion failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    dataset = await dataset_service.save_dataset(
        db,
        current_user.id,
        filename,
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
    run = await _get_owned_run(run_id, current_user, db)
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


async def _get_owned_run(run_id: str, current_user: User, db: AsyncSession):
    run = await analysis_run_service.get_run(db, current_user.id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis run not found.")
    return run


@router.get(
    "/history/{run_id}/export/pdf",
    status_code=status.HTTP_200_OK,
    summary="Download a PDF report for one analysis run",
)
async def export_run_pdf(
    run_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    run = await _get_owned_run(run_id, current_user, db)
    pdf_bytes = export_service.generate_pdf(run)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="mage-report-{run.id}.pdf"'},
    )


@router.get(
    "/history/{run_id}/export/json",
    status_code=status.HTTP_200_OK,
    summary="Download the raw analysis run as JSON",
)
async def export_run_json(
    run_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    run = await _get_owned_run(run_id, current_user, db)
    return Response(
        content=export_service.generate_json(run),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="mage-export-{run.id}.json"'},
    )


@router.get(
    "/history/{run_id}/export/citations",
    status_code=status.HTTP_200_OK,
    summary="Download a BibTeX citation bundle for one analysis run",
)
async def export_run_citations(
    run_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    run = await _get_owned_run(run_id, current_user, db)
    return Response(
        content=export_service.generate_citation_bundle(run),
        media_type="application/x-bibtex",
        headers={"Content-Disposition": f'attachment; filename="mage-citations-{run.id}.bib"'},
    )


def _to_summary(d) -> DatasetSummary:
    return DatasetSummary(
        id=d.id,
        filename=d.filename,
        row_count=d.row_count,
        column_count=d.column_count,
        created_at=d.created_at,
        root_id=d.root_id,
        parent_id=d.parent_id,
        version=d.version,
        transform_type=d.transform_type,
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
    return [_to_summary(d) for d in datasets]


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


async def _build_detail(dataset, *, report: list[str] | None = None) -> DatasetDetail:
    """Profile a Dataset row's content into a DatasetDetail — reuses
    IngestionAgent's column-summary logic rather than duplicating it."""
    stored = dataset_service.as_stored_file(dataset)
    try:
        profile = _ingestion_agent.run(source=stored)
    except IngestionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return DatasetDetail(
        **_to_summary(dataset).model_dump(),
        column_summary=profile.column_summary,
        transform_params=dataset.transform_params,
        report=report,
    )


@router.get(
    "/datasets/{dataset_id}/preview",
    response_model=DatasetPreview,
    status_code=status.HTTP_200_OK,
    summary="Paginated rows for the spreadsheet view",
)
async def preview_dataset(
    dataset_id: str,
    offset: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetPreview:
    offset = max(0, offset)
    limit = max(1, min(limit, 200))
    try:
        df = await transform_service.load_dataframe(db, current_user.id, dataset_id)
    except transform_service.TransformNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return DatasetPreview(**transform_service.build_preview(df, offset, limit))


@router.get(
    "/datasets/{root_id}/versions",
    response_model=list[DatasetSummary],
    status_code=status.HTTP_200_OK,
    summary="List every version sharing a lineage root, oldest first",
)
async def list_dataset_versions(
    root_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[DatasetSummary]:
    versions = await dataset_service.list_versions(db, current_user.id, root_id)
    if not versions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    return [_to_summary(d) for d in versions]


@router.post(
    "/datasets/{dataset_id}/transform",
    response_model=DatasetDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Apply cleaning ops / cell edits, producing a new dataset version",
)
async def transform_dataset(
    dataset_id: str,
    request: TransformRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetDetail:
    ops = [op.model_dump() for op in request.ops]
    try:
        new_dataset, report = await transform_service.apply_transform(db, current_user.id, dataset_id, ops)
    except transform_service.TransformNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ProcessingError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return await _build_detail(new_dataset, report=report)


@router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetDetail,
    status_code=status.HTTP_200_OK,
    summary="Fetch full detail for one dataset version",
)
async def get_dataset_detail(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetDetail:
    dataset = await dataset_service.get_dataset(db, current_user.id, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    return await _build_detail(dataset)


@router.post(
    "/datasets/{dataset_id}/query",
    response_model=DatasetPreview,
    status_code=status.HTTP_200_OK,
    summary="Run a read-only SQL query against a dataset (preview only, not persisted)",
)
async def query_dataset(
    dataset_id: str,
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetPreview:
    try:
        result_df = await transform_service.run_query(db, current_user.id, dataset_id, request.sql)
    except transform_service.TransformNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except transform_service.QueryValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return DatasetPreview(**transform_service.build_preview(result_df, 0, len(result_df)))


@router.post(
    "/datasets/{dataset_id}/query/save",
    response_model=DatasetDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Re-run a query server-side and persist the result as a new dataset version",
)
async def save_dataset_query(
    dataset_id: str,
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetDetail:
    try:
        new_dataset, row_count = await transform_service.save_query_result(
            db, current_user.id, dataset_id, request.sql
        )
    except transform_service.TransformNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except transform_service.QueryValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return await _build_detail(new_dataset, report=[f"Saved query result: {row_count} row(s)."])


@router.post(
    "/datasets/{dataset_id}/query/nl",
    response_model=NLQueryResult,
    status_code=status.HTTP_200_OK,
    summary="Translate a plain-English question into SQL and run it (preview only, not persisted)",
)
async def nl_query_dataset(
    dataset_id: str,
    request: NLQueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NLQueryResult:
    """The LLM only ever produces SQL text — that text goes through the
    exact same validation/sandboxed execution as a hand-typed query in
    /query above, so this introduces no new trust boundary. Never persists;
    the returned `sql` can be POSTed to /query/save unchanged to keep it."""
    try:
        sql, result_df = await transform_service.generate_sql_from_question(
            db, current_user.id, dataset_id, request.question
        )
    except transform_service.TransformNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except transform_service.QueryValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return NLQueryResult(
        sql=sql,
        preview=DatasetPreview(**transform_service.build_preview(result_df, 0, len(result_df))),
    )


@router.post(
    "/explain",
    response_model=ExplainResult,
    status_code=status.HTTP_200_OK,
    summary="Deeper, RAG-grounded explanation of a specific finding (LLM-synthesized when configured)",
)
async def explain_finding(
    request: ExplainRequest,
    current_user: User = Depends(get_current_user),
) -> ExplainResult:
    """Stateless — the frontend already has the finding text client-side
    from the already-fetched analysis result, so no run_id lookup is
    needed. Always grounded: falls back to a raw retrieved excerpt
    (synthesized=False) rather than ever returning ungrounded LLM text."""
    result = _explain_agent.explain(request.finding, request.goal)
    return ExplainResult(**result)
