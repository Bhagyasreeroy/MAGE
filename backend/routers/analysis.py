"""
Analysis endpoints - run the goal-conditioned pipeline and ingest tabular files.

Endpoints:
    POST /analysis/run              - trigger a full MAGE analysis pipeline run
    WS   /analysis/stream           - run the pipeline, streaming each agent step live
    POST /analysis/ingest           - upload a CSV or XLSX file and profile it
    GET  /analysis/knowledge-sources - list the RAG knowledge base documents
    GET  /analysis/history          - list the current user's past analysis runs
    GET  /analysis/history/{run_id} - fetch one past run in full
    GET  /analysis/history/{run_id}/export/pdf       - download a PDF report
    GET  /analysis/history/{run_id}/export/json      - download the raw run as JSON
    GET  /analysis/history/{run_id}/export/citations - download a BibTeX citation bundle
    GET  /analysis/datasets         - list the current user's uploaded datasets

All endpoints except /knowledge-sources require authentication — analysis
runs and datasets are scoped to the authenticated user.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import Response
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.ingestion_agent import IngestionAgent
from backend.core.database import async_session, get_db
from backend.core.deps import get_current_user
from backend.core.security import decode_token
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
from backend.services import analysis_run_service, dataset_service, export_service
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


# ── Live streaming ───────────────────────────────────────────────────────────
#
# Application-defined WebSocket close codes. The 4000-4999 range is reserved
# for private use by RFC 6455, so these will never collide with protocol codes.
WS_UNAUTHORIZED = 4401
WS_BAD_REQUEST = 4400
WS_INTERNAL_ERROR = 4500

# Polling interval for the step queue while the pipeline thread works. Short
# enough that steps appear immediately, long enough not to spin the event loop.
_STREAM_POLL_SECONDS = 0.05


async def _user_from_ws_token(token: str | None, db: AsyncSession) -> User | None:
    """
    Resolve a user from a JWT passed as a WebSocket query parameter.

    The browser WebSocket API cannot set an Authorization header, so the token
    travels as a query parameter instead — the standard workaround. This
    mirrors ``get_current_user``'s checks exactly (valid signature, ``access``
    token type, user exists and is active) rather than relaxing any of them;
    only the transport differs.
    """
    if not token:
        return None
    try:
        payload = decode_token(token)
    except JWTError:
        return None

    user_id = payload.get("sub")
    if user_id is None or payload.get("type") != "access":
        return None

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        return None
    return user


def _stream_safe_step(step: dict[str, Any], index: int) -> dict[str, Any]:
    """
    Shrink one step to what a live view actually renders.

    The full ``output`` payload carries every statistic the MiningAgent
    produced and can run to hundreds of kilobytes — pushing it per step would
    make the stream slower than the analysis it is narrating. The complete,
    unabridged result still arrives in the terminal ``complete`` message and is
    what gets persisted, so nothing is lost.
    """
    return {
        "type": "step",
        "index": index,
        "agent_name": step.get("agent_name"),
        "action": step.get("action"),
        "reasoning": step.get("reasoning"),
        "observation": step.get("observation"),
        "status": step.get("status"),
        "latency_ms": step.get("latency_ms"),
    }


@router.websocket("/stream")
async def stream_analysis(websocket: WebSocket, token: str | None = None) -> None:
    """
    Run the pipeline over a WebSocket, streaming each agent step as it completes.

    Protocol
    --------
    Connect to ``/analysis/stream?token=<access_token>``, then send one JSON
    message to start the run::

        {"goal": "...", "expertise_level": "intermediate", "dataset_id": "..."}

    The server then emits, in order::

        {"type": "accepted", "goal": ...}
        {"type": "step", "index": 0, "agent_name": "IngestionAgent", ...}
        ...one per completed step...
        {"type": "complete", "run_id": ..., "result": {...}}

    or ``{"type": "error", "detail": ...}`` followed by a close.

    Datasets are referenced by ``dataset_id`` rather than uploaded over the
    socket — the client uploads via ``POST /analysis/ingest`` first, which
    already persists the file and returns its id.
    """
    # The session is opened explicitly rather than via ``Depends(get_db)``:
    # FastAPI's yield-dependency teardown does not run reliably for WebSocket
    # routes, so every connection that touched the database leaked its
    # connection until SQLAlchemy's garbage collector reclaimed it. An explicit
    # ``async with`` closes it deterministically when the socket is done.
    async with async_session() as db:
        await _stream_analysis(websocket, token, db)


async def _stream_analysis(websocket: WebSocket, token: str | None, db: AsyncSession) -> None:
    """Body of the streaming endpoint, with the DB session's lifetime fixed by the caller."""
    user = await _user_from_ws_token(token, db)
    if user is None:
        # Accept before closing so the client receives a close *code* it can
        # act on. Rejecting pre-accept yields a bare HTTP 403, which browsers
        # surface as an indistinguishable connection error.
        await websocket.accept()
        await websocket.close(code=WS_UNAUTHORIZED, reason="Invalid or missing access token.")
        return

    await websocket.accept()

    try:
        payload = await websocket.receive_json()
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001 - malformed frame, not valid JSON
        await websocket.close(code=WS_BAD_REQUEST, reason="Expected a JSON start message.")
        return

    goal = str(payload.get("goal") or "").strip()
    if not goal:
        await websocket.close(code=WS_BAD_REQUEST, reason="A non-empty 'goal' is required.")
        return

    try:
        expertise = ExpertiseLevel(payload.get("expertise_level") or ExpertiseLevel.intermediate)
    except ValueError:
        await websocket.close(code=WS_BAD_REQUEST, reason="Unknown expertise_level.")
        return

    request = AnalysisRequest(goal=goal, expertise_level=expertise)
    await websocket.send_json({"type": "accepted", "goal": goal, "expertise_level": expertise.value})

    # Bridge the worker thread back to the event loop. `on_step` is invoked by
    # the agent on the thread running the analysis, so it must not touch the
    # socket directly — it hands each step to the loop via a thread-safe call
    # and this coroutine does the sending.
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def on_step(step: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, step)

    run_task = asyncio.create_task(
        _orchestrator_service.run(
            request,
            db=db,
            user_id=user.id,
            dataset_id=payload.get("dataset_id"),
            on_step=on_step,
        )
    )

    index = 0
    try:
        # Drain until the pipeline has finished *and* the queue is empty, so a
        # step enqueued just before completion is never dropped.
        while not (run_task.done() and queue.empty()):
            try:
                step = await asyncio.wait_for(queue.get(), timeout=_STREAM_POLL_SECONDS)
            except asyncio.TimeoutError:
                continue
            await websocket.send_json(_stream_safe_step(step, index))
            index += 1

        result = await run_task
        await websocket.send_json(
            {"type": "complete", "run_id": result.run_id, "result": result.model_dump(mode="json")}
        )
    except WebSocketDisconnect:
        # The client hung up. The run is left to finish so work already done
        # still gets persisted — the `finally` below is what waits for it.
        logger.info("Stream client disconnected; letting the run finish.")
    except Exception as exc:  # noqa: BLE001 - report, then close cleanly
        logger.exception("Streaming analysis failed: %s", exc)
        try:
            await websocket.send_json({"type": "error", "detail": f"Analysis failed: {exc}"})
            await websocket.close(code=WS_INTERNAL_ERROR)
        except Exception:  # noqa: BLE001 - socket may already be gone
            pass
    finally:
        # The pipeline task holds the request-scoped `db` session. Returning
        # while it is still running would let FastAPI tear that session down
        # underneath it, orphaning the connection (SQLAlchemy then reclaims it
        # via the garbage collector and warns). Settle the task first, always.
        await _settle(run_task)

    await _close_quietly(websocket)


async def _settle(task: asyncio.Task) -> None:
    """Wait for a task to finish, absorbing its result or failure."""
    try:
        await task
    except Exception:  # noqa: BLE001 - already reported by the caller, or client-gone
        logger.debug("Streamed run ended with an exception.", exc_info=True)


async def _close_quietly(websocket: WebSocket) -> None:
    """Close a socket that may already have been closed by either side."""
    try:
        await websocket.close()
    except Exception:  # noqa: BLE001 - closing an already-closed socket is fine
        pass


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
