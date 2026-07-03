"""
routers/analysis.py
────────────────────
Analysis endpoints — accept a goal + expertise level (and optionally
multimodal data), delegate to the OrchestratorService, and return a
structured EDA result.

Endpoints:
    POST /analysis/run   — trigger a full MAGE analysis pipeline run
"""

import logging
from fastapi import APIRouter, HTTPException, status

from backend.schemas.analysis import AnalysisRequest, AnalysisResponse
from backend.services.orchestrator_service import OrchestratorService

logger = logging.getLogger(__name__)
router = APIRouter()

_orchestrator_service = OrchestratorService()


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

    **Note (skeleton):** Currently returns a dummy response — business
    logic will be wired in subsequent milestones.
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
