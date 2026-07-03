"""
routers/health.py
─────────────────
Health-check endpoint — used by Docker health checks and load balancers
to confirm the service is alive and ready.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check() -> HealthResponse:
    """Return service liveness status."""
    return HealthResponse(
        status="ok",
        service="MAGE Backend",
        version="0.1.0",
    )
