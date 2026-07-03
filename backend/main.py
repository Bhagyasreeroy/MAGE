"""
main.py
───────
MAGE FastAPI application entry point.

Registers routers, configures CORS middleware, and exposes the app
object for uvicorn to serve.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.routers import health, analysis

app = FastAPI(
    title=settings.app_name,
    description=(
        "MAGE — Multi-Agent Goal-conditioned EDA. "
        "Ingests heterogeneous data, conditions EDA workflows on a user-defined "
        "analytical goal, and returns RAG-grounded, explainable recommendations."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.environment == "development",
    )
