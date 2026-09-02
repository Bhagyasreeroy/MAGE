"""
main.py
───────
MAGE FastAPI application entry point.

Registers routers, configures CORS middleware, and exposes the app
object for uvicorn to serve.
"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from backend.core.rate_limit import limiter
from starlette.middleware.sessions import SessionMiddleware

from backend.core.config import settings
from backend.core.database import init_db
from backend.routers import analysis, auth, health, oauth, ocr

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Run startup / shutdown tasks."""
    # Import models so Base.metadata knows about all tables
    import backend.models  # noqa: F401
    await init_db()

    # Preload the embedding model in the background. It otherwise loads on the
    # first request that needs a retrieval, stalling that user's analysis for
    # ~10s — plainly visible in the live stream as a hang on the
    # RecommendationAgent step. Backgrounded rather than awaited so the API
    # starts serving immediately; requests arriving before it finishes simply
    # take the lazy path they would have taken anyway.
    from rag.embeddings import warm_up

    warm_task = asyncio.create_task(asyncio.to_thread(warm_up))

    try:
        yield
    finally:
        warm_task.cancel()
        with suppress(asyncio.CancelledError):
            await warm_task


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
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
# SessionMiddleware must be added first — authlib needs it for OAuth state
app.add_middleware(SessionMiddleware, secret_key=settings.session_secret_key)

# ── Rate limiting (M8) ────────────────────────────────────────────────────────
# Registered before CORS so a 429 still carries the CORS headers a browser needs
# to read it — otherwise the frontend sees an opaque network error instead of
# "slow down", which is indistinguishable from the server being down.
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Explain the refusal. A bare 429 with no body reads as a broken server."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                "Rate limit exceeded — too many requests. "
                f"This endpoint allows {exc.detail}. Please retry shortly."
            )
        },
    )


# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # allow_headers governs *request* headers; a browser cannot read a
    # *response* header cross-origin unless it is exposed explicitly. The
    # frontend (:3000) reads Content-Disposition off the export responses
    # (:8000) to name the downloaded file, so without this every export saves
    # as "download" — and the BibTeX bundle lands with no extension at all,
    # since Chrome can only guess one from the MIME type and does not know
    # application/x-bibtex. The server was always sending the right filename;
    # it just never reached JavaScript.
    expose_headers=["Content-Disposition"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(oauth.router, prefix="/auth", tags=["Authentication"])
app.include_router(ocr.router, prefix="/api/v1", tags=["OCR"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.environment == "development",
    )
