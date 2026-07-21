# ── MAGE Backend Dockerfile ───────────────────────────────────────────────────
# Multi-stage build:
#   Stage 1 (builder): Install all Python dependencies.
#   Stage 2 (runtime): Copy only what's needed to keep the image lean.

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for native extensions (faiss, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first (leverages layer caching)
COPY backend/pyproject.toml ./

# Install pip + hatch, then install the project's dependencies
RUN pip install --upgrade pip hatch && \
    pip install --no-cache-dir ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pytest \
    pytest-asyncio \
    httpx \
    python-multipart \
    pydantic \
    pydantic-settings \
    python-dotenv \
    langchain \
    langchain-community \
    langchain-openai \
    faiss-cpu \
    chromadb \
    openpyxl \
    pandas \
    numpy \
    pyarrow


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY backend/ ./backend/
COPY agents/ ./agents/
COPY rag/ ./rag/
COPY data_pipeline/ ./data_pipeline/

# Create data directories
RUN mkdir -p /app/data/faiss_index /app/data/chroma_db /app/uploads

# Non-root user for security
RUN useradd -m -u 1001 mage && chown -R mage:mage /app
USER mage

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
