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

# Copy pyproject.toml + README first (hatchling requires the declared
# readme to exist at build time) — leverages layer caching.
COPY backend/pyproject.toml backend/README.md ./

# Install pip + hatch, then install the project's dependencies.
#
# pyproject.toml is the only source of truth, deliberately. There used to be a
# hardcoded package list behind `2>/dev/null ||` as a fallback, and it had
# drifted eight dependencies behind pyproject: scikit-learn, shap, reportlab,
# pdfplumber, duckdb, authlib, itsdangerous, psycopg2-binary. Because stderr
# was discarded, a failed resolution produced an image that *built cleanly* and
# then died at runtime with nothing in the log explaining why — no models in
# M3 at all (scikit-learn), no PDF export, no PDF ingestion, no OAuth, no
# query console. A build that fails loudly beats an image that lies about
# being complete, so if this step fails, let it fail.
#
# Torch is installed first, from PyTorch's CPU index, on purpose.
# `sentence-transformers` depends on torch, and plain PyPI resolution on
# linux/arm64 now drags in the whole CUDA stack — nvidia-cublas (543 MB),
# nvidia-cudnn (445 MB), cusolver, cusparselt, nccl, triton and friends, about
# 2.8 GB of wheels — into a container that will never see an NVIDIA GPU. That
# bloat is what pushed this image to 9.87 GB and filled 49 GB of build cache.
# MAGE's embeddings are CPU-only by design (all-MiniLM-L6-v2, no key, no GPU),
# so pre-installing the CPU build means the project install below finds torch
# already satisfied and never resolves the GPU variant.
RUN pip install --upgrade pip hatch && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir ".[dev]"

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

# The RAG knowledge base and bundled sample datasets are read from disk at
# runtime (KnowledgeBaseLoader, the /sample-datasets endpoint) — without this
# the container boots fine but has nothing to ground recommendations in and
# no sample datasets to offer, silently. faiss_index/chroma_db/uploads stay
# runtime-created scratch space, not shipped in the image.
COPY data/knowledge_base/ ./data/knowledge_base/
COPY data/samples/ ./data/samples/
RUN mkdir -p /app/data/faiss_index /app/data/chroma_db /app/uploads

# Non-root user for security
RUN useradd -m -u 1001 mage && chown -R mage:mage /app
USER mage

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Note: curl is installed in the builder stage only, so the healthcheck uses
# python (always present here) rather than shelling out to curl.
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
