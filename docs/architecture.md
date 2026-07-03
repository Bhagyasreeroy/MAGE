# MAGE Architecture

## Overview

MAGE (Multi-Agent Goal-conditioned EDA) is a multi-agent AI system for
exploratory data analysis. It ingests multimodal heterogeneous data,
conditions EDA workflows on a user-defined analytical goal, and returns
RAG-grounded, explainable recommendations adapted to the user's expertise level.

---

## 5-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — User Interface                                        │
│  Next.js 14 + TypeScript + Tailwind                              │
│  • Goal input field                                              │
│  • Expertise-level selector (Beginner / Intermediate / Expert)   │
│  • Multimodal file upload                                        │
│  • Explainability dashboard (charts + recommendations)           │
└──────────────────────┬──────────────────────────────────────────┘
                       │ HTTP (REST)
┌──────────────────────▼──────────────────────────────────────────┐
│  Layer 2 — Backend API                                           │
│  FastAPI + Pydantic + Uvicorn                                    │
│  • POST /analysis/run                                            │
│  • GET  /health                                                  │
│  • OrchestratorService (bridges HTTP ↔ agents)                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Python function calls
┌──────────────────────▼──────────────────────────────────────────┐
│  Layer 3 — Orchestrator + Specialist Agents                      │
│                                                                  │
│  OrchestratorAgent (ReAct loop)                                  │
│    ├── IngestionAgent      — multimodal data parsing & QA        │
│    ├── MiningAgent         — statistical profiling & patterns    │
│    ├── VisualizationAgent  — chart selection & VizSpec output    │
│    └── RecommendationAgent — RAG-grounded, expertise-adapted     │
└──────────┬───────────────────────────┬──────────────────────────┘
           │ DataFrame / signals        │ RAG queries
┌──────────▼──────────┐   ┌────────────▼──────────────────────────┐
│  Layer 4a           │   │  Layer 4b — RAG Pipeline               │
│  Data Pipeline      │   │  VectorStore (FAISS / ChromaDB)        │
│  DataIngestionEngine│   │  embed_text() — text → dense vectors   │
│  DataProcessingEngine│  │  VectorStore.retrieve() — ANN search   │
│  (Pandas/Dask/Spark)│   │  KnowledgeBaseLoader — doc ingestion   │
└─────────────────────┘   └────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│  Layer 5 — Deployment & Infrastructure                           │
│  Docker + docker-compose (dev) → Kubernetes (prod)               │
│  Services: backend (8000) · frontend (3000) · chromadb (8001)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### OrchestratorAgent (`agents/orchestrator.py`)

The central coordinator. Implements a **ReAct** (Reason → Act → Observe) loop:

1. **REASON** — inspect the current context and goal; decide which specialist agent to invoke next (currently static; future: LLM-driven dynamic planning).
2. **ACT** — call the specialist agent's `run(context)` method.
3. **OBSERVE** — record the output; update the shared context.

The loop repeats up to `MAX_REACT_STEPS` times, then calls `_aggregate()` to produce the final result.

### IngestionAgent (`agents/ingestion_agent.py`)

Parses and validates heterogeneous data uploads (CSV, JSON, Parquet, PDF, images). Normalises everything into a canonical Pandas DataFrame + metadata dict.

### MiningAgent (`agents/mining_agent.py`)

Performs statistical profiling (descriptive stats, correlation, outlier detection) and goal-conditioned pattern discovery.

### VisualizationAgent (`agents/visualization_agent.py`)

Selects chart types based on data characteristics and goal, produces `VizSpec` manifests renderable by the frontend.

### RecommendationAgent (`agents/recommendation_agent.py`)

Queries the RAG vector store, synthesises outputs from all upstream agents, and generates expertise-adapted EDA recommendations with source citations.

### VectorStore (`rag/vector_store.py`)

Unified abstraction over FAISS (file-based) and ChromaDB (server-based). Exposes `initialize()`, `add_documents()`, `retrieve()`.

### DataIngestionEngine (`data_pipeline/ingestion.py`)

Multi-source loader: local files, remote URLs, databases. Normalises to Pandas DataFrame.

### DataProcessingEngine (`data_pipeline/processing.py`)

Transform layer. Pandas by default; raises `NotImplementedError` for Spark/Dask until those backends are implemented.

---

## Data Flow (Single Request)

```
User submits goal + data
        │
        ▼
POST /analysis/run (FastAPI)
        │
        ▼
OrchestratorService.run()
        │
        ▼
OrchestratorAgent.run()
  ├─ ReAct step 0: IngestionAgent.run(context)
  ├─ ReAct step 1: MiningAgent.run(context)
  ├─ ReAct step 2: VisualizationAgent.run(context)
  └─ ReAct step 3: RecommendationAgent.run(context)
                        │
                        ├─ VectorStore.retrieve(query)
                        └─ LLM call (future)
        │
        ▼
AnalysisResponse (JSON)
        │
        ▼
Frontend renders dashboard
```

---

## Technology Choices

| Component | Technology | Rationale |
|---|---|---|
| Backend framework | FastAPI | Async-native, automatic OpenAPI docs, Pydantic v2 integration |
| Config | Pydantic Settings | Type-safe env loading, no runtime surprises |
| Vector DB (server) | ChromaDB | Easy Docker deployment, persistent, REST API |
| Vector DB (local) | FAISS | Zero-dependency ANN, fast for single-node experiments |
| Data processing | Pandas → Dask/Spark | Start simple, scale when needed |
| Agent loop | Custom ReAct | Full control; swap for LangGraph when complexity grows |
| Frontend | Next.js 14 App Router | RSC support, file-based routing, TypeScript-first |
| Containerisation | Docker Compose → K8s | Dev simplicity; prod-grade manifests to follow |
