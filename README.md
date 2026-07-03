# 🧙 MAGE — Multi-Agent Goal-conditioned EDA

> **Ingest data. Define a goal. Let MAGE's agents do the analysis.**

MAGE is a multi-agent AI system for **exploratory data analysis**. It accepts
heterogeneous multimodal data, conditions the EDA workflow on a user-defined
analytical goal, and returns **RAG-grounded, explainable recommendations**
adapted to the user's expertise level.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1 — User Interface (Next.js 14 + TypeScript + Tailwind)   │
│  Goal input · Expertise selector · Results dashboard             │
└──────────────────────┬───────────────────────────────────────────┘
                       │ REST / HTTP
┌──────────────────────▼───────────────────────────────────────────┐
│  Layer 2 — Backend API (FastAPI + Pydantic)                       │
│  POST /analysis/run · GET /health                                 │
└──────────────────────┬───────────────────────────────────────────┘
                       │ Python
┌──────────────────────▼───────────────────────────────────────────┐
│  Layer 3 — Agents (ReAct loop)                                    │
│  OrchestratorAgent → IngestionAgent → MiningAgent                 │
│                    → VisualizationAgent → RecommendationAgent     │
└──────┬───────────────────────────────┬────────────────────────────┘
       │ Pandas / Dask / Spark          │ RAG queries
┌──────▼───────────┐   ┌───────────────▼────────────────────────────┐
│ Data Pipeline    │   │ RAG Pipeline                                │
│ Ingestion Engine │   │ VectorStore (FAISS / ChromaDB)              │
│ Processing Engine│   │ embed_text · retrieve · KnowledgeLoader     │
└──────────────────┘   └─────────────────────────────────────────────┘
                       │ Docker Compose → Kubernetes
┌──────────────────────▼───────────────────────────────────────────┐
│  Layer 5 — Infrastructure                                         │
│  backend:8000 · frontend:3000 · chromadb:8001                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Folder Layout

```
MAGE/
├── agents/                     # Orchestrator + specialist agents
│   ├── orchestrator.py         # OrchestratorAgent (ReAct loop)
│   ├── ingestion_agent.py      # IngestionAgent
│   ├── mining_agent.py         # MiningAgent
│   ├── visualization_agent.py  # VisualizationAgent
│   └── recommendation_agent.py # RecommendationAgent
│
├── backend/                    # FastAPI application
│   ├── main.py                 # App entry point
│   ├── core/config.py          # Pydantic Settings
│   ├── routers/                # health, analysis
│   ├── services/               # orchestrator_service
│   ├── schemas/                # Pydantic models
│   ├── Dockerfile              # Multi-stage Python image
│   └── pyproject.toml          # Dependencies + build config
│
├── frontend/                   # Next.js 14 application
│   ├── app/
│   │   ├── page.tsx            # Landing page (goal input + results)
│   │   ├── layout.tsx          # Root layout + metadata
│   │   └── globals.css         # Global styles
│   ├── Dockerfile              # Multi-stage Node image
│   └── next.config.ts          # Next.js config (standalone mode)
│
├── rag/                        # RAG pipeline
│   ├── vector_store.py         # VectorStore (FAISS / ChromaDB)
│   ├── embeddings.py           # embed_text() + embed_batch()
│   └── knowledge_loader.py     # KnowledgeBaseLoader
│
├── data_pipeline/              # Data processing engines
│   ├── ingestion.py            # DataIngestionEngine
│   └── processing.py          # DataProcessingEngine (Pandas/Dask/Spark)
│
├── infra/
│   └── k8s/README.md           # Kubernetes manifests (placeholder)
│
├── tests/
│   ├── backend/test_health.py  # Health endpoint tests
│   └── agents/test_orchestrator.py  # OrchestratorAgent smoke tests
│
├── docs/
│   ├── architecture.md         # Detailed architecture doc
│   └── api_contracts.md        # API endpoint documentation
│
├── .env.example                # Environment variable template
├── .gitignore                  # Python + Node + Docker gitignore
├── docker-compose.yml          # Full dev stack
└── README.md                   # This file
```

---

## Quick Start — Local Development

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker + Docker Compose

### 1. Clone & configure

```bash
git clone <repo-url>
cd MAGE
cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Backend only

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Run the server
cd ..
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000

# Check it works
curl http://localhost:8000/health
```

### 3. Frontend only

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

### 4. Full stack via Docker Compose

```bash
# From the repo root
docker compose up --build

# Services:
#   Backend   → http://localhost:8000
#   Frontend  → http://localhost:3000
#   ChromaDB  → http://localhost:8001
#   API docs  → http://localhost:8000/docs
```

---

## Running Tests

```bash
# Backend tests (from repo root)
cd backend
pip install -e ".[dev]"
cd ..
PYTHONPATH=. pytest tests/ -v

# Frontend tests
cd frontend
npm run lint
```

---

## End-to-End Vertical Slice

The skeleton already wires the full request path:

```
POST http://localhost:8000/analysis/run
Content-Type: application/json

{
  "goal": "Identify the top factors driving customer churn.",
  "expertise_level": "intermediate"
}
```

Returns a structured response with dummy agent steps + placeholder recommendations.
Real logic will be added incrementally to each agent.

---

## Roadmap

| Milestone | Focus |
|---|---|
| M0 (now) | Clean skeleton: end-to-end vertical slice, all stubs wired |
| M1 | IngestionAgent: real CSV/JSON/Parquet loading |
| M2 | MiningAgent: Pandas statistical profiling |
| M3 | RAG: ChromaDB + real embeddings (OpenAI / sentence-transformers) |
| M4 | RecommendationAgent: LLM-powered, RAG-grounded recommendations |
| M5 | VisualizationAgent: Vega-Lite specs + frontend chart rendering |
| M6 | LangGraph-based orchestration |
| M7 | Kubernetes production deployment |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Pydantic v2, Uvicorn |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Agents | Custom ReAct loop (LangGraph-ready) |
| Vector Store | FAISS (local) + ChromaDB (server) |
| Data Processing | Pandas → Dask / Spark hooks |
| Containerisation | Docker + docker-compose → Kubernetes |
| Testing | pytest (backend), Vitest (frontend — future) |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit your changes: `git commit -m "feat: add X"`
4. Push: `git push origin feat/my-feature`
5. Open a pull request

---

## License

MIT — see `LICENSE` for details.
