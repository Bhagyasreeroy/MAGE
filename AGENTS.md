# MAGE — Agent Rules

This is the MAGE monorepo (Multi-Agent Goal-conditioned EDA).

## Current scope: M1 — Ingestion
- Make `IngestionAgent` load real data via `DataIngestionEngine`.
- Files in scope: `data_pipeline/ingestion.py`, `agents/ingestion_agent.py`, and a new test file only.
- DO NOT touch: other agents, the RAG layer, the frontend, or infra.

## Conventions
- Match the existing stub docstring style.
- Use pandas only — no Spark/Dask (those raise NotImplementedError on purpose).
- Verify with: `PYTHONPATH=. pytest tests/ -v`
- Keep all 12 existing tests green.