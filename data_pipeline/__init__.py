"""
data_pipeline/
──────────────
MAGE data ingestion and processing engine wrappers.

Exports:
    DataIngestionEngine  — source loaders (CSV, JSON, Parquet, SQL, …)
    DataProcessingEngine — Pandas-first transform layer with Spark/Dask hooks
"""

from data_pipeline.ingestion import DataIngestionEngine
from data_pipeline.processing import DataProcessingEngine

__all__ = ["DataIngestionEngine", "DataProcessingEngine"]
