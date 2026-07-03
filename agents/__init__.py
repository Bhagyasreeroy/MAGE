"""
agents/
───────
MAGE specialist and orchestrator agents.

Exports:
    OrchestratorAgent  — goal conditioning + ReAct loop
    IngestionAgent     — multimodal data ingestion & validation
    MiningAgent        — statistical profiling & pattern discovery
    VisualizationAgent — chart selection & rendering
    RecommendationAgent — RAG-grounded, expertise-adapted recommendations
"""

from agents.orchestrator import OrchestratorAgent
from agents.ingestion_agent import IngestionAgent
from agents.mining_agent import MiningAgent
from agents.visualization_agent import VisualizationAgent
from agents.recommendation_agent import RecommendationAgent

__all__ = [
    "OrchestratorAgent",
    "IngestionAgent",
    "MiningAgent",
    "VisualizationAgent",
    "RecommendationAgent",
]
