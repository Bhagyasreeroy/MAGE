"""
agents/visualization_agent.py
──────────────────────────────
VisualizationAgent — intelligent chart selection and rendering.

Responsibilities:
    • Consume the StatisticalProfile from MiningAgent and the user goal to
      select the most informative chart types for each insight.
    • Chart selection heuristics (and eventually LLM-driven decisions):
        - Distribution → histogram, violin, box plot
        - Correlation  → scatter matrix, heatmap
        - Time-series  → line chart, area chart, decomposition plot
        - Categorical  → bar chart, treemap, pie (when slices < 6)
        - Geospatial   → choropleth, bubble map
    • Adapt visual complexity and annotation density to the user's
      expertise level (fewer jargon labels for beginners).
    • Produce a VizSpec manifest (chart type + data bindings) that can be
      rendered by the frontend (e.g., via Vega-Lite, Recharts, or Plotly).

Wiring (future milestones):
    - Will consume MiningAgent_output from context.
    - Will emit a list of VizSpec objects.
    - Frontend will render the specs via a Recharts / Vega-Lite component.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VisualizationAgent:
    """
    Selects and specifies EDA charts conditioned on the analytical goal.

    Input context keys consumed:
        - ``MiningAgent_output`` : statistical profile + patterns
        - ``goal``               : analytical goal
        - ``expertise_level``    : adapts chart complexity

    Output keys produced:
        - ``viz_specs``: list of VizSpec dicts (type, title, data bindings)
    """

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Select and specify visualisations for the current EDA context.

        Parameters
        ----------
        context : dict, optional
            Shared pipeline context forwarded by the Orchestrator.

        Returns
        -------
        dict
            Visualisation specification payload.

        TODO:
            - Read StatisticalProfile from context.
            - Implement chart-type selection heuristics.
            - Integrate LLM-based chart recommendation.
            - Produce valid Vega-Lite / Recharts specs.
        """
        context = context or {}
        directives = context.get("directives", {}) or {}
        task_type = directives.get("task_type") or context.get("task_type")
        charts = directives.get("charts", [])
        logger.info(
            "VisualizationAgent.run() | goal=%r task_type=%s charts=%s",
            context.get("goal"), task_type, charts,
        )

        # ── Stub output ───────────────────────────────────────────────────────
        # Module 2 wires goal-conditioning through: the agent echoes the chart
        # types selected for this task type. Module 4 replaces the stub with
        # real Vega-Lite / Recharts specs.
        return {
            "task_type": task_type,
            "planned_charts": charts,   # directives from the orchestrator
            "viz_specs": [],            # TODO(M4): list of {type, title, x, y, color, …}
            "message": (
                f"[STUB] VisualizationAgent conditioned for '{task_type}' — "
                f"would render: {', '.join(charts) or 'none'}."
            ),
        }
