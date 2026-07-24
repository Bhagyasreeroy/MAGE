"""
agents/planner.py
──────────────────
PipelinePlanner — builds a *conditional* analysis pipeline from a goal's task
type (Module 2).

This is where MAGE's core claim lives: the goal conditions **which
computations are prioritized and executed** (FR-02). Rather than running a
fixed checklist, the planner emits an ordered list of ``PlannedStep``s, each
naming a specialist agent and the *directives* telling it which computations to
run. Two different goals on the same dataset therefore produce two different
plans — the same-dataset/different-goal design the project is evaluated on.

The specialist agents (mining in M3, visualization in M4) consume these
directives; in Module 2 they accept and echo them, making the conditioning
visible and testable before those modules are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.schemas.analysis import TaskType

# Specialist agent identifiers used across the pipeline.
INGESTION = "IngestionAgent"
MINING = "MiningAgent"
VISUALIZATION = "VisualizationAgent"
RECOMMENDATION = "RecommendationAgent"


@dataclass
class PlannedStep:
    """One step in the conditional pipeline."""

    agent_name: str
    directives: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


# Per-task-type computation directives. Ingestion and recommendation bracket
# every pipeline; mining and visualization are conditioned on the task type.
_MINING_DIRECTIVES: dict[TaskType, dict[str, Any]] = {
    TaskType.classification: {
        "computations": ["class_balance", "feature_importance", "correlation"],
        "priority": "feature_importance",
    },
    TaskType.regression: {
        "computations": ["correlation", "feature_importance", "linearity_check"],
        "priority": "correlation",
    },
    TaskType.clustering: {
        "computations": ["standardize", "kmeans", "dbscan", "silhouette"],
        "priority": "kmeans",
    },
    TaskType.anomaly_detection: {
        "computations": ["iqr_outliers", "isolation_forest", "distribution_tails"],
        "priority": "isolation_forest",
    },
    TaskType.reporting: {
        "computations": ["descriptive_profile", "missingness", "distribution"],
        "priority": "descriptive_profile",
    },
}

_VIZ_DIRECTIVES: dict[TaskType, dict[str, Any]] = {
    TaskType.classification: {"charts": ["grouped_bar", "box_by_class"]},
    TaskType.regression: {"charts": ["scatter", "correlation_heatmap"]},
    TaskType.clustering: {"charts": ["cluster_scatter", "pairplot"]},
    TaskType.anomaly_detection: {"charts": ["box", "highlighted_scatter"]},
    TaskType.reporting: {"charts": ["histograms", "missingness_matrix"]},
}


class PipelinePlanner:
    """Constructs a conditional pipeline plan from a task type."""

    def build_plan(
        self,
        task_type: TaskType,
        target_column: str | None = None,
        dataset_schema: Any = None,
    ) -> list[PlannedStep]:
        """
        Build the ordered conditional pipeline for ``task_type``.

        Parameters
        ----------
        task_type : TaskType
            The classified analytical task driving the plan.
        target_column : str | None
            Target/label column, when the goal implies one; threaded into
            mining directives so supervised computations know their target.
        dataset_schema : IngestionResult | dict | None
            Reserved for future schema-aware refinements (e.g. skipping
            clustering when there are no numeric columns).

        Returns
        -------
        list[PlannedStep]
            Ingestion → Mining (conditioned) → [Visualization (conditioned)]
            → Recommendation.
        """
        mining_directives = dict(_MINING_DIRECTIVES.get(task_type, {}))
        if target_column:
            mining_directives["target_column"] = target_column

        steps: list[PlannedStep] = [
            PlannedStep(
                agent_name=INGESTION,
                directives={},
                reason="Ingest and profile the dataset into a canonical schema.",
            ),
            PlannedStep(
                agent_name=MINING,
                directives={"task_type": task_type.value, **mining_directives},
                reason=(
                    f"Run {task_type.value}-specific computations "
                    f"({', '.join(mining_directives.get('computations', []))})."
                ),
            ),
        ]

        # Every task type — including reporting — gets a conditioned
        # visualization step; the chart set itself is what varies by goal.
        steps.append(
            PlannedStep(
                agent_name=VISUALIZATION,
                directives={"task_type": task_type.value, **_VIZ_DIRECTIVES.get(task_type, {})},
                reason=f"Generate {task_type.value}-appropriate charts.",
            )
        )

        steps.append(
            PlannedStep(
                agent_name=RECOMMENDATION,
                directives={"task_type": task_type.value},
                reason="Ground findings in retrievable methodology and rank recommendations.",
            )
        )
        return steps
