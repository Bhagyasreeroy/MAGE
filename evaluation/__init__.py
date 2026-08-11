"""
evaluation/
────────────
Empirical evaluation harness for MAGE (Objective 7).

Implements the proposal's *same-dataset / different-goal* experiment: hold the
dataset constant, vary only the natural-language goal, and measure whether the
set of statistical computations that actually execute changes as a result.

A generic AutoEDA baseline (ydata-profiling) runs an identical, goal-invariant
computation set on every dataset by construction; MAGE conditions its pipeline
on the goal. The headline result is the divergence between the two.

Entry point:
    python -m evaluation.harness
"""

from evaluation.baseline import BASELINE_COMPUTATIONS, baseline_computations
from evaluation.datasets import EvalDataset, load_eval_datasets
from evaluation.metrics import (
    citation_coverage,
    jaccard_distance,
    mean_pairwise_divergence,
    task_relevant_precision,
)

__all__ = [
    "BASELINE_COMPUTATIONS",
    "baseline_computations",
    "EvalDataset",
    "load_eval_datasets",
    "citation_coverage",
    "jaccard_distance",
    "mean_pairwise_divergence",
    "task_relevant_precision",
]
