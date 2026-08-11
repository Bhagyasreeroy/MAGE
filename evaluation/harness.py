"""
evaluation/harness.py
──────────────────────
Runs the same-dataset / different-goal experiment (Objective 7).

For every dataset in the pinned suite, the harness pushes the *same* file
through ``OrchestratorAgent`` once per goal — one goal per task type — and
records what each run actually did: the computations that executed, the charts
selected, the recommendations produced, the sources cited, and the wall-clock
time taken. The generic AutoEDA baseline is recorded for the same
(dataset, goal) pairs.

Because the dataset is held constant and only the goal varies, any difference
between runs is attributable to goal-conditioning and nothing else. The
baseline, having no way to receive a goal, produces an identical computation
set every time — which is the comparison.

Run it:
    PYTHONPATH=. python -m evaluation.harness
    PYTHONPATH=. python -m evaluation.harness --live-baseline
    PYTHONPATH=. python -m evaluation.harness --datasets iris,wine
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from agents.orchestrator import OrchestratorAgent
from evaluation import baseline as baseline_mod
from evaluation.datasets import DATA_DIR, EvalDataset, load_eval_datasets
from evaluation.metrics import (
    citation_coverage,
    f1,
    mean_pairwise_divergence,
    pairwise_divergence_matrix,
    task_relevant_precision,
    task_relevant_recall,
)

logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

# One goal per task type. Phrased the way a user would actually phrase them —
# not as keyword bait — so the classifier is exercised rather than spoon-fed.
# `{target}` is filled from the dataset's target column when it has one.
GOAL_TEMPLATES: list[tuple[str, str]] = [
    ("classification", "Predict which category each record falls into based on {target}"),
    ("regression", "Understand what drives {target} and estimate its value from the other fields"),
    ("clustering", "Find natural groupings and segments hidden in this data"),
    ("anomaly_detection", "Identify unusual or suspicious records that don't fit the pattern"),
    ("reporting", "Give me a general overview of this dataset and its data quality"),
]

# Fallback phrasing for the supervised goals when a dataset has no target.
_NO_TARGET_FALLBACK: dict[str, str] = {
    "classification": "Predict which category each record belongs to",
    "regression": "Model the numeric outcome in this data and find what drives it",
}


@dataclass
class RunRecord:
    """One (dataset × goal) execution of the MAGE pipeline, plus its baseline."""

    dataset: str
    goal: str
    intended_task_type: str
    classified_task_type: str
    computations_run: list[str]
    chart_types: list[str]
    recommendation_count: int
    cited_source_count: int
    citation_coverage: float
    precision: float
    recall: float
    f1: float
    runtime_seconds: float
    row_count: int
    column_count: int
    status: str = "success"
    error: str | None = None
    baseline_computations: list[str] = field(default_factory=list)
    baseline_precision: float = 0.0
    baseline_recall: float = 0.0
    baseline_f1: float = 0.0


# ── Extraction from the orchestrator's result ────────────────────────────────


def _step_output(result: dict[str, Any], agent_name: str) -> dict[str, Any]:
    """
    Pull one agent's raw output out of the Reason/Act/Observe step log.

    The orchestrator's aggregate flattens recommendations into display strings
    and drops mining internals, so the step log is the only place the evidence
    fields survive.
    """
    for step in result.get("steps", []):
        if step.get("agent_name") == agent_name:
            output = step.get("output") or {}
            return output if isinstance(output, dict) else {}
    return {}


def _extract_computations(result: dict[str, Any]) -> list[str]:
    """
    The computations the MiningAgent actually executed.

    ``MiningAgent`` reports the literal ``"<default profile>"`` when it runs
    unconditioned. That placeholder is expanded to the real token set here, so
    a conditioned and an unconditioned run remain comparable — otherwise the
    Jaccard measure would treat the placeholder as a single exotic computation
    and manufacture divergence that did not happen.
    """
    mining = _step_output(result, "MiningAgent")
    computations = [str(c) for c in mining.get("computations_run", [])]
    if computations == ["<default profile>"]:
        return sorted(
            {"descriptive_profile", "missingness", "correlation", "iqr_outliers",
             "feature_importance", "kmeans", "silhouette", "standardize"}
        )
    return sorted(computations)


def _extract_chart_types(result: dict[str, Any]) -> list[str]:
    """Distinct chart types the VisualizationAgent emitted specs for."""
    viz = _step_output(result, "VisualizationAgent")
    specs = viz.get("viz_specs", []) or []
    return sorted({str(spec.get("type")) for spec in specs if spec.get("type")})


def _extract_recommendations(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Structured recommendation records, which retain their ``sources`` lists."""
    rec = _step_output(result, "RecommendationAgent")
    recs = rec.get("recommendations", []) or []
    return [r for r in recs if isinstance(r, dict)]


def _build_goal(task_type: str, dataset: EvalDataset) -> str:
    """Fill the goal template for this dataset, falling back when no target exists."""
    template = dict(GOAL_TEMPLATES)[task_type]
    if "{target}" not in template:
        return template
    if dataset.target_column:
        return template.format(target=dataset.target_column)
    return _NO_TARGET_FALLBACK.get(task_type, template.format(target="the outcome"))


# ── Execution ────────────────────────────────────────────────────────────────


def run_one(
    orchestrator: OrchestratorAgent,
    dataset: EvalDataset,
    csv_path: Path,
    task_type: str,
    live_baseline: bool,
) -> RunRecord:
    """
    Execute a single (dataset, goal) pair and score it.

    Failures are captured into the record rather than raised: one bad pair must
    not destroy a full evaluation sweep, and a zero-scored failure is more
    honest in the results table than a missing row.
    """
    goal = _build_goal(task_type, dataset)
    logger.info("▶ %s × %s | %r", dataset.name, task_type, goal)

    t0 = perf_counter()
    try:
        result = orchestrator.run(goal=goal, expertise_level="intermediate",
                                  data={"source": str(csv_path)})
        status, error = "success", None
    except Exception as exc:  # noqa: BLE001 - a failed pair is data, not a crash
        logger.exception("Run failed for %s × %s", dataset.name, task_type)
        result, status, error = {}, "error", str(exc)
    runtime = perf_counter() - t0

    computations = _extract_computations(result)
    charts = _extract_chart_types(result)
    recommendations = _extract_recommendations(result)
    classified = str(result.get("task_type", "")) or "unknown"

    ingestion = _step_output(result, "IngestionAgent")

    # Baseline, scored against the *same* classified task type so the
    # comparison is like-for-like.
    if live_baseline and baseline_mod.ydata_profiling_available():
        import pandas as pd

        base_computations = baseline_mod.profile_computations(pd.read_csv(csv_path), goal=goal)
    else:
        base_computations = baseline_mod.baseline_computations(goal=goal)

    precision = task_relevant_precision(computations, classified)
    recall = task_relevant_recall(computations, classified)
    base_precision = task_relevant_precision(base_computations, classified)
    base_recall = task_relevant_recall(base_computations, classified)

    return RunRecord(
        dataset=dataset.name,
        goal=goal,
        intended_task_type=task_type,
        classified_task_type=classified,
        computations_run=computations,
        chart_types=charts,
        recommendation_count=len(recommendations),
        cited_source_count=len(result.get("rag_sources", [])),
        citation_coverage=round(citation_coverage(recommendations), 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1(precision, recall), 4),
        runtime_seconds=round(runtime, 3),
        row_count=int(ingestion.get("row_count") or 0),
        column_count=int(ingestion.get("column_count") or 0),
        status=status,
        error=error,
        baseline_computations=sorted(base_computations),
        baseline_precision=round(base_precision, 4),
        baseline_recall=round(base_recall, 4),
        baseline_f1=round(f1(base_precision, base_recall), 4),
    )


def _warm_up(orchestrator: OrchestratorAgent, dataset: EvalDataset) -> None:
    """
    Run one discarded pipeline pass before timing begins.

    The sentence-transformer embedding model is loaded lazily on first
    retrieval, costing ~15s once per process. Left in place, that one-off load
    lands entirely on whichever run happens to go first and is then reported as
    the FR-05 worst case — turning a startup cost into an apparent analysis
    cost roughly fifty times the true figure. Paying it here means every timed
    run measures analysis, and the FR-05 number reflects steady-state
    behaviour, which is what the requirement is about.
    """
    logger.info("Warming up (loading embedding model) — this run is discarded.")
    t0 = perf_counter()
    try:
        orchestrator.run(
            goal="Give me a general overview of this dataset",
            expertise_level="intermediate",
            data={"source": str(dataset.materialize())},
        )
    except Exception:  # noqa: BLE001 - a failed warm-up must not abort the sweep
        logger.warning("Warm-up run failed; timings may include model-load cost.", exc_info=True)
    logger.info("Warm-up complete in %.2fs.", perf_counter() - t0)


def run_harness(
    dataset_filter: list[str] | None = None,
    live_baseline: bool = False,
) -> dict[str, Any]:
    """
    Execute the full experiment and return the complete results payload.

    Parameters
    ----------
    dataset_filter : list[str], optional
        Restrict the sweep to these dataset names. ``None`` runs the suite.
    live_baseline : bool
        Derive the baseline by actually running ydata-profiling rather than
        using the pinned declarative spec.

    Returns
    -------
    dict
        The full results payload, also written to ``evaluation/results.json``.
    """
    datasets = load_eval_datasets()
    if dataset_filter:
        wanted = {d.strip() for d in dataset_filter}
        datasets = [d for d in datasets if d.name in wanted]
        if not datasets:
            raise SystemExit(f"No datasets matched {sorted(wanted)}.")

    if live_baseline and not baseline_mod.ydata_profiling_available():
        logger.warning(
            "--live-baseline requested but ydata-profiling is not installed; "
            "falling back to the pinned declarative spec."
        )
        live_baseline = False

    orchestrator = OrchestratorAgent()
    _warm_up(orchestrator, datasets[0])

    records: list[RunRecord] = []

    for dataset in datasets:
        csv_path = dataset.materialize()
        for task_type, _ in GOAL_TEMPLATES:
            records.append(run_one(orchestrator, dataset, csv_path, task_type, live_baseline))

    return _assemble(records, datasets, live_baseline)


def _assemble(
    records: list[RunRecord], datasets: list[EvalDataset], live_baseline: bool
) -> dict[str, Any]:
    """Aggregate per-run records into the per-dataset and overall summaries."""
    per_dataset: list[dict[str, Any]] = []

    for dataset in datasets:
        rows = [r for r in records if r.dataset == dataset.name]
        if not rows:
            continue

        by_goal_comp = {r.intended_task_type: set(r.computations_run) for r in rows}
        by_goal_chart = {r.intended_task_type: set(r.chart_types) for r in rows}
        base_sets = [set(r.baseline_computations) for r in rows]

        per_dataset.append(
            {
                "dataset": dataset.name,
                "description": dataset.description,
                "notes": dataset.notes,
                "row_count": rows[0].row_count,
                "column_count": rows[0].column_count,
                "mage_computation_divergence": round(
                    mean_pairwise_divergence(list(by_goal_comp.values())), 4
                ),
                "baseline_computation_divergence": round(
                    mean_pairwise_divergence(base_sets), 4
                ),
                "mage_chart_divergence": round(
                    mean_pairwise_divergence(list(by_goal_chart.values())), 4
                ),
                "computation_pairs": pairwise_divergence_matrix(by_goal_comp),
                "chart_pairs": pairwise_divergence_matrix(by_goal_chart),
                "mean_precision": round(sum(r.precision for r in rows) / len(rows), 4),
                "mean_recall": round(sum(r.recall for r in rows) / len(rows), 4),
                "mean_baseline_precision": round(
                    sum(r.baseline_precision for r in rows) / len(rows), 4
                ),
                "mean_baseline_recall": round(
                    sum(r.baseline_recall for r in rows) / len(rows), 4
                ),
                "mean_citation_coverage": round(
                    sum(r.citation_coverage for r in rows) / len(rows), 4
                ),
                "max_runtime_seconds": round(max(r.runtime_seconds for r in rows), 3),
            }
        )

    n = len(records) or 1
    ok = [r for r in records if r.status == "success"]
    classified_correctly = sum(1 for r in records if r.classified_task_type == r.intended_task_type)

    summary = {
        "total_runs": len(records),
        "successful_runs": len(ok),
        "mage_computation_divergence": round(
            sum(d["mage_computation_divergence"] for d in per_dataset) / (len(per_dataset) or 1), 4
        ),
        "baseline_computation_divergence": round(
            sum(d["baseline_computation_divergence"] for d in per_dataset)
            / (len(per_dataset) or 1),
            4,
        ),
        "mage_chart_divergence": round(
            sum(d["mage_chart_divergence"] for d in per_dataset) / (len(per_dataset) or 1), 4
        ),
        "mean_precision": round(sum(r.precision for r in records) / n, 4),
        "mean_recall": round(sum(r.recall for r in records) / n, 4),
        "mean_baseline_precision": round(sum(r.baseline_precision for r in records) / n, 4),
        "mean_baseline_recall": round(sum(r.baseline_recall for r in records) / n, 4),
        "mean_f1": round(sum(r.f1 for r in records) / n, 4),
        "mean_baseline_f1": round(sum(r.baseline_f1 for r in records) / n, 4),
        "mean_citation_coverage": round(sum(r.citation_coverage for r in records) / n, 4),
        "goal_classification_accuracy": round(classified_correctly / n, 4),
        # FR-05: end-to-end under 60s. Reported as the worst case, not the mean.
        "max_runtime_seconds": round(max((r.runtime_seconds for r in records), default=0.0), 3),
        "mean_runtime_seconds": round(sum(r.runtime_seconds for r in records) / n, 3),
        "fr05_under_60s": max((r.runtime_seconds for r in records), default=0.0) < 60.0,
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "baseline_mode": "live-ydata-profiling" if live_baseline else "declarative-spec",
        },
        "experiment": {
            "design": "same-dataset / different-goal",
            "datasets": [d.name for d in datasets],
            "goals_per_dataset": [t for t, _ in GOAL_TEMPLATES],
            "data_dir": str(DATA_DIR),
        },
        "summary": summary,
        "per_dataset": per_dataset,
        "runs": [asdict(r) for r in records],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Writes results.json and the markdown report."""
    parser = argparse.ArgumentParser(description="Run the MAGE evaluation harness.")
    parser.add_argument(
        "--datasets",
        help="Comma-separated dataset names to restrict the sweep to.",
    )
    parser.add_argument(
        "--live-baseline",
        action="store_true",
        help="Derive the baseline by running ydata-profiling instead of the pinned spec.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s | %(message)s",
    )

    results = run_harness(
        dataset_filter=args.datasets.split(",") if args.datasets else None,
        live_baseline=args.live_baseline,
    )

    # Imported here to keep report generation out of the import graph of the
    # metric modules, which the tests import directly.
    from evaluation.report import write_reports

    json_path, md_path = write_reports(results)

    s = results["summary"]
    print(f"\nRuns: {s['successful_runs']}/{s['total_runs']} successful")
    print(f"Computation divergence — MAGE {s['mage_computation_divergence']:.3f} "
          f"vs baseline {s['baseline_computation_divergence']:.3f}")
    print(f"Chart divergence       — MAGE {s['mage_chart_divergence']:.3f}")
    print(f"Task-relevant precision — MAGE {s['mean_precision']:.3f} "
          f"vs baseline {s['mean_baseline_precision']:.3f}")
    print(f"Citation coverage       — {s['mean_citation_coverage']:.3f}")
    print(f"FR-05 (<60s)            — {'PASS' if s['fr05_under_60s'] else 'FAIL'} "
          f"(max {s['max_runtime_seconds']:.2f}s)")
    print(f"\nWrote {json_path}\n      {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
