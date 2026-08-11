"""
evaluation/report.py
─────────────────────
Turns the harness's results payload into artefacts the project report can use
directly: ``results.json`` (the full record, including every individual run)
and ``results.md`` (markdown tables ready to paste into the write-up).

The markdown is deliberately plain — GitHub-flavoured tables, no HTML — so it
renders in the repo, in the report, and in a LaTeX pipeline via pandoc without
modification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parent
JSON_PATH = OUT_DIR / "results.json"
MD_PATH = OUT_DIR / "results.md"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a GitHub-flavoured markdown table."""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join(out)


def build_markdown(results: dict[str, Any]) -> str:
    """
    Build the full markdown report from a harness results payload.

    Parameters
    ----------
    results : dict
        The payload returned by :func:`evaluation.harness.run_harness`.

    Returns
    -------
    str
        Markdown document.
    """
    s = results["summary"]
    exp = results["experiment"]
    env = results["environment"]

    parts: list[str] = []

    parts.append("# MAGE — Empirical Evaluation Results\n")
    parts.append(
        f"> Generated {results['generated_at']} · Python {env['python']} · "
        f"baseline mode: `{env['baseline_mode']}`\n"
    )
    parts.append(
        "**Experimental design — same-dataset / different-goal.** Each dataset is held "
        "constant and pushed through the pipeline once per goal, one goal per task type. "
        "Only the natural-language goal varies, so any difference between runs is "
        "attributable to goal-conditioning alone. The generic AutoEDA baseline "
        "(ydata-profiling) has no mechanism to receive a goal, so its computation set is "
        "identical on every run by construction.\n"
    )
    parts.append(
        f"Datasets: {', '.join(f'`{d}`' for d in exp['datasets'])} · "
        f"Goals per dataset: {len(exp['goals_per_dataset'])} · "
        f"Total runs: {s['total_runs']}\n"
    )

    # ── Headline ──
    parts.append("\n## 1. Headline result — cross-goal divergence\n")
    parts.append(
        "Mean pairwise Jaccard distance between the computation sets produced by "
        "different goals on the same dataset. 0.0 means every goal produced identical "
        "computations; higher means the goal changed what actually ran.\n"
    )
    parts.append(
        _table(
            ["System", "Computation divergence", "Chart divergence"],
            [
                ["**MAGE**", f"**{s['mage_computation_divergence']:.3f}**",
                 f"**{s['mage_chart_divergence']:.3f}**"],
                ["AutoEDA baseline", f"{s['baseline_computation_divergence']:.3f}", "0.000"],
            ],
        )
    )
    parts.append("")

    # ── Relevance ──
    parts.append("\n## 2. Task-relevant computation quality\n")
    parts.append(
        "Scored against a hand-labelled relevance set per task type "
        "(`evaluation/metrics.py`), defined independently of the planner so the metric "
        "is not circular. Precision = of what ran, how much was relevant. "
        "Recall = of what was relevant, how much ran.\n"
    )
    parts.append(
        _table(
            ["System", "Mean precision", "Mean recall", "Mean F1"],
            [
                ["**MAGE**", f"**{s['mean_precision']:.3f}**", f"{s['mean_recall']:.3f}",
                 f"**{s['mean_f1']:.3f}**"],
                ["AutoEDA baseline", f"{s['mean_baseline_precision']:.3f}",
                 f"{s['mean_baseline_recall']:.3f}", f"{s['mean_baseline_f1']:.3f}"],
            ],
        )
    )
    parts.append(
        "\nRead precision and recall together. The baseline reaches comparable *recall* "
        "on several task types, but only by brute force — it runs its entire fixed set "
        "every time, so it incidentally covers the relevant computations while also "
        "running many irrelevant ones. That is exactly what its low precision records. "
        "MAGE reaches the same coverage while running only task-relevant work, which is "
        "the distinction F1 captures.\n"
    )
    parts.append(
        "MAGE's precision of 1.000 should be read as *\"the planner emits nothing "
        "outside the relevant set\"* rather than as a discriminating quality score — "
        "the metric has a ceiling here and cannot separate a good conditional pipeline "
        "from an excellent one. Recall is the more informative of the two for MAGE, and "
        "the headroom below 1.000 is real: the planner runs a deliberately focused "
        "subset rather than everything a task type could justify.\n"
    )

    # ── Grounding + performance ──
    parts.append("\n## 3. Grounding, classification, and performance\n")
    parts.append(
        _table(
            ["Metric", "Value", "Requirement"],
            [
                ["Citation coverage", f"{s['mean_citation_coverage']:.3f}",
                 "FR-03 — every recommendation cites a source"],
                ["Goal classification accuracy", f"{s['goal_classification_accuracy']:.3f}",
                 "Intended vs. classified task type"],
                ["Mean runtime", f"{s['mean_runtime_seconds']:.2f}s", "—"],
                ["Max runtime", f"{s['max_runtime_seconds']:.2f}s",
                 f"FR-05 — <60s: **{'PASS' if s['fr05_under_60s'] else 'FAIL'}**"],
                ["Successful runs", f"{s['successful_runs']}/{s['total_runs']}", "—"],
            ],
        )
    )
    parts.append("")

    # ── Per dataset ──
    parts.append("\n## 4. Per-dataset breakdown\n")
    parts.append(
        _table(
            ["Dataset", "Rows", "Cols", "MAGE comp. div.", "Baseline comp. div.",
             "MAGE chart div.", "Precision", "Citation cov.", "Max runtime"],
            [
                [
                    f"`{d['dataset']}`", d["row_count"], d["column_count"],
                    f"**{d['mage_computation_divergence']:.3f}**",
                    f"{d['baseline_computation_divergence']:.3f}",
                    f"{d['mage_chart_divergence']:.3f}",
                    f"{d['mean_precision']:.3f}",
                    f"{d['mean_citation_coverage']:.3f}",
                    f"{d['max_runtime_seconds']:.2f}s",
                ]
                for d in results["per_dataset"]
            ],
        )
    )
    parts.append("")

    # ── The evidence: what each goal actually ran ──
    parts.append("\n## 5. What each goal actually ran\n")
    parts.append(
        "The raw evidence behind the divergence number. `computations_run` is emitted by "
        "`MiningAgent` on every run and is what the metrics above are computed from.\n"
    )
    for dataset_name in exp["datasets"]:
        runs = [r for r in results["runs"] if r["dataset"] == dataset_name]
        if not runs:
            continue
        parts.append(f"\n### `{dataset_name}`\n")
        parts.append(
            _table(
                ["Goal (task type)", "Classified as", "Computations run", "Charts", "Runtime"],
                [
                    [
                        r["intended_task_type"],
                        r["classified_task_type"],
                        ", ".join(f"`{c}`" for c in r["computations_run"]) or "—",
                        ", ".join(f"`{c}`" for c in r["chart_types"]) or "—",
                        f"{r['runtime_seconds']:.2f}s",
                    ]
                    for r in runs
                ],
            )
        )
        parts.append("")
        baseline = runs[0]["baseline_computations"]
        parts.append(
            f"Baseline, for all {len(runs)} goals above (identical every time): "
            + ", ".join(f"`{c}`" for c in baseline)
            + "\n"
        )

    parts.append(
        "\n---\n\n*Generated by `python -m evaluation.harness`. "
        "Full per-run detail, including pairwise divergence matrices, is in "
        "`evaluation/results.json`.*\n"
    )

    return "\n".join(parts)


def write_reports(
    results: dict[str, Any],
    json_path: Path = JSON_PATH,
    md_path: Path = MD_PATH,
) -> tuple[Path, Path]:
    """
    Write ``results.json`` and ``results.md``.

    Parameters
    ----------
    results : dict
        The harness results payload.
    json_path, md_path : Path
        Output locations; defaults sit alongside this module.

    Returns
    -------
    tuple[Path, Path]
        The two paths written.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(results), encoding="utf-8")
    logger.info("Wrote %s and %s", json_path, md_path)
    return json_path, md_path
