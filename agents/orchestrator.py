"""
agents/orchestrator.py
───────────────────────
OrchestratorAgent — the goal-conditioning and coordination core of MAGE
(Module 2).

Responsibilities:
    • Classify the user's natural-language goal into a task type
      (classification / regression / clustering / anomaly_detection / reporting).
    • Build a *conditional* analysis pipeline tailored to that task type — the
      goal decides which computations run, not just which results are shown.
    • Drive a ReAct-style loop (Reason → Act → Observe → repeat) with a hard
      step cap to prevent runaway execution.
    • Route each step to the appropriate specialist agent, passing it the
      directives that condition its computations.
    • Maintain session state across steps and aggregate the outputs, along with
      a full, inspectable Reason-Act-Observe trail (the explainability log).

The orchestrator is the only agent aware of the others; specialists are
deliberately isolated from one another.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any

from agents.goal_classifier import GoalClassifier, _extract_columns, _find_target_column
from agents.ingestion_agent import IngestionAgent
from agents.mining_agent import MiningAgent
from agents.planner import INGESTION, MINING, RECOMMENDATION, VISUALIZATION, PipelinePlanner
from agents.recommendation_agent import RecommendationAgent
from agents.visualization_agent import VisualizationAgent
from backend.schemas.analysis import GoalClassification, TaskType

logger = logging.getLogger(__name__)

# Upper bound on total ReAct steps actually emitted (not on planned entries —
# see `len(steps) >= MAX_REACT_STEPS` below).
#
# PipelinePlanner still always emits exactly four planned entries, but the run
# is no longer just those four: `_reflect_on_mining` below inspects what
# Mining actually found and can insert 1-3 extra "OrchestratorAgent / reflect"
# steps that change what Visualization and Recommendation do next (see their
# docstrings). A normal run is therefore 4-7 steps depending on the data, not
# a compile-time constant — this bound now guards a loop whose length is
# genuinely data-dependent. It still won't fire in ordinary operation (7 < 10
# even when every trigger fires at once), which is expected, not evidence the
# guard is unreachable: tests/agents/test_react_step_cap.py proves it does
# truncate and log when a plan runs away regardless.
MAX_REACT_STEPS = 10

# ── Reflection thresholds ────────────────────────────────────────────────────
# Read literally: below this, KMeans's own silhouette score calls its fit
# "weak" (common rule-of-thumb bands: <0.25 weak, 0.25-0.5 reasonable
# structure, >0.5 strong). DBSCAN is already computed unconditionally for
# every clustering goal, so when KMeans is weak the orchestrator considers
# switching to it rather than silently presenting a poor k-means fit.
WEAK_SILHOUETTE_THRESHOLD = 0.35
# Below this train R²/accuracy, a SHAP/permutation attribution ranking is
# describing a model that can't predict the target, not the target itself.
# Caveat, stated rather than hidden: this is an absolute cutoff, not one
# relative to a baseline (e.g. majority-class accuracy for an imbalanced
# classification target) — a classifier at 0.3 accuracy over 10 balanced
# classes is unremarkable, the same 0.3 over 2 classes is barely above chance.
# A follow-up could compare against that baseline instead of a fixed number.
LOW_ATTRIBUTION_MODEL_SCORE = 0.3
# Below this completeness, downstream modelling on the target (attribution,
# classification, regression) is being fit on a shrunken, possibly biased
# subset of rows — worth flagging ahead of whatever that modelling found.
LOW_TARGET_COMPLETENESS_PCT = 70.0

# FR-04 — which recommendation register each expertise level reads.
# ExpertiseLevel value → RecommendationAgent field, mapping the system's enum
# onto the three audiences the proposal names:
#   beginner     → Beginner        (plain language, no jargon)
#   intermediate → Analyst         (finding + the statistic behind it)
#   expert       → Data Scientist  (full methodology, markdown intact)
_REGISTER_BY_EXPERTISE: dict[str, str] = {
    "beginner": "text_plain",
    "intermediate": "text_analyst",
    "expert": "text_technical",
}


class OrchestratorAgent:
    """Goal-conditioned orchestrator that drives the full MAGE EDA pipeline."""

    def __init__(
        self,
        classifier: GoalClassifier | None = None,
        planner: PipelinePlanner | None = None,
    ) -> None:
        self._classifier = classifier or GoalClassifier()
        self._planner = planner or PipelinePlanner()

        # Specialist registry, keyed by the names the planner emits.
        self._agents: dict[str, Any] = {
            INGESTION: IngestionAgent(),
            MINING: MiningAgent(),
            "VisualizationAgent": VisualizationAgent(),
            "RecommendationAgent": RecommendationAgent(),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        goal: str,
        expertise_level: str = "intermediate",
        data: dict[str, Any] | None = None,
        mode: str = "rag",
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the goal-conditioned, ReAct-style EDA pipeline.

        Parameters
        ----------
        goal : str
            Natural-language analytical goal from the user.
        expertise_level : str
            One of "beginner", "intermediate", "expert".
        data : dict, optional
            Data payload forwarded to agents (e.g. ``{"source": <file>}``).
        mode : str
            "rag" (default, grounded/cited) or "llm" (freeform Gemini
            response) — read by RecommendationAgent only.
        on_step : callable, optional
            Invoked with each step dict immediately after it is appended to the
            log, enabling live streaming of the Reason/Act/Observe trail while
            the pipeline is still running. Defaults to ``None``, in which case
            behaviour is byte-for-byte identical to a run without it.

            The callback is **advisory**: exceptions raised by it are logged
            and swallowed. A disconnected WebSocket client must not be able to
            abort an analysis that is already underway.

        Returns
        -------
        dict
            Aggregated result: task_type, classification, ReAct step log,
            recommendations, rag_sources, and a summary.
        """
        logger.info("OrchestratorAgent.run() | goal=%r expertise=%s mode=%s", goal, expertise_level, mode)

        context: dict[str, Any] = {
            "goal": goal,
            "expertise_level": expertise_level,
            "data": data or {},
            "mode": mode,
        }

        # Prior-run memory arrives on the data payload (the service layer owns
        # the database) but is consumed by the RecommendationAgent as ordinary
        # context, so lift it to the top level here rather than making every
        # agent reach into `data` for it.
        prior_runs = (data or {}).get("prior_runs")
        if prior_runs:
            context["prior_runs"] = prior_runs

        # 1. Classify the goal → task type. (Rules work on the goal text alone;
        #    the target column is refined once ingestion reveals the schema.)
        classification = self._classifier.classify(goal)
        context["task_type"] = classification.task_type.value

        # 2. Build the conditional pipeline for this task type.
        plan = self._planner.build_plan(classification.task_type, classification.target_column)

        # 3. ReAct loop.
        steps: list[dict[str, Any]] = []

        def emit(step: dict[str, Any]) -> None:
            """Append a step to the log and, if streaming, publish it."""
            steps.append(step)
            if on_step is None:
                return
            try:
                on_step(step)
            except Exception:  # noqa: BLE001 - a listener must never break the pipeline
                logger.warning("on_step callback raised; continuing run.", exc_info=True)

        for planned in plan:
            if len(steps) >= MAX_REACT_STEPS:
                logger.warning("ReAct step limit (%d) reached — halting.", MAX_REACT_STEPS)
                break

            agent = self._agents.get(planned.agent_name)
            if agent is None:
                emit(
                    self._step(planned.agent_name, "skip", planned.reason,
                               f"No agent registered for {planned.agent_name}.", "skipped", 0, {})
                )
                continue

            # REASON — the planner's rationale for running this agent now.
            reasoning = planned.reason

            # Condition the agent: inject this step's directives (and, for
            # mining, the target column detected from the ingested schema).
            directives = dict(planned.directives)
            if planned.agent_name == MINING and not directives.get("target_column"):
                detected = context.get("detected_target_column")
                if detected:
                    directives["target_column"] = detected
            # Visualization and Recommendation additionally read whatever the
            # post-Mining reflection decided (see _reflect_on_mining) — empty
            # unless a reflection step actually fired for this run.
            if planned.agent_name in (VISUALIZATION, RECOMMENDATION):
                directives.update(context.get("mining_reflection", {}))
            context["directives"] = directives

            # ACT — invoke the specialist, timing the call.
            t0 = perf_counter()
            try:
                output = agent.run(context=context)
                status = "success"
            except Exception as exc:  # noqa: BLE001 - one agent must not crash the pipeline
                logger.exception("Agent %s raised: %s", planned.agent_name, exc)
                output = {"error": str(exc)}
                status = "error"
            latency_ms = int((perf_counter() - t0) * 1000)

            output_dict = output.model_dump() if hasattr(output, "model_dump") else output

            # OBSERVE — record the step and update session state.
            observation = self._summarize(planned.agent_name, output_dict, status)
            emit(
                self._step(
                    planned.agent_name,
                    f"run {planned.agent_name} with directives {list(directives.keys())}",
                    reasoning, observation, status, latency_ms, output_dict,
                )
            )
            context[f"{planned.agent_name}_output"] = output

            # Post-ingestion: refine the target column from the real schema.
            if planned.agent_name == INGESTION and status == "success":
                columns = _extract_columns(output)
                detected = _find_target_column(goal, columns)
                if detected:
                    context["detected_target_column"] = detected
                    if not classification.target_column:
                        classification.target_column = detected

            # OBSERVE → RE-PLAN — react to what Mining actually found before
            # Visualization/Recommendation run. This is the loop's one real
            # branch point: zero to three extra steps, depending on the data.
            if planned.agent_name == MINING and status == "success":
                target_column = classification.target_column or context.get("detected_target_column")
                for reflection_step in self._reflect_on_mining(output_dict, context, target_column):
                    emit(reflection_step)

        return self._aggregate(goal, expertise_level, mode, classification, steps, context)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reflect_on_mining(
        self, mining_output: dict[str, Any], context: dict[str, Any], target_column: str | None,
    ) -> list[dict[str, Any]]:
        """
        Inspect what Mining actually found and decide whether it changes what
        gets presented next — the loop's observe-and-replan point.

        Three independent checks, each against a signal MiningAgent already
        computes (silhouette score, SHAP/permutation model_score, target
        completeness). Any that fire append one synthetic
        ``"OrchestratorAgent" / "reflect"`` step naming the observation and the
        decision it caused, and record the decision in
        ``context["mining_reflection"]`` — merged into the directives for the
        Visualization and Recommendation steps that follow (see `run()`), so
        the decision actually changes what those agents do rather than only
        being logged. Returns ``[]``, and leaves ``mining_reflection`` empty,
        when nothing fires — the common case on clean data.

        Reflection steps are tagged "OrchestratorAgent", not "MiningAgent", so
        `orchestrator_service._remember()`'s `next(... agent_name == MINING)`
        lookup keeps resolving to the real Mining step's output.
        """
        reflection: dict[str, Any] = {}
        emitted: list[dict[str, Any]] = []

        clustering = mining_output.get("clustering")
        if clustering and clustering.get("silhouette_score") is not None:
            score = clustering["silhouette_score"]
            if score < WEAK_SILHOUETTE_THRESHOLD:
                dbscan = mining_output.get("dbscan")
                if dbscan and dbscan.get("n_clusters", 0) >= 2 and dbscan.get("points"):
                    reflection["preferred_clustering"] = "dbscan"
                    emitted.append(self._step(
                        "OrchestratorAgent", "reflect",
                        f"KMeans silhouette score {score} is weak (<{WEAK_SILHOUETTE_THRESHOLD}) — "
                        "the k-means partition doesn't separate this data well.",
                        f"Switching to DBSCAN as the reported clustering result: "
                        f"{dbscan['n_clusters']} density-based cluster(s), {dbscan['n_noise']} noise point(s).",
                        "success", 0,
                        {"decision": "prefer_dbscan", "kmeans_silhouette": score, "dbscan": dbscan},
                    ))
                else:
                    reflection["preferred_clustering"] = "kmeans"
                    emitted.append(self._step(
                        "OrchestratorAgent", "reflect",
                        f"KMeans silhouette score {score} is weak (<{WEAK_SILHOUETTE_THRESHOLD}), and "
                        "DBSCAN didn't find a usable alternative (fewer than 2 clusters).",
                        "Reporting the KMeans result with an explicit weak-fit caveat rather than "
                        "presenting it as a confident finding.",
                        "success", 0,
                        {"decision": "flag_weak_clustering", "kmeans_silhouette": score},
                    ))

        attribution = mining_output.get("feature_attribution") or {}
        model_score = attribution.get("model_score")
        if model_score is not None:
            if model_score < LOW_ATTRIBUTION_MODEL_SCORE:
                reflection["attribution_trusted"] = False
                emitted.append(self._step(
                    "OrchestratorAgent", "reflect",
                    f"The feature-attribution model scored {model_score} predicting "
                    f"'{attribution.get('target')}' — below {LOW_ATTRIBUTION_MODEL_SCORE}, too low to "
                    "trust which feature the ranking says matters most.",
                    "Marking this attribution as low-confidence rather than surfacing it as a top finding.",
                    "success", 0,
                    {"decision": "distrust_attribution", "model_score": model_score},
                ))
            else:
                reflection["attribution_trusted"] = True

        if target_column:
            quality = (mining_output.get("data_quality") or {}).get(target_column)
            if quality is not None:
                completeness = quality.get("completeness_pct", 100.0)
                if completeness < LOW_TARGET_COMPLETENESS_PCT:
                    reflection["target_quality_warning"] = (
                        f"Target column '{target_column}' is only {completeness}% complete "
                        f"({quality.get('missing_count', 0)} missing) — downstream modelling on it "
                        "is unreliable until this is addressed."
                    )
                    emitted.append(self._step(
                        "OrchestratorAgent", "reflect",
                        f"Target '{target_column}' is {round(100 - completeness, 1)}% missing — "
                        f"below the {100 - LOW_TARGET_COMPLETENESS_PCT:.0f}% missingness this run "
                        "treats as trustworthy for modelling.",
                        "Leading with a data-quality recommendation for this target ahead of "
                        "whatever attribution/classification found on it.",
                        "success", 0,
                        {"decision": "target_quality_warning", "completeness_pct": completeness},
                    ))

        context["mining_reflection"] = reflection
        return emitted

    @staticmethod
    def _step(
        agent_name: str, action: str, reasoning: str, observation: str,
        status: str, latency_ms: int, output: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a ReActStep-shaped dict for the step log."""
        return {
            "agent_name": agent_name,
            "action": action,
            "reasoning": reasoning,
            "observation": observation,
            "status": status,
            "latency_ms": latency_ms,
            "output": output,
        }

    @staticmethod
    def _summarize(agent_name: str, output: dict[str, Any], status: str) -> str:
        """Produce a short human-readable observation of a step's result."""
        if status == "error":
            return f"{agent_name} failed: {output.get('error', 'unknown error')}."
        if agent_name == INGESTION:
            rows, cols = output.get("row_count"), output.get("column_count")
            if rows is not None:
                return f"Ingested {rows} rows × {cols} columns."
        if agent_name == "RecommendationAgent":
            recs = output.get("recommendations", [])
            return f"Produced {len(recs)} grounded recommendation(s)."
        msg = output.get("message")
        return msg if isinstance(msg, str) else f"{agent_name} completed."

    def _aggregate(
        self,
        goal: str,
        expertise_level: str,
        mode: str,
        classification: GoalClassification,
        steps: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine specialist outputs into a unified, goal-conditioned result."""
        recommendation_output = context.get("RecommendationAgent_output") or {}
        if hasattr(recommendation_output, "model_dump"):
            recommendation_output = recommendation_output.model_dump()
        structured_recs = recommendation_output.get("recommendations", [])

        # FR-04 — the reader's declared expertise selects the register. Three
        # levels, three registers; the enum values are the system's, the names
        # in comments are the spec's.
        text_field = _REGISTER_BY_EXPERTISE.get(expertise_level, "text_technical")
        recommendations = [
            # Fall back down the registers rather than to the bare insight, so
            # an older persisted run written before `text_analyst` existed
            # still renders prose instead of a one-line title.
            rec.get(text_field) or rec.get("text_technical") or rec.get("insight", "")
            for rec in structured_recs
        ]
        rag_sources = recommendation_output.get("rag_sources", [])

        return {
            "goal": goal,
            "expertise_level": expertise_level,
            "mode": mode,
            "task_type": classification.task_type.value,
            "classification": classification.model_dump(),
            "steps": steps,
            "recommendations": recommendations,
            "rag_sources": rag_sources,
            "summary": (
                f"Goal classified as '{classification.task_type.value}'. "
                f"Ran a conditional pipeline of {len(steps)} step(s), "
                f"grounding {len(recommendations)} recommendation(s) in {len(rag_sources)} source(s)."
            ),
        }
