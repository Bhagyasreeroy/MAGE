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
from agents.planner import INGESTION, MINING, PipelinePlanner
from agents.recommendation_agent import RecommendationAgent
from agents.visualization_agent import VisualizationAgent
from backend.schemas.analysis import GoalClassification, TaskType

logger = logging.getLogger(__name__)

# Upper bound on ReAct iterations.
#
# Describe this accurately: it is a working bound that the *current* planner
# cannot reach, not an active safety guard. PipelinePlanner emits exactly four
# steps for every task type, so the check below has never fired in production
# and cannot until the plan becomes variable-length — which is what a
# model-driven loop would make it. The bound is kept rather than deleted
# because it is the thing that makes such a loop safe to introduce, and
# tests/agents/test_react_step_cap.py proves it truncates a runaway plan and
# logs when it does. Do not present it as evidence of a guarded ReAct loop.
MAX_REACT_STEPS = 10

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

        for step_idx, planned in enumerate(plan):
            if step_idx >= MAX_REACT_STEPS:
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

        return self._aggregate(goal, expertise_level, mode, classification, steps, context)

    # ── Internal helpers ──────────────────────────────────────────────────────

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
