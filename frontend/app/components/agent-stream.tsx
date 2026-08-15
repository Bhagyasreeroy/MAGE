'use client';

/**
 * components/agent-stream.tsx
 * ───────────────────────────
 * The live analysis view — MAGE's agents narrating their own work.
 *
 * Renders the Reason/Act/Observe trail as it streams in (FR-06), one card per
 * agent. Pending agents are shown greyed ahead of time so the audience can see
 * what the plan is before it executes, and watch it fill in.
 */

import type { AgentStep, StreamPhase } from '../lib/use-analysis-stream';

/** The agents the planner schedules for every task type, in pipeline order. */
const PIPELINE_AGENTS = [
  { name: 'IngestionAgent', label: 'Ingestion', blurb: 'Loading and profiling your dataset' },
  { name: 'MiningAgent', label: 'Mining', blurb: 'Running the computations your goal selected' },
  { name: 'VisualizationAgent', label: 'Visualization', blurb: 'Choosing charts that fit the task' },
  { name: 'RecommendationAgent', label: 'Recommendations', blurb: 'Grounding advice in the knowledge base' },
];

const CheckIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
  </svg>
);

const AlertIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const Spinner = () => (
  <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
  </svg>
);

function StatusBadge({ status }: { status: AgentStep['status'] }) {
  if (status === 'success') {
    return (
      <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-lavender-light text-navy shrink-0">
        <CheckIcon />
      </span>
    );
  }
  if (status === 'error') {
    return (
      <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-dusty-rose/20 text-dusty-rose shrink-0">
        <AlertIcon />
      </span>
    );
  }
  return (
    <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-cream text-navy/40 text-xs font-bold shrink-0">
      –
    </span>
  );
}

interface AgentStreamProps {
  phase: StreamPhase;
  steps: AgentStep[];
  error: string | null;
}

export function AgentStream({ phase, steps, error }: AgentStreamProps) {
  if (phase === 'idle') return null;

  const completed = new Map(steps.map((step) => [step.agent_name, step]));
  const totalLatency = steps.reduce((sum, step) => sum + (step.latency_ms ?? 0), 0);

  return (
    <div className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8 animate-fade-in">
      <div className="flex items-center justify-between mb-6">
        <div>
          <p className="text-xs font-bold text-navy/60 uppercase tracking-widest mb-1">
            Live Agent Trail
          </p>
          <p className="text-xs text-navy/40 font-light">
            {phase === 'connecting' && 'Connecting to the analysis service…'}
            {phase === 'running' && `${steps.length} of ${PIPELINE_AGENTS.length} agents finished`}
            {phase === 'complete' && `All agents finished in ${totalLatency} ms`}
            {phase === 'error' && 'The run stopped early'}
          </p>
        </div>
        {(phase === 'connecting' || phase === 'running') && (
          <span className="text-navy/40">
            <Spinner />
          </span>
        )}
      </div>

      <ol className="space-y-3">
        {PIPELINE_AGENTS.map(({ name, label, blurb }) => {
          const step = completed.get(name);
          const isPending = !step;
          // The next unfinished agent is the one currently working.
          const isActive =
            isPending &&
            phase === 'running' &&
            PIPELINE_AGENTS.findIndex((a) => !completed.has(a.name)) ===
              PIPELINE_AGENTS.findIndex((a) => a.name === name);

          return (
            <li
              key={name}
              className={`rounded-2xl border p-5 transition-all duration-500 ${
                isPending
                  ? isActive
                    ? 'bg-cream/60 border-lavender/50'
                    : 'bg-cream/20 border-dusty-rose/10 opacity-50'
                  : 'bg-cream/50 border-dusty-rose/20'
              }`}
            >
              <div className="flex items-start gap-4">
                {step ? (
                  <StatusBadge status={step.status} />
                ) : (
                  <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-cream text-navy/30 shrink-0">
                    {isActive ? <Spinner /> : <span className="w-1.5 h-1.5 rounded-full bg-navy/20" />}
                  </span>
                )}

                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-3 mb-1">
                    <p className="font-semibold text-sm text-navy">{label}</p>
                    {step && (
                      <span className="text-xs text-navy/35 font-medium shrink-0">
                        {step.latency_ms} ms
                      </span>
                    )}
                  </div>

                  {step ? (
                    <>
                      <p className="text-xs text-navy/55 font-light leading-relaxed mb-2">
                        <span className="font-semibold text-navy/70">Reasoning — </span>
                        {step.reasoning}
                      </p>
                      <p className="text-xs text-navy/55 font-light leading-relaxed">
                        <span className="font-semibold text-navy/70">Observed — </span>
                        {step.observation}
                      </p>
                    </>
                  ) : (
                    <p className="text-xs text-navy/40 font-light leading-relaxed">{blurb}</p>
                  )}
                </div>
              </div>
            </li>
          );
        })}
      </ol>

      {phase === 'error' && error && (
        <div className="mt-6 bg-dusty-rose/10 border border-dusty-rose/30 rounded-2xl p-5 text-sm flex items-start gap-4">
          <span className="text-dusty-rose shrink-0 mt-0.5">
            <AlertIcon />
          </span>
          <div>
            <p className="font-bold mb-1 text-navy">Analysis interrupted</p>
            <p className="font-light text-dusty-rose">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
