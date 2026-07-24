'use client';

import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import { authFetchFormData, fetchAnalysisRun } from '../../../lib/api';
import { BarChart, BoxPlot, ClusterScatter, CorrelationHeatmap, Histogram } from '../../../components/charts';
import { Markdown } from '../../../components/markdown';

interface StepResult {
  // Mirrors the backend ReActStep schema (backend/schemas/analysis.py).
  agent_name: string;
  action: string;
  reasoning: string;
  observation: string;
  status: string;
  latency_ms: number;
  output: Record<string, unknown>;
}

interface GoalClassification {
  task_type: string;
  target_column: string | null;
  confidence: number;
  rationale: string;
}

interface AnalysisResult {
  goal: string;
  expertise_level: string;
  task_type: string | null;
  classification: GoalClassification | null;
  steps: StepResult[];
  recommendations: string[];
  rag_sources: string[];
  summary: string;
  dataset_id?: string | null;
  run_id?: string | null;
}

interface DataQualityRow {
  completeness_pct: number;
  uniqueness_pct: number;
  missing_count: number;
}

interface VizSpec {
  type: string;
  title: string;
  [key: string]: unknown;
}

function stepSummary(step: StepResult): string {
  const output = step.output ?? {};
  if (step.status === 'error') {
    return String(output.error ?? 'This step failed.');
  }
  if (typeof output.message === 'string') return output.message;
  if (typeof output.row_count === 'number') {
    return `Loaded ${output.row_count} rows × ${output.column_count ?? '?'} columns.`;
  }
  return 'Completed.';
}

const CheckIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
  </svg>
);

const SparkleIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
  </svg>
);

const ArrowRightIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
  </svg>
);

const DocumentIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
  </svg>
);

const SendIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
  </svg>
);

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export default function AnalysisResultPage() {
  const params = useParams<{ id: string }>();
  const [result, setResult] = useState<AnalysisResult | null | undefined>(undefined);
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchAnalysisRun(params.id)
      .then((data) => {
        if (!cancelled) setResult(data as AnalysisResult);
      })
      .catch(() => {
        if (!cancelled) setResult(null);
      });
    return () => {
      cancelled = true;
    };
  }, [params.id]);

  if (result === undefined) {
    return null;
  }

  if (result === null) {
    return (
      <div className="max-w-2xl mx-auto text-center py-24">
        <h1 className="font-[family-name:var(--font-serif)] text-3xl font-bold text-navy mb-4">
          No analysis found
        </h1>
        <p className="text-navy/50 font-light mb-8">
          This analysis doesn&apos;t exist, isn&apos;t yours, or the link is wrong.
          Start a new analysis to continue.
        </p>
        <Link
          href="/dashboard/analysis/new"
          className="inline-block bg-navy text-cream font-semibold px-8 py-4 rounded-2xl hover:bg-navy-light transition-all"
        >
          Start New Analysis
        </Link>
      </div>
    );
  }

  async function handleSendMessage(e: React.FormEvent) {
    e.preventDefault();
    const goal = chatInput.trim();
    if (!goal || isSending || !result) return;

    setChatHistory((prev) => [...prev, { id: crypto.randomUUID(), role: 'user', content: goal }]);
    setChatInput('');
    setIsSending(true);

    try {
      const formData = new FormData();
      formData.append('goal', goal);
      formData.append('expertise_level', result.expertise_level);
      if (result.dataset_id) {
        // Re-references the same uploaded dataset server-side — no
        // re-upload needed, and the full pipeline (real stats + RAG)
        // re-runs against it for this new goal.
        formData.append('dataset_id', result.dataset_id);
      }

      const data = await authFetchFormData<AnalysisResult>('/analysis/run', formData);
      // Keep dataset_id current in case the backend rotated/re-issued it.
      if (data.dataset_id) {
        setResult((prev) => (prev ? { ...prev, dataset_id: data.dataset_id } : prev));
      }
      const reply =
        data.recommendations.length > 0
          ? data.recommendations.join('\n\n')
          : "I couldn't ground a recommendation for that — try rephrasing.";

      setChatHistory((prev) => [...prev, { id: crypto.randomUUID(), role: 'assistant', content: reply }]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Something went wrong';
      setChatHistory((prev) => [...prev, { id: crypto.randomUUID(), role: 'assistant', content: `Error: ${message}` }]);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <div className="max-w-4xl mx-auto pb-32">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between mb-10 animate-fade-in">
        <div>
          <div className="flex items-center gap-4 mb-3">
            <h1 className="font-[family-name:var(--font-serif)] text-4xl font-bold text-navy">
              Workspace
            </h1>
            {result.steps.some((s) => s.status === 'error') ? (
              <span className="bg-dusty-rose/10 text-dusty-rose border border-dusty-rose/30 text-xs font-bold px-3 py-1.5 rounded-full flex items-center gap-1.5 uppercase tracking-wider">
                Completed With Errors
              </span>
            ) : (
              <span className="bg-sage-light/40 text-sage border border-sage/30 text-xs font-bold px-3 py-1.5 rounded-full flex items-center gap-1.5 uppercase tracking-wider">
                <CheckIcon /> Initial Analysis Complete
              </span>
            )}
          </div>
          <p className="text-navy/60 font-light text-lg">Goal: {result.goal}</p>
        </div>
      </div>

      {/* ── Initial Report (Treated as first AI response) ──────────── */}
      <div className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8 mb-8 animate-slide-up delay-100 shadow-sm shadow-navy/5">
        <div className="flex items-center gap-4 mb-6 pb-6 border-b border-dusty-rose/15">
          <div className="w-12 h-12 bg-navy text-cream rounded-2xl flex items-center justify-center">
            <SparkleIcon />
          </div>
          <div>
            <h2 className="font-[family-name:var(--font-serif)] font-bold text-navy text-xl">Initial Report Generated</h2>
            <p className="text-xs text-navy/40 uppercase tracking-widest font-medium mt-0.5">Automated Pipeline</p>
          </div>
        </div>

        {/* Goal classification — the Module 2 output that conditions the pipeline */}
        {result.classification && (
          <div className="mb-8">
            <h3 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-3">Goal Classification</h3>
            <div className="flex flex-wrap items-center gap-3 mb-2">
              <span className="bg-navy text-cream text-sm font-bold px-4 py-1.5 rounded-full uppercase tracking-wide">
                {result.classification.task_type.replace('_', ' ')}
              </span>
              <span className="text-navy/60 text-sm font-light">
                {Math.round(result.classification.confidence * 100)}% confidence
              </span>
              {result.classification.target_column && (
                <span className="text-navy/60 text-sm font-light">
                  target: <span className="font-medium">{result.classification.target_column}</span>
                </span>
              )}
            </div>
            {result.classification.rationale && (
              <p className="text-navy/50 text-sm font-light leading-relaxed">{result.classification.rationale}</p>
            )}
          </div>
        )}

        {/* Summary */}
        <div className="mb-8">
          <h3 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-3">Executive Summary</h3>
          <p className="text-navy/70 leading-relaxed font-light">{result.summary}</p>
        </div>

        {/* Pipeline */}
        <div className="mb-8">
          <h3 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-3">Agent Actions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {result.steps.map((step, idx) => (
              <div key={idx} className="bg-cream/50 border border-dusty-rose/15 rounded-2xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <p className="font-bold text-sm text-navy">{step.agent_name}</p>
                  {step.status === 'error' ? (
                    <span className="text-dusty-rose text-xs font-bold uppercase tracking-wide">Failed</span>
                  ) : (
                    <CheckIcon />
                  )}
                </div>
                <p className="text-xs text-navy/50 leading-relaxed font-light">{stepSummary(step)}</p>
                {step.reasoning && (
                  <p className="text-[11px] text-navy/35 italic leading-relaxed font-light mt-1.5">
                    {step.reasoning}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Data Quality */}
        {(() => {
          const miningOutput = result.steps.find((s) => s.agent_name === 'MiningAgent')?.output;
          const dataQuality = miningOutput?.data_quality as Record<string, DataQualityRow> | undefined;
          if (!dataQuality || Object.keys(dataQuality).length === 0) return null;

          return (
            <div className="mb-8">
              <h3 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-3">Data Quality</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-navy/40 uppercase tracking-wide">
                      <th className="pb-2 pr-4">Column</th>
                      <th className="pb-2 pr-4">Completeness</th>
                      <th className="pb-2 pr-4">Uniqueness</th>
                      <th className="pb-2">Missing</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(dataQuality).map(([col, q]) => (
                      <tr key={col} className="border-t border-dusty-rose/10">
                        <td className="py-2 pr-4 font-medium text-navy">{col}</td>
                        <td className="py-2 pr-4 text-navy/60">{q.completeness_pct}%</td>
                        <td className="py-2 pr-4 text-navy/60">{q.uniqueness_pct}%</td>
                        <td className="py-2 text-navy/60">{q.missing_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })()}

        {/* Visualizations */}
        {(() => {
          const vizOutput = result.steps.find((s) => s.agent_name === 'VisualizationAgent')?.output;
          const specs = (vizOutput?.viz_specs as VizSpec[] | undefined) ?? [];
          if (specs.length === 0) return null;

          return (
            <div className="mb-8">
              <h3 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-3">Visualizations</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {specs.map((spec, idx) => (
                  <div key={idx} className="bg-cream/40 border border-dusty-rose/15 rounded-2xl p-5">
                    <p className="text-xs font-bold text-navy mb-3">{spec.title}</p>
                    {spec.type === 'histogram' && <Histogram bins={spec.bins as { label: string; count: number }[]} />}
                    {spec.type === 'bar' && <BarChart items={spec.items as { label: string; value: number }[]} />}
                    {spec.type === 'feature_importance' && (
                      <BarChart items={spec.items as { label: string; value: number }[]} />
                    )}
                    {spec.type === 'boxplot' && (
                      <BoxPlot
                        min={spec.min as number | null}
                        q1={spec.q1 as number | null}
                        median={spec.median as number | null}
                        q3={spec.q3 as number | null}
                        max={spec.max as number | null}
                      />
                    )}
                    {spec.type === 'correlation_heatmap' && (
                      <CorrelationHeatmap
                        columns={spec.columns as string[]}
                        matrix={spec.matrix as (number | null)[][]}
                      />
                    )}
                    {spec.type === 'cluster_scatter' && (
                      <ClusterScatter points={spec.points as { x: number; y: number; cluster: number }[]} />
                    )}
                  </div>
                ))}
              </div>
            </div>
          );
        })()}

        {/* Recommendations */}
        <div className="mb-8">
          <h3 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-3">Recommendations</h3>
          {result.recommendations.length > 0 ? (
            <ul className="space-y-3">
              {result.recommendations.map((rec, idx) => (
                <li key={idx} className="flex gap-3 text-navy/70 font-light">
                  <span className="text-navy-muted bg-cream-dark w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5">{idx + 1}</span>
                  <Markdown text={rec} className="flex-1 min-w-0 text-sm" />
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-navy/40 font-light text-sm">
              No recommendations were grounded for this goal — try a more specific goal, or check that your dataset uploaded correctly above.
            </p>
          )}
        </div>

        {/* Sources */}
        <div>
          <h3 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-3">Grounded In</h3>
          <div className="flex flex-wrap gap-2">
            {result.rag_sources.map((src, idx) => (
              <span key={idx} className="bg-cream-dark/60 border border-dusty-rose/20 rounded-xl px-3 py-1.5 text-xs text-navy/60 font-light flex items-center gap-1.5 cursor-default">
                <DocumentIcon /> {src}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── Continuous Chat History ────────────────────────────────── */}
      <div className="space-y-6 animate-slide-up delay-200">
        {chatHistory.length === 0 && (
          <p className="text-navy/40 font-light text-sm text-center">
            {result.dataset_id
              ? 'Ask a follow-up — it re-runs the full pipeline against the same dataset, goal-conditioned on your new question.'
              : 'Ask a follow-up — it re-runs the pipeline goal-conditioned on your new question (no dataset was uploaded for this run).'}
          </p>
        )}
        {chatHistory.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-[2rem] p-6 ${
              msg.role === 'user' 
                ? 'bg-navy text-cream shadow-xl shadow-navy/10 rounded-tr-none' 
                : 'bg-warm-white/80 border border-dusty-rose/20 text-navy rounded-tl-none shadow-sm shadow-navy/5'
            }`}>
              {msg.role === 'assistant' && (
                <div className="flex items-center gap-2 mb-3 text-navy-muted">
                  <SparkleIcon />
                  <span className="text-xs font-bold uppercase tracking-widest">MAGE</span>
                </div>
              )}
              {msg.role === 'assistant' ? (
                <Markdown text={msg.content} className="font-light leading-relaxed text-sm md:text-base" />
              ) : (
                <p className="font-light leading-relaxed text-sm md:text-base">{msg.content}</p>
              )}
            </div>
          </div>
        ))}
        {isSending && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-[2rem] rounded-tl-none p-6 bg-warm-white/80 border border-dusty-rose/20 shadow-sm shadow-navy/5">
              <div className="flex items-center gap-2 text-navy-muted">
                <SparkleIcon />
                <span className="text-xs font-bold uppercase tracking-widest">MAGE is thinking…</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Sticky Chat Input ──────────────────────────────────────── */}
      <div className="fixed bottom-0 left-64 right-0 p-8 bg-gradient-to-t from-cream via-cream to-transparent pointer-events-none z-30">
        <div className="max-w-4xl mx-auto pointer-events-auto">
          <form onSubmit={handleSendMessage} className="relative group">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Ask a follow-up question…"
              disabled={isSending}
              className="w-full bg-warm-white/90 backdrop-blur-md border border-dusty-rose/30 rounded-full pl-6 pr-16 py-5 text-navy placeholder:text-navy/40 shadow-xl shadow-navy/5 focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all disabled:opacity-60"
            />
            <button
              type="submit"
              disabled={!chatInput.trim() || isSending}
              className="absolute right-3 top-3 bottom-3 w-12 bg-navy text-cream rounded-full flex items-center justify-center hover:bg-navy-light disabled:opacity-50 disabled:hover:bg-navy transition-all"
            >
              <SendIcon />
            </button>
          </form>
          <p className="text-center text-[10px] text-navy/40 font-medium uppercase tracking-widest mt-4">
            MAGE can process new queries and control agents dynamically
          </p>
        </div>
      </div>
    </div>
  );
}
