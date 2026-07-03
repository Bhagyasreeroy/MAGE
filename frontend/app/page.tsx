'use client';

import { useState } from 'react';

type ExpertiseLevel = 'beginner' | 'intermediate' | 'expert';

interface AnalysisResult {
  goal: string;
  expertise_level: ExpertiseLevel;
  steps: Array<{ agent: string; status: string; output: Record<string, unknown> }>;
  recommendations: string[];
  rag_sources: string[];
  summary: string;
}

const EXPERTISE_OPTIONS: { value: ExpertiseLevel; label: string; description: string }[] = [
  { value: 'beginner', label: 'Beginner', description: 'Plain language, step-by-step explanations' },
  { value: 'intermediate', label: 'Intermediate', description: 'Concise insights with statistics' },
  { value: 'expert', label: 'Expert', description: 'Dense technical recommendations' },
];

const AGENT_ICONS: Record<string, string> = {
  IngestionAgent: '📥',
  MiningAgent: '⛏️',
  VisualizationAgent: '📊',
  RecommendationAgent: '💡',
};

export default function HomePage() {
  const [goal, setGoal] = useState('');
  const [expertiseLevel, setExpertiseLevel] = useState<ExpertiseLevel>('intermediate');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!goal.trim()) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${apiUrl}/analysis/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal: goal.trim(), expertise_level: expertiseLevel }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail ?? `HTTP ${res.status}`);
      }

      const data: AnalysisResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-950 text-white">
      {/* ── Animated Background ─────────────────────────────────────────────── */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-indigo-500/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-violet-500/20 rounded-full blur-3xl animate-pulse delay-1000" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-blue-600/10 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-6 py-16">
        {/* ── Hero ──────────────────────────────────────────────────────────── */}
        <header className="text-center mb-16">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 bg-indigo-500/10 border border-indigo-500/30 rounded-full px-4 py-1.5 text-sm text-indigo-300 mb-6 backdrop-blur-sm">
            <span className="w-2 h-2 bg-indigo-400 rounded-full animate-ping" />
            Multi-Agent AI · EDA · RAG-Grounded
          </div>

          <h1 className="text-7xl font-black tracking-tight mb-4">
            <span className="bg-gradient-to-r from-indigo-400 via-violet-400 to-purple-400 bg-clip-text text-transparent">
              MAGE
            </span>
          </h1>
          <p className="text-xl text-slate-300 font-medium mb-2">
            Multi-Agent Goal-conditioned EDA
          </p>
          <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Describe your analytical goal. MAGE&apos;s specialist agents will ingest your data,
            mine patterns, select visualisations, and deliver{' '}
            <span className="text-indigo-300 font-medium">RAG-grounded recommendations</span>{' '}
            tailored to your expertise.
          </p>
        </header>

        {/* ── Analysis Form ─────────────────────────────────────────────────── */}
        <section className="mb-12">
          <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl">
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Goal Input */}
              <div>
                <label
                  htmlFor="goal-input"
                  className="block text-sm font-semibold text-slate-300 mb-3"
                >
                  Analytical Goal
                </label>
                <div className="relative">
                  <textarea
                    id="goal-input"
                    rows={4}
                    value={goal}
                    onChange={(e) => setGoal(e.target.value)}
                    placeholder="e.g. Identify the top factors driving customer churn in the last 6 months…"
                    className="w-full bg-slate-900/60 border border-white/10 rounded-2xl px-5 py-4 text-white placeholder:text-slate-500 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all duration-200 text-sm leading-relaxed"
                    minLength={5}
                    maxLength={2000}
                    required
                  />
                  <span className="absolute bottom-3 right-4 text-xs text-slate-500">
                    {goal.length}/2000
                  </span>
                </div>
              </div>

              {/* Expertise Level */}
              <div>
                <label
                  htmlFor="expertise-select"
                  className="block text-sm font-semibold text-slate-300 mb-3"
                >
                  Expertise Level
                </label>
                <div className="grid grid-cols-3 gap-3" role="radiogroup" aria-label="Expertise level">
                  {EXPERTISE_OPTIONS.map(({ value, label, description }) => (
                    <button
                      key={value}
                      type="button"
                      id={`expertise-${value}`}
                      role="radio"
                      aria-checked={expertiseLevel === value}
                      onClick={() => setExpertiseLevel(value)}
                      className={`relative p-4 rounded-xl border text-left transition-all duration-200 cursor-pointer ${
                        expertiseLevel === value
                          ? 'bg-indigo-500/20 border-indigo-500/60 shadow-lg shadow-indigo-500/10'
                          : 'bg-slate-900/40 border-white/10 hover:border-white/20 hover:bg-slate-800/40'
                      }`}
                    >
                      {expertiseLevel === value && (
                        <span className="absolute top-2 right-2 w-2 h-2 bg-indigo-400 rounded-full" />
                      )}
                      <p className="font-semibold text-sm text-white mb-1">{label}</p>
                      <p className="text-xs text-slate-400 leading-snug">{description}</p>
                    </button>
                  ))}
                </div>
                {/* Hidden select for accessibility / form semantics */}
                <select
                  id="expertise-select"
                  value={expertiseLevel}
                  onChange={(e) => setExpertiseLevel(e.target.value as ExpertiseLevel)}
                  className="sr-only"
                  aria-hidden="true"
                >
                  {EXPERTISE_OPTIONS.map(({ value, label }) => (
                    <option key={value} value={value}>{label}</option>
                  ))}
                </select>
              </div>

              {/* Submit */}
              <button
                id="run-analysis-btn"
                type="submit"
                disabled={isLoading || goal.trim().length < 5}
                className="w-full bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed text-white font-bold py-4 px-8 rounded-2xl transition-all duration-200 shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 hover:-translate-y-0.5 active:translate-y-0 flex items-center justify-center gap-3"
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Running agents…
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Run Analysis
                  </>
                )}
              </button>
            </form>
          </div>
        </section>

        {/* ── Error State ──────────────────────────────────────────────────── */}
        {error && (
          <div
            role="alert"
            className="bg-red-500/10 border border-red-500/30 rounded-2xl p-5 mb-8 text-red-300 text-sm flex items-start gap-3"
          >
            <svg className="w-5 h-5 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <p className="font-semibold mb-1">Analysis failed</p>
              <p className="text-red-400">{error}</p>
            </div>
          </div>
        )}

        {/* ── Results Dashboard ─────────────────────────────────────────────── */}
        {result && (
          <section aria-label="Analysis results" className="space-y-6 animate-fade-in">
            {/* Summary Card */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-indigo-500/20 rounded-xl flex items-center justify-center text-xl">✨</div>
                <div>
                  <h2 className="font-bold text-white">Analysis Summary</h2>
                  <p className="text-xs text-slate-400 capitalize">{result.expertise_level} level</p>
                </div>
              </div>
              <p className="text-slate-300 text-sm leading-relaxed">{result.summary}</p>
            </div>

            {/* Agent Steps */}
            <div>
              <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4 px-1">
                Agent Pipeline
              </h2>
              <div className="grid grid-cols-2 gap-4">
                {result.steps.map((step, idx) => (
                  <div
                    key={idx}
                    className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-5"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-2xl">{AGENT_ICONS[step.agent] ?? '🤖'}</span>
                      <div>
                        <p className="font-semibold text-sm text-white">{step.agent}</p>
                        <span
                          className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                            step.status === 'success'
                              ? 'bg-emerald-500/20 text-emerald-400'
                              : 'bg-red-500/20 text-red-400'
                          }`}
                        >
                          {step.status}
                        </span>
                      </div>
                    </div>
                    <p className="text-xs text-slate-400 font-mono truncate">
                      {(step.output as { message?: string }).message ?? JSON.stringify(step.output).slice(0, 60)}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            {result.recommendations.length > 0 && (
              <div>
                <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4 px-1">
                  Recommendations
                </h2>
                <div className="space-y-3">
                  {result.recommendations.map((rec, idx) => (
                    <div
                      key={idx}
                      className="bg-indigo-500/5 border border-indigo-500/20 rounded-2xl p-4 flex items-start gap-3"
                    >
                      <span className="text-indigo-400 mt-0.5">→</span>
                      <p className="text-sm text-slate-300 leading-relaxed">{rec}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* RAG Sources */}
            {result.rag_sources.length > 0 && (
              <div>
                <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-3 px-1">
                  Knowledge Sources
                </h2>
                <div className="flex flex-wrap gap-2">
                  {result.rag_sources.map((src, idx) => (
                    <span
                      key={idx}
                      className="bg-slate-800/60 border border-white/10 rounded-lg px-3 py-1 text-xs text-slate-400 font-mono"
                    >
                      📄 {src}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        {/* ── Architecture Pills ────────────────────────────────────────────── */}
        {!result && !isLoading && (
          <section className="text-center mt-4">
            <p className="text-xs text-slate-500 mb-4 uppercase tracking-widest font-medium">Powered by</p>
            <div className="flex flex-wrap justify-center gap-2">
              {[
                'FastAPI', 'Next.js 14', 'ReAct Agents', 'FAISS / ChromaDB',
                'LangChain', 'Pandas', 'Docker',
              ].map((tech) => (
                <span
                  key={tech}
                  className="bg-white/5 border border-white/10 rounded-full px-3 py-1 text-xs text-slate-400 hover:text-slate-300 hover:border-white/20 transition-colors"
                >
                  {tech}
                </span>
              ))}
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
