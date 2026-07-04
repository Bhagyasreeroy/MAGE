'use client';

import { useMemo, useState } from 'react';

type ExpertiseLevel = 'beginner' | 'intermediate' | 'expert';

type AnalysisResult = {
  goal: string;
  expertise_level: ExpertiseLevel;
  steps: Array<{ agent: string; status: string; output: Record<string, unknown> }>;
  recommendations: string[];
  rag_sources: string[];
  summary: string;
};

type IngestionResult = {
  row_count: number;
  column_count: number;
  column_summary: Array<{
    name: string;
    dtype: string;
    missing_count: number;
    stats: {
      min?: number | null;
      max?: number | null;
      mean?: number | null;
      unique_count?: number | null;
    } | null;
  }>;
  warnings: string[];
};

const QUICK_GOALS = [
  'Predict customer churn',
  'Discover hidden patterns',
  'Explain feature importance',
];

const WORKFLOW_STEPS = [
  { label: 'Upload', icon: 'file_up' },
  { label: 'AI Agents', icon: 'smart_toy' },
  { label: 'Analysis', icon: 'analytics' },
  { label: 'Insights', icon: 'sparkles' },
  { label: 'Report', icon: 'description' },
];

const EXPERTISE_OPTIONS: { value: ExpertiseLevel; label: string; description: string }[] = [
  { value: 'beginner', label: 'Beginner', description: 'Plain language, step-by-step guidance' },
  { value: 'intermediate', label: 'Intermediate', description: 'Balanced detail and statistics' },
  { value: 'expert', label: 'Expert', description: 'Dense technical recommendations' },
];

const AGENT_ICONS: Record<string, string> = {
  IngestionAgent: 'file_up',
  MiningAgent: 'query_stats',
  VisualizationAgent: 'bar_chart',
  RecommendationAgent: 'psychology',
};

function Icon({ name }: { name: string }) {
  const common = 'h-5 w-5';

  if (name === 'file_up') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={common}>
        <path d="M6 20h12a2 2 0 0 0 2-2V8l-5-5H8a2 2 0 0 0-2 2v15" />
        <path d="M14 3v5h5" />
        <path d="M12 18V9" />
        <path d="M8.5 11.5 12 8l3.5 3.5" />
      </svg>
    );
  }

  if (name === 'smart_toy') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={common}>
        <rect x="6" y="6" width="12" height="12" rx="3" />
        <path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3" />
        <circle cx="9" cy="11" r="0.9" fill="currentColor" stroke="none" />
        <circle cx="15" cy="11" r="0.9" fill="currentColor" stroke="none" />
        <path d="M9 15c1.2 1 4.8 1 6 0" />
      </svg>
    );
  }

  if (name === 'analytics') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={common}>
        <path d="M4 19h16" />
        <path d="M6 15v-4" />
        <path d="M10 15V7" />
        <path d="M14 15v-2" />
        <path d="M18 15V9" />
      </svg>
    );
  }

  if (name === 'sparkles') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={common}>
        <path d="m12 3 1.6 5.4L19 10l-5.4 1.6L12 17l-1.6-5.4L5 10l5.4-1.6L12 3Z" />
        <path d="m5 14 .8 2.5L8 17.3l-2.2.8L5 20.5l-.8-2.4-2.2-.8 2.2-.8L5 14Z" />
      </svg>
    );
  }

  if (name === 'description') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={common}>
        <path d="M7 3h7l5 5v13H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z" />
        <path d="M14 3v5h5" />
        <path d="M9 12h6M9 16h6" />
      </svg>
    );
  }

  if (name === 'account_circle') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={common}>
        <circle cx="12" cy="8" r="3" />
        <path d="M5 20a7 7 0 0 1 14 0" />
      </svg>
    );
  }

  if (name === 'check_circle') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={common}>
        <circle cx="12" cy="12" r="9" />
        <path d="m8 12 2.5 2.5L16 9" />
      </svg>
    );
  }

  if (name === 'lightbulb') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={common}>
        <path d="M9 18h6" />
        <path d="M10 21h4" />
        <path d="M8.5 14.5A6 6 0 1 1 15.5 14.5c-.8.7-1.5 1.7-1.5 2.5h-4c0-.8-.7-1.8-1.5-2.5Z" />
      </svg>
    );
  }

  if (name === 'trending_up') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={common}>
        <path d="M4 16l6-6 4 4 6-6" />
        <path d="M14 8h6v6" />
      </svg>
    );
  }

  if (name === 'menu_book') {
    return (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={common}>
        <path d="M4 6a3 3 0 0 1 3-3h11v17H7a3 3 0 0 0-3 3V6Z" />
        <path d="M7 3v18" />
        <path d="M10 7h6M10 11h6" />
      </svg>
    );
  }

  return <span className={common}>•</span>;
}

export default function HomePage() {
  const [goal, setGoal] = useState('');
  const [expertiseLevel, setExpertiseLevel] = useState<ExpertiseLevel>('intermediate');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [ingestionResult, setIngestionResult] = useState<IngestionResult | null>(null);
  const [ingestionError, setIngestionError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
  const canSubmit = goal.trim().length >= 5 && !isLoading;

  const pipelineStatus = useMemo(
    () => [
      { label: 'Data Ingestion', value: '100%', width: '100%' },
      { label: 'Transformation', value: result ? '64%' : '54%', width: result ? '64%' : '54%' },
    ],
    [result],
  );

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

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!uploadFile) return;

    setIsUploading(true);
    setIngestionError(null);
    setIngestionResult(null);

    try {
      const formData = new FormData();
      formData.append('file', uploadFile);

      const res = await fetch(`${apiUrl}/analysis/ingest`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail ?? `HTTP ${res.status}`);
      }

      const data: IngestionResult = await res.json();
      setIngestionResult(data);
    } catch (err) {
      setIngestionError(err instanceof Error ? err.message : 'Unknown upload error');
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[var(--bg)] text-[var(--ink)]">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-32 top-0 h-[28rem] w-[28rem] rounded-full bg-[radial-gradient(circle,rgba(126,99,164,0.18),transparent_70%)] blur-3xl" />
        <div className="absolute right-[-8rem] top-24 h-[24rem] w-[24rem] rounded-full bg-[radial-gradient(circle,rgba(201,176,224,0.25),transparent_70%)] blur-3xl" />
        <div className="absolute bottom-[-9rem] left-1/4 h-[24rem] w-[24rem] rounded-full bg-[radial-gradient(circle,rgba(98,122,84,0.15),transparent_70%)] blur-3xl" />
      </div>

      <div className="relative z-10 mx-auto flex min-h-screen max-w-[1440px] flex-col px-4 pb-8 pt-3 sm:px-6 lg:px-8">
        <header className="mb-6 rounded-[28px] border border-stone-200/80 bg-[rgba(247,244,237,0.88)] px-4 py-3 backdrop-blur-xl sm:px-6">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="font-serif text-2xl font-bold tracking-tight text-[var(--accent)]">MAGE</div>
              <nav className="hidden items-center gap-6 text-sm text-[var(--muted-dark)] sm:flex">
                <a className="border-b-2 border-[var(--accent)] pb-1 text-[var(--ink)]" href="#analysis">
                  Analysis
                </a>
                <a href="#visualizations">Visualizations</a>
                <a href="#conversations">Conversations</a>
                <a href="#reports">Reports</a>
              </nav>
            </div>
            <button
              type="button"
              aria-label="Account"
              className="flex h-9 w-9 items-center justify-center rounded-full border border-stone-300 bg-white text-[var(--accent)] shadow-sm transition hover:-translate-y-0.5 hover:shadow-md"
            >
              <Icon name="account_circle" />
            </button>
          </div>
        </header>

        <section id="analysis" className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <div className="space-y-6">
            <div className="rounded-[32px] border border-stone-200 bg-[rgba(251,249,243,0.92)] px-6 py-10 shadow-[0_24px_80px_rgba(98,73,140,0.08)] sm:px-10">
              <div className="mx-auto max-w-3xl text-center">
                <p className="font-serif text-5xl font-bold tracking-tight text-[var(--accent)] sm:text-6xl">MAGE</p>
                <p className="mx-auto mt-3 max-w-2xl text-[1rem] leading-7 text-[var(--muted-dark)] sm:text-[1.05rem]">
                  Goal-conditioned exploratory data analysis powered by collaborative AI agents.
                </p>
              </div>

              <form onSubmit={handleSubmit} className="mx-auto mt-6 max-w-4xl rounded-2xl border border-stone-200 bg-white p-4 shadow-[0_12px_40px_rgba(84,66,111,0.08)] sm:p-5">
                <label htmlFor="goal-input" className="mb-2 block text-sm text-[var(--muted-dark)]">
                  What would you like to understand?
                </label>
                <textarea
                  id="goal-input"
                  rows={3}
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                  placeholder="Predict customer churn, discover hidden patterns, explain feature importance..."
                  className="w-full resize-none rounded-xl border border-transparent bg-transparent px-0 py-1 text-[1rem] leading-7 text-[var(--ink)] outline-none placeholder:text-stone-400"
                />

                <div className="mt-4 flex flex-wrap gap-2">
                  {QUICK_GOALS.map((item) => (
                    <button
                      key={item}
                      type="button"
                      onClick={() => setGoal(item)}
                      className="rounded-full border border-stone-300 bg-stone-100 px-3 py-1 text-xs text-[var(--ink)] transition hover:border-[var(--accent)] hover:bg-[var(--accent-soft)]"
                    >
                      {item}
                    </button>
                  ))}
                </div>

                <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex flex-wrap gap-2">
                    {EXPERTISE_OPTIONS.map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() => setExpertiseLevel(option.value)}
                        className={`rounded-full px-4 py-2 text-xs transition ${
                          expertiseLevel === option.value
                            ? 'bg-[var(--accent)] text-white shadow-[0_8px_20px_rgba(111,88,144,0.24)]'
                            : 'border border-stone-300 bg-white text-[var(--muted-dark)] hover:border-[var(--accent)]'
                        }`}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>

                  <button
                    id="run-analysis-btn"
                    type="submit"
                    disabled={!canSubmit}
                    className="inline-flex items-center justify-center gap-2 rounded-full bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white shadow-[0_10px_24px_rgba(111,88,144,0.25)] transition hover:-translate-y-0.5 hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:bg-stone-400"
                  >
                    {isLoading ? 'Running analysis...' : 'Start Analysis'}
                  </button>
                </div>
              </form>
            </div>

            <div className="space-y-4">
              <div className="text-center">
                <p className="text-sm text-[var(--muted-dark)]">The MAGE Workflow</p>
                <div className="mx-auto mt-2 h-0.5 w-10 rounded-full bg-[var(--accent)]" />
              </div>
              <div className="grid gap-3 sm:grid-cols-5">
                {WORKFLOW_STEPS.map((step, index) => (
                  <div key={step.label} className="flex flex-col items-center gap-2 text-center">
                    <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-stone-200 bg-[var(--paper)] text-[var(--accent)] shadow-sm">
                      <Icon name={step.icon} />
                    </div>
                    <p className="text-[0.7rem] uppercase tracking-[0.22em] text-[var(--muted-dark)]">{step.label}</p>
                    {index < WORKFLOW_STEPS.length - 1 && <div className="hidden sm:block" />}
                  </div>
                ))}
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-[1.35fr_0.85fr]">
              <div className="rounded-[26px] border border-stone-200 bg-[var(--paper)] p-5 shadow-[0_12px_40px_rgba(86,74,113,0.07)]">
                <div className="inline-flex rounded-md bg-[var(--accent-soft)] px-2 py-1 text-xs text-[var(--accent)]">Methodology</div>
                <h2 className="mt-3 font-serif text-3xl font-bold text-[var(--ink)] sm:text-4xl">Scientific Precision</h2>
                <p className="mt-3 max-w-2xl text-sm leading-7 text-[var(--muted-dark)] sm:text-[0.96rem]">
                  Our agents employ statistically rigorous methods to ensure every insight is grounded in data integrity, moving beyond superficial correlation.
                </p>
                <div className="mt-6 h-28 rounded-2xl bg-[radial-gradient(circle_at_20%_20%,rgba(111,88,144,0.16),transparent_28%),radial-gradient(circle_at_70%_50%,rgba(92,124,77,0.12),transparent_28%),linear-gradient(135deg,rgba(232,227,217,0.9),rgba(247,244,237,0.95))]" />
              </div>

              <div className="rounded-[26px] border border-stone-200 bg-[var(--paper)] p-5 shadow-[0_12px_40px_rgba(86,74,113,0.07)]">
                <p className="font-serif text-2xl text-[var(--accent)]">Traceability</p>
                <p className="mt-3 text-sm leading-7 text-[var(--muted-dark)]">
                  Audit every analytical step. Every conclusion is linked to the exact slice of data and transformation logic used by the agent.
                </p>
                <div className="mt-5 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-[var(--accent-soft)] text-[var(--accent)]">
                  <Icon name="check_circle" />
                </div>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-[0.95fr_1.05fr]">
              <div className="rounded-[26px] border border-stone-200 bg-[var(--paper)] p-5 shadow-[0_12px_40px_rgba(86,74,113,0.07)]">
                <p className="font-serif text-2xl text-[var(--accent)]">Explainability</p>
                <p className="mt-3 text-sm leading-7 text-[var(--muted-dark)]">
                  Complex models translated into narrative context. We do not just provide numbers; we tell the story of the underlying phenomena.
                </p>
                <div className="mt-5 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-[var(--accent-soft)] text-[var(--accent)]">
                  <Icon name="menu_book" />
                </div>
              </div>

              <div className="rounded-[26px] border border-stone-200 bg-[var(--paper)] p-6 shadow-[0_12px_40px_rgba(86,74,113,0.07)]">
                <p className="font-serif text-xl italic leading-8 text-[var(--accent)] sm:text-2xl">
                  “The goal of MAGE is not just to compute, but to converse with complexity until it yields clarity.”
                </p>
                <p className="mt-4 text-xs uppercase tracking-[0.3em] text-[var(--muted-dark)]">MAGE Research Collective</p>
              </div>
            </div>

            <div className="grid overflow-hidden rounded-[30px] border border-stone-200 shadow-[0_24px_80px_rgba(86,74,113,0.1)] lg:grid-cols-[0.92fr_1.08fr]">
              <div className="bg-[var(--accent)] p-6 text-white sm:p-8">
                <p className="font-serif text-3xl font-bold sm:text-4xl">High-Dimensional Exploration</p>
                <p className="mt-4 max-w-md text-sm leading-7 text-white/90">
                  Visualizing the invisible. Watch as agents isolate variables and identify clusters across thousands of parameters in real-time.
                </p>
                <button className="mt-5 rounded-full border border-white/60 bg-white px-4 py-2 text-sm font-semibold text-[var(--accent)] transition hover:-translate-y-0.5">
                  View Demo
                </button>
              </div>
              <div className="min-h-[18rem] bg-[#232734] p-4 sm:p-5">
                <div className="grid h-full grid-cols-2 gap-3 rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,rgba(34,37,56,0.95),rgba(21,24,34,0.98))] p-3">
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                    <div className="mb-3 h-2 w-16 rounded-full bg-white/20" />
                    <div className="h-[calc(100%-1.2rem)] rounded-xl bg-[radial-gradient(circle_at_30%_20%,rgba(199,126,180,0.65),transparent_22%),linear-gradient(180deg,rgba(51,56,79,0.95),rgba(24,28,39,0.95))]" />
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                    <div className="mb-3 h-2 w-20 rounded-full bg-white/20" />
                    <div className="grid h-[calc(100%-1.2rem)] gap-2">
                      <div className="h-16 rounded-xl bg-[linear-gradient(135deg,rgba(200,121,171,0.86),rgba(126,97,164,0.45))]" />
                      <div className="h-full rounded-xl bg-[linear-gradient(180deg,rgba(66,72,98,0.95),rgba(31,34,48,0.95))]" />
                    </div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                    <div className="mb-3 h-2 w-24 rounded-full bg-white/20" />
                    <div className="flex h-[calc(100%-1.2rem)] items-end gap-2 rounded-xl bg-[linear-gradient(180deg,rgba(35,40,58,0.95),rgba(21,24,34,0.95))] p-3">
                      {[35, 55, 28, 74, 42, 67, 23, 89].map((height, idx) => (
                        <div
                          key={`${height}-${idx}`}
                          className="flex-1 rounded-t-md bg-[linear-gradient(180deg,rgba(200,121,171,0.95),rgba(126,97,164,0.65))]"
                          style={{ height: `${height}%` }}
                        />
                      ))}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                    <div className="mb-3 h-2 w-14 rounded-full bg-white/20" />
                    <div className="h-[calc(100%-1.2rem)] rounded-xl bg-[radial-gradient(circle_at_70%_40%,rgba(199,126,180,0.65),transparent_20%),linear-gradient(180deg,rgba(51,56,79,0.95),rgba(24,28,39,0.95))]" />
                  </div>
                </div>
              </div>
            </div>

            <section className="rounded-[28px] border border-stone-200 bg-[var(--paper)] p-5 shadow-[0_12px_40px_rgba(86,74,113,0.08)]">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-[var(--muted-dark)]">Ingestion Demo</p>
                  <h2 className="mt-1 font-serif text-3xl text-[var(--ink)]">Upload CSV or XLSX</h2>
                </div>
                <p className="text-sm text-[var(--muted-dark)]">
                  The backend will parse your file and return a live profile.
                </p>
              </div>

              <form onSubmit={handleUpload} className="mt-5 grid gap-4 lg:grid-cols-[1fr_auto] lg:items-end">
                <label className="block rounded-[22px] border border-dashed border-[rgba(118,96,149,0.32)] bg-white px-5 py-6 text-sm text-[var(--muted-dark)] transition hover:border-[var(--accent)]">
                  <span className="mb-2 flex items-center gap-2 font-medium text-[var(--ink)]">
                    <Icon name="file_up" />
                    Choose a CSV or XLSX file
                  </span>
                  <span className="block text-xs leading-6 text-[var(--muted-dark)]">
                    Upload a table to see row counts, column metadata, missingness, and summary statistics.
                  </span>
                  <input
                    type="file"
                    accept=".csv,.xlsx"
                    className="mt-4 block w-full text-sm text-[var(--muted-dark)] file:mr-4 file:rounded-full file:border-0 file:bg-[var(--accent)] file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white hover:file:bg-[var(--accent-strong)]"
                    onChange={(event) => {
                      const selected = event.target.files?.[0] ?? null;
                      setUploadFile(selected);
                      setIngestionError(null);
                      setIngestionResult(null);
                    }}
                  />
                </label>

                <button
                  type="submit"
                  disabled={!uploadFile || isUploading}
                  className="inline-flex items-center justify-center rounded-full bg-[var(--accent)] px-5 py-3 text-sm font-semibold text-white shadow-[0_10px_24px_rgba(111,88,144,0.25)] transition hover:-translate-y-0.5 hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:bg-stone-400"
                >
                  {isUploading ? 'Uploading…' : 'Run Demo'}
                </button>
              </form>

              {uploadFile && (
                <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-stone-200 bg-white px-4 py-2 text-xs text-[var(--muted-dark)]">
                  <Icon name="description" />
                  Selected file: {uploadFile.name}
                </div>
              )}

              {ingestionError && (
                <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700" role="alert">
                  <strong className="block">Upload failed</strong>
                  <span>{ingestionError}</span>
                </div>
              )}

              {ingestionResult && (
                <div className="mt-5 space-y-4 rounded-[24px] border border-stone-200 bg-white p-5">
                  <div className="grid gap-4 sm:grid-cols-3">
                    <div className="rounded-2xl bg-[var(--accent-soft)] p-4">
                      <p className="text-xs uppercase tracking-[0.22em] text-[var(--muted-dark)]">Rows</p>
                      <p className="mt-2 font-serif text-3xl text-[var(--ink)]">{ingestionResult.row_count}</p>
                    </div>
                    <div className="rounded-2xl bg-[var(--accent-soft)] p-4">
                      <p className="text-xs uppercase tracking-[0.22em] text-[var(--muted-dark)]">Columns</p>
                      <p className="mt-2 font-serif text-3xl text-[var(--ink)]">{ingestionResult.column_count}</p>
                    </div>
                    <div className="rounded-2xl bg-[var(--accent-soft)] p-4">
                      <p className="text-xs uppercase tracking-[0.22em] text-[var(--muted-dark)]">Warnings</p>
                      <p className="mt-2 font-serif text-3xl text-[var(--ink)]">{ingestionResult.warnings.length}</p>
                    </div>
                  </div>

                  <div className="overflow-hidden rounded-2xl border border-stone-200">
                    <table className="min-w-full divide-y divide-stone-200 text-left text-sm">
                      <thead className="bg-stone-50 text-[var(--muted-dark)]">
                        <tr>
                          <th className="px-4 py-3 font-medium">Column</th>
                          <th className="px-4 py-3 font-medium">Type</th>
                          <th className="px-4 py-3 font-medium">Missing</th>
                          <th className="px-4 py-3 font-medium">Stats</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-stone-200 bg-white">
                        {ingestionResult.column_summary.map((column) => (
                          <tr key={column.name}>
                            <td className="px-4 py-3 font-medium text-[var(--ink)]">{column.name}</td>
                            <td className="px-4 py-3 text-[var(--muted-dark)]">{column.dtype}</td>
                            <td className="px-4 py-3 text-[var(--muted-dark)]">{column.missing_count}</td>
                            <td className="px-4 py-3 text-[var(--muted-dark)]">
                              {column.stats?.unique_count !== null && column.stats?.unique_count !== undefined
                                ? `unique: ${column.stats.unique_count}`
                                : `min: ${column.stats?.min ?? '—'}, max: ${column.stats?.max ?? '—'}, mean: ${
                                    column.stats?.mean ?? '—'
                                  }`}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {ingestionResult.warnings.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-xs uppercase tracking-[0.22em] text-[var(--muted-dark)]">Warnings</p>
                      <div className="space-y-2">
                        {ingestionResult.warnings.map((warning) => (
                          <div key={warning} className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                            {warning}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </section>

            {result && (
              <section className="grid gap-4 rounded-[28px] border border-stone-200 bg-white p-6 shadow-[0_18px_60px_rgba(86,74,113,0.1)]" aria-label="Analysis results">
                <div className="flex items-center gap-3">
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[var(--accent-soft)] text-[var(--accent)]">
                    <Icon name="sparkles" />
                  </div>
                  <div>
                    <h2 className="font-serif text-3xl text-[var(--ink)]">Analysis Summary</h2>
                    <p className="text-sm text-[var(--muted-dark)] capitalize">{result.expertise_level} level</p>
                  </div>
                </div>
                <p className="max-w-4xl text-sm leading-7 text-[var(--muted-dark)]">{result.summary}</p>
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                  {result.steps.map((step) => (
                    <div key={step.agent} className="rounded-2xl border border-stone-200 bg-[var(--paper)] p-4">
                      <div className="flex items-start gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-[var(--accent)]">
                          <Icon name={AGENT_ICONS[step.agent] ?? 'smart_toy'} />
                        </div>
                        <div className="min-w-0">
                          <p className="font-medium text-[var(--ink)]">{step.agent}</p>
                          <p className="text-xs text-[var(--muted-dark)]">{step.status}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                {result.recommendations.length > 0 && (
                  <div className="space-y-3">
                    <h3 className="text-xs uppercase tracking-[0.22em] text-[var(--muted-dark)]">Recommendations</h3>
                    {result.recommendations.map((rec) => (
                      <div key={rec} className="rounded-2xl border border-[rgba(111,88,144,0.18)] bg-[var(--accent-soft)] px-4 py-3 text-sm leading-7 text-[var(--ink)]">
                        {rec}
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}

            {error && (
              <div className="rounded-[24px] border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700 shadow-sm" role="alert">
                <div className="font-semibold">Analysis failed</div>
                <div className="mt-1 text-red-600">{error}</div>
              </div>
            )}
          </div>

          <aside className="space-y-4 lg:sticky lg:top-6 lg:self-start">
            <div className="rounded-[28px] border border-stone-200 bg-[rgba(251,249,243,0.95)] p-5 shadow-[0_18px_60px_rgba(86,74,113,0.08)]">
              <div className="flex items-center justify-between">
                <p className="font-serif text-2xl text-[var(--accent)]">Context Panel</p>
                <span className="text-xs uppercase tracking-[0.24em] text-[var(--muted-dark)]">analytics</span>
              </div>

              <div className="mt-4 space-y-3">
                <p className="text-xs uppercase tracking-[0.24em] text-[var(--muted-dark)]">Retrieved Docs</p>
                <div className="rounded-2xl border border-stone-200 bg-white p-3">
                  <p className="text-sm font-medium text-[var(--ink)]">Q3 Sales Forecast Methodology</p>
                  <p className="mt-1 text-xs leading-5 text-[var(--muted-dark)]">
                    Detailed breakdown of the moving average weights used in the projection model.
                  </p>
                </div>
                <div className="rounded-2xl border border-stone-200 bg-white p-3">
                  <p className="text-sm font-medium text-[var(--ink)]">Regional Compliance Logs</p>
                  <p className="mt-1 text-xs leading-5 text-[var(--muted-dark)]">
                    Audit trail for APAC data ingestion pipelines and normalization steps.
                  </p>
                </div>
              </div>

              <div className="mt-5 space-y-3">
                <p className="text-xs uppercase tracking-[0.24em] text-[var(--muted-dark)]">Pipeline Status</p>
                {pipelineStatus.map((item) => (
                  <div key={item.label} className="rounded-2xl border border-stone-200 bg-white px-3 py-3">
                    <div className="flex items-center justify-between gap-3 text-sm">
                      <span className="text-[var(--ink)]">{item.label}</span>
                      <span className="text-[var(--muted-dark)]">{item.value}</span>
                    </div>
                    <div className="mt-2 h-1.5 rounded-full bg-stone-200">
                      <div className="h-1.5 rounded-full bg-[var(--accent)]" style={{ width: item.width }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[28px] border border-stone-200 bg-[var(--paper)] p-5 shadow-[0_12px_40px_rgba(86,74,113,0.08)]">
              <p className="font-serif text-2xl text-[var(--accent)]">Quick Demo</p>
              <p className="mt-3 text-sm leading-7 text-[var(--muted-dark)]">
                Pick a CSV or XLSX file to preview the ingestion step. Auth can be added later without changing this flow.
              </p>
              <div className="mt-4 flex items-center gap-2 rounded-2xl border border-stone-200 bg-white px-4 py-3 text-sm text-[var(--muted-dark)]">
                <Icon name="lightbulb" />
                Supports CSV and XLSX only
              </div>
            </div>

            <div className="rounded-[28px] border border-stone-200 bg-[var(--paper)] p-5 shadow-[0_12px_40px_rgba(86,74,113,0.08)]">
              <p className="font-serif text-2xl text-[var(--accent)]">Why it matters</p>
              <p className="mt-3 text-sm leading-7 text-[var(--muted-dark)]">
                This UI is ready for the analysis flow today. Auth, persisted sessions, and account controls can be layered in later without changing the core workflow.
              </p>
            </div>
          </aside>
        </section>

        <footer className="mt-6 grid gap-4 rounded-[28px] border border-stone-200 bg-[rgba(247,244,237,0.92)] px-6 py-6 text-sm text-[var(--muted-dark)] md:grid-cols-[1.3fr_0.7fr_0.7fr]">
          <div>
            <div className="font-serif text-xl text-[var(--accent)]">MAGE</div>
            <p className="mt-2 max-w-md leading-7">
              Redefining data analysis through collaborative agentic workflows and rigorous academic principles.
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-[var(--muted-dark)]">Resources</p>
            <ul className="mt-3 space-y-1">
              <li>Documentation</li>
              <li>Case Studies</li>
              <li>Agent Framework</li>
            </ul>
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-[var(--muted-dark)]">Company</p>
            <ul className="mt-3 space-y-1">
              <li>Research</li>
              <li>Privacy Policy</li>
              <li>Terms of Service</li>
            </ul>
          </div>
        </footer>
      </div>
    </main>
  );
}
