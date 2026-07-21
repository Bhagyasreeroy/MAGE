'use client';

import { useRouter } from 'next/navigation';
import { useRef, useState } from 'react';

type ExpertiseLevel = 'beginner' | 'intermediate' | 'expert';

const SeedIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
  </svg>
);

const ChartUpIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
  </svg>
);

const MicroscopeIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
  </svg>
);

const EXPERTISE_OPTIONS: { value: ExpertiseLevel; label: string; icon: React.ReactNode; description: string }[] = [
  { value: 'beginner', label: 'Beginner', icon: <SeedIcon />, description: 'Plain language, step-by-step explanations' },
  { value: 'intermediate', label: 'Intermediate', icon: <ChartUpIcon />, description: 'Concise insights with key statistics' },
  { value: 'expert', label: 'Expert', icon: <MicroscopeIcon />, description: 'Dense technical recommendations' },
];

export default function NewAnalysisPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [goal, setGoal] = useState('');
  const [expertiseLevel, setExpertiseLevel] = useState<ExpertiseLevel>('intermediate');
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!goal.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('goal', goal.trim());
      formData.append('expertise_level', expertiseLevel);
      if (file) {
        formData.append('file', file);
      }

      const res = await fetch(`${apiUrl}/analysis/run`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errBody.detail ?? `HTTP ${res.status}`);
      }

      const data = await res.json();
      const id = crypto.randomUUID();
      sessionStorage.setItem(`mage-analysis-${id}`, JSON.stringify(data));
      router.push(`/dashboard/analysis/${id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="mb-10 animate-fade-in">
        <h1 className="font-[family-name:var(--font-serif)] text-4xl font-bold text-navy mb-2">
          New Analysis
        </h1>
        <p className="text-navy/50 font-light">
          Describe your goal and let MAGE&apos;s agents do the work.
        </p>
      </div>

      {/* ── Form ───────────────────────────────────────────────────── */}
      <form onSubmit={handleSubmit} className="space-y-8 animate-fade-in delay-100">
        {/* Goal Input */}
        <div className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8">
          <label htmlFor="goal-input" className="block text-xs font-bold text-navy/60 uppercase tracking-widest mb-4">
            Analytical Goal
          </label>
          <div className="relative">
            <textarea
              id="goal-input"
              rows={4}
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="e.g. Identify the top factors driving customer churn in the last 6 months…"
              className="w-full bg-cream/50 border border-dusty-rose/20 rounded-2xl px-5 py-4 text-navy placeholder:text-navy/30 resize-none focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all text-sm leading-relaxed"
              minLength={5}
              maxLength={2000}
              required
            />
            <span className="absolute bottom-4 right-5 text-xs text-navy/30 font-medium">
              {goal.length}/2000
            </span>
          </div>
        </div>

        {/* Expertise Level */}
        <div className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8">
          <label className="block text-xs font-bold text-navy/60 uppercase tracking-widest mb-5">
            Expertise Level
          </label>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {EXPERTISE_OPTIONS.map(({ value, label, icon, description }) => (
              <button
                key={value}
                type="button"
                onClick={() => setExpertiseLevel(value)}
                className={`relative p-6 rounded-[1.5rem] border text-left transition-all duration-300 cursor-pointer ${
                  expertiseLevel === value
                    ? 'bg-lavender-light/40 border-lavender shadow-lg shadow-lavender/10'
                    : 'bg-cream/40 border-dusty-rose/20 hover:border-dusty-rose/40 hover:bg-cream/80'
                }`}
              >
                {expertiseLevel === value && (
                  <span className="absolute top-4 right-4 w-2.5 h-2.5 bg-peach rounded-full" />
                )}
                <span className={`block mb-3 ${expertiseLevel === value ? 'text-navy' : 'text-navy/50'}`}>
                  {icon}
                </span>
                <p className="font-semibold text-sm text-navy mb-1.5">{label}</p>
                <p className="text-xs text-navy/50 leading-relaxed font-light">{description}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Dataset Upload */}
        <div className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8">
          <label htmlFor="dataset-file" className="block text-xs font-bold text-navy/60 uppercase tracking-widest mb-4">
            Dataset
          </label>
          <input
            ref={fileInputRef}
            id="dataset-file"
            type="file"
            accept=".csv,.tsv,.json,.parquet,.xlsx,.xls"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="w-full bg-cream/50 border border-dashed border-dusty-rose/40 rounded-2xl px-5 py-6 text-sm text-navy/60 hover:border-lavender hover:text-navy transition-all text-left"
          >
            {file ? (
              <span className="text-navy font-medium">
                {file.name} — {(file.size / 1024).toFixed(1)} KB
              </span>
            ) : (
              'Click to choose a CSV, TSV, JSON, Parquet, or Excel file…'
            )}
          </button>
          {file && (
            <button
              type="button"
              onClick={() => {
                setFile(null);
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
              className="text-xs text-navy/40 mt-3 underline underline-offset-4 hover:text-navy transition-colors"
            >
              Remove file
            </button>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="bg-dusty-rose/10 border border-dusty-rose/30 rounded-2xl p-5 text-dusty-rose text-sm flex items-start gap-4">
            <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <p className="font-bold mb-1 text-navy">Analysis failed</p>
              <p className="font-light">{error}</p>
            </div>
          </div>
        )}

        {/* Submit */}
        <button
          id="run-analysis-btn"
          type="submit"
          disabled={isLoading || goal.trim().length < 5}
          className="w-full bg-navy text-cream font-semibold py-5 rounded-[2rem] hover:bg-navy-light disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:-translate-y-1 shadow-xl shadow-navy/15 flex items-center justify-center gap-3 text-base"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin w-5 h-5 text-cream/70" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Running agents…
            </>
          ) : (
            <>
              Run Analysis
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
