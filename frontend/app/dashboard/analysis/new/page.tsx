'use client';

import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useRef, useState } from 'react';
import { AgentStream } from '../../../components/agent-stream';
import { MicButton } from '../../../components/mic-button';
import {
  fetchCurrentUser,
  fetchSampleDatasets,
  ingestDataset,
  loadSampleDataset,
  type SampleDataset,
} from '../../../lib/api';
import { useAnalysisStream } from '../../../lib/use-analysis-stream';
import { useVoiceInput } from '../../../lib/use-voice-input';

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

// Labels follow the three audiences named in the project spec. The stored
// values stay `beginner` / `intermediate` / `expert` — they are the API enum
// and are persisted on the user record, so only the display text changes.
const EXPERTISE_OPTIONS: { value: ExpertiseLevel; label: string; icon: React.ReactNode; description: string }[] = [
  { value: 'beginner', label: 'Beginner', icon: <SeedIcon />, description: 'Plain language, no statistical jargon' },
  { value: 'intermediate', label: 'Analyst', icon: <ChartUpIcon />, description: 'Each finding with the statistic behind it' },
  { value: 'expert', label: 'Data Scientist', icon: <MicroscopeIcon />, description: 'Full methodology, with citations in context' },
];

export default function NewAnalysisPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [goal, setGoal] = useState('');
  const [expertiseLevel, setExpertiseLevel] = useState<ExpertiseLevel>('intermediate');
  const [file, setFile] = useState<File | null>(null);
  // Covers the upload that happens *before* the socket opens; once the stream
  // starts, `stream.phase` is the source of truth for progress.
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const goalRef = useRef<HTMLTextAreaElement>(null);

  // Voice input appends rather than replaces, so a goal can be part typed and
  // part spoken, and a second recording extends the first instead of wiping
  // it. The caret is then parked at the end, ready for editing — the
  // transcript is a draft to correct, never a committed answer.
  const appendTranscript = useCallback((text: string) => {
    setGoal((current) => {
      const next = current.trim() ? `${current.trim()} ${text}` : text;
      return next.slice(0, 2000);
    });
    requestAnimationFrame(() => {
      const el = goalRef.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(el.value.length, el.value.length);
    });
  }, []);

  const voice = useVoiceInput({ onTranscript: appendTranscript });

  const [sampleDatasets, setSampleDatasets] = useState<SampleDataset[]>([]);
  const [selectedSample, setSelectedSample] = useState<SampleDataset | null>(null);
  const [sampleDatasetId, setSampleDatasetId] = useState<string | null>(null);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);

  // Either source satisfies the run: freshly-chosen file bytes, or the id of
  // a sample already persisted server-side.
  const hasDataset = file !== null || sampleDatasetId !== null;

  const stream = useAnalysisStream();
  const isRunning = isUploading || stream.phase === 'connecting' || stream.phase === 'running';

  useEffect(() => {
    fetchCurrentUser()
      .then((user) => {
        if (user.default_expertise_level) {
          setExpertiseLevel(user.default_expertise_level as ExpertiseLevel);
        }
      })
      .catch(() => {});

    fetchSampleDatasets()
      .then(setSampleDatasets)
      .catch(() => {});
  }, []);

  async function handleSelectSample(sample: SampleDataset) {
    setError(null);
    setLoadingSample(sample.filename);
    try {
      const result = await loadSampleDataset(sample.filename);
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      setSelectedSample(sample);
      setSampleDatasetId(result.dataset_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load sample dataset');
    } finally {
      setLoadingSample(null);
    }
  }

  function clearSample() {
    setSelectedSample(null);
    setSampleDatasetId(null);
  }

  // Hold on the finished trail briefly before navigating, so the last agent's
  // result is legible rather than flashing past on the way to the report.
  useEffect(() => {
    if (stream.phase !== 'complete' || !stream.runId) return;
    const timer = setTimeout(() => router.push(`/dashboard/analysis/${stream.runId}`), 900);
    return () => clearTimeout(timer);
  }, [stream.phase, stream.runId, router]);

  // React 19 deprecates `FormEvent` in favour of the event type that actually
  // fires — `SubmitEvent` for a form submission.
  async function handleSubmit(e: React.SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!goal.trim() || isRunning) return;

    setError(null);
    setIsUploading(true);

    try {
      // The socket carries a dataset id, not file bytes, so any new upload is
      // persisted over HTTP first. A selected sample dataset already has an id.
      const datasetId = file ? (await ingestDataset(file)).dataset_id : sampleDatasetId;
      stream.start({ goal, expertiseLevel, datasetId });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setIsUploading(false);
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
              ref={goalRef}
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

          {/* Voice input. The whole row is conditional, not just the button:
              on a browser that cannot record, this should look exactly like
              the page always has rather than leaving an empty gap. */}
          {voice.isSupported && (
            <div className="flex items-center justify-between gap-4 mt-4">
              <div className="flex items-center gap-3">
                <MicButton voice={voice} disabled={isRunning} />
                {voice.state === 'idle' && !voice.error && (
                  <span className="text-xs text-navy/40 font-medium">
                    …or say it out loud
                  </span>
                )}
              </div>
              {voice.error && (
                <button
                  type="button"
                  onClick={voice.dismissError}
                  title="Dismiss"
                  className="text-xs text-red-500 font-medium text-right hover:text-red-600 transition-colors"
                >
                  {voice.error}
                </button>
              )}
            </div>
          )}
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
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
              clearSample();
            }}
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

          {sampleDatasets.length > 0 && (
            <div className="mt-6 pt-6 border-t border-dusty-rose/15">
              <p className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-3">
                Or use a sample dataset
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {sampleDatasets.map((sample) => {
                  const isSelected = selectedSample?.filename === sample.filename;
                  const isLoadingThis = loadingSample === sample.filename;
                  return (
                    <button
                      key={sample.filename}
                      type="button"
                      onClick={() => (isSelected ? clearSample() : handleSelectSample(sample))}
                      disabled={loadingSample !== null}
                      className={`text-left p-4 rounded-2xl border transition-all disabled:opacity-60 ${
                        isSelected
                          ? 'bg-lavender-light/40 border-lavender'
                          : 'bg-cream/40 border-dusty-rose/20 hover:border-dusty-rose/40 hover:bg-cream/80'
                      }`}
                    >
                      <p className="font-semibold text-sm text-navy mb-1">
                        {sample.title}
                        {isSelected && <span className="text-navy ml-2 text-xs font-bold">✓ selected</span>}
                      </p>
                      <p className="text-xs text-navy/50 leading-relaxed font-light mb-1.5">
                        {sample.description}
                      </p>
                      <p className="text-[0.65rem] text-navy/30 font-medium">
                        {sample.filename} · {sample.size_kb} KB
                      </p>
                      {isLoadingThis && <p className="text-xs text-navy/50 mt-2">Loading…</p>}
                    </button>
                  );
                })}
              </div>
            </div>
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
        {/* A run needs a dataset as much as it needs a goal. The API refuses a
            run with neither (422 from MissingDataSourceError), so this only
            stops the pointless round-trip and says which input is missing —
            the API check is the real guard, since this button is not the only
            way in. */}
        {!hasDataset && goal.trim().length >= 5 && (
          <p className="text-navy/40 font-light text-sm text-center -mt-4">
            Upload a dataset or pick a sample above to run this analysis.
          </p>
        )}
        <button
          id="run-analysis-btn"
          type="submit"
          disabled={isRunning || stream.phase === 'complete' || goal.trim().length < 5 || !hasDataset}
          className="w-full bg-navy text-cream font-semibold py-5 rounded-[2rem] hover:bg-navy-light disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:-translate-y-1 shadow-xl shadow-navy/15 flex items-center justify-center gap-3 text-base"
        >
          {isRunning || stream.phase === 'complete' ? (
            <>
              <svg className="animate-spin w-5 h-5 text-cream/70" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              {isUploading && 'Uploading dataset…'}
              {stream.phase === 'connecting' && 'Connecting…'}
              {stream.phase === 'running' && 'Running agents…'}
              {stream.phase === 'complete' && 'Opening your report…'}
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

        {/* Live agent trail — appears as soon as a run starts */}
        <AgentStream phase={stream.phase} steps={stream.steps} error={stream.error} />

        {stream.phase === 'error' && (
          <button
            type="button"
            onClick={stream.cancel}
            className="w-full text-sm text-navy/50 underline underline-offset-4 hover:text-navy transition-colors"
          >
            Reset and try again
          </button>
        )}
      </form>
    </div>
  );
}
