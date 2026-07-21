'use client';

import Link from 'next/link';
import { useRef, useState } from 'react';
import { parseApiError } from '../../lib/api';

const FileIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
  </svg>
);

const UploadIcon = () => (
  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
  </svg>
);

interface ColumnStats {
  min: number | null;
  max: number | null;
  mean: number | null;
  unique_count: number | null;
}

interface ColumnSummary {
  name: string;
  dtype: string;
  missing_count: number;
  stats: ColumnStats | null;
}

interface IngestionResult {
  row_count: number;
  column_count: number;
  column_summary: ColumnSummary[];
  warnings: string[];
}

interface UploadedDataset {
  id: string;
  name: string;
  size: string;
  result: IngestionResult;
  expanded: boolean;
}

export default function DatasetsPage() {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<UploadedDataset[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

  async function ingestFile(file: File) {
    setIsUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${apiUrl}/analysis/ingest`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(parseApiError(errBody, res.status));
      }

      const result: IngestionResult = await res.json();
      setDatasets((prev) => [
        {
          id: crypto.randomUUID(),
          name: file.name,
          size: `${(file.size / 1024).toFixed(1)} KB`,
          result,
          expanded: false,
        },
        ...prev,
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) ingestFile(file);
  }

  function toggleExpanded(id: string) {
    setDatasets((prev) => prev.map((d) => (d.id === id ? { ...d, expanded: !d.expanded } : d)));
  }

  function removeDataset(id: string) {
    setDatasets((prev) => prev.filter((d) => d.id !== id));
  }

  return (
    <div className="max-w-5xl mx-auto">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-10 animate-fade-in">
        <div>
          <h1 className="font-[family-name:var(--font-serif)] text-4xl font-bold text-navy mb-2">
            Datasets
          </h1>
          <p className="text-navy/50 font-light text-lg">
            Upload a file to profile it. Datasets aren&apos;t stored between sessions yet —
            upload again here whenever you want to inspect a file.
          </p>
        </div>
      </div>

      {/* ── Upload Zone ────────────────────────────────────────────── */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.tsv,.json,.parquet,.xlsx,.xls"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) ingestFile(file);
          e.target.value = '';
        }}
      />
      <div
        className={`mb-6 animate-slide-up delay-100 border-2 border-dashed rounded-[2rem] p-12 text-center transition-all duration-300 cursor-pointer ${
          isDragOver
            ? 'border-lavender bg-lavender-light/30 scale-[1.02]'
            : 'border-dusty-rose/30 bg-warm-white/50 hover:border-dusty-rose/50 hover:bg-warm-white/80'
        }`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
      >
        <div className="w-20 h-20 bg-lavender-light/40 text-navy-muted rounded-[1.5rem] flex items-center justify-center mx-auto mb-5">
          <UploadIcon />
        </div>
        <p className="text-navy font-bold text-xl mb-2">
          {isUploading ? 'Profiling…' : 'Drag & drop your file here'}
        </p>
        <p className="text-navy/40 font-light mb-6">
          Supports CSV, TSV, JSON, Parquet, and Excel files
        </p>
        <button
          type="button"
          disabled={isUploading}
          onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
          className="bg-navy text-cream font-medium px-8 py-3 rounded-2xl hover:bg-navy-light transition-all hover:-translate-y-0.5 shadow-lg shadow-navy/10 disabled:opacity-50"
        >
          Browse Files
        </button>
      </div>

      {error && (
        <div className="mb-6 bg-dusty-rose/10 border border-dusty-rose/30 rounded-2xl p-5 text-dusty-rose text-sm">
          {error}
        </div>
      )}

      {/* ── Dataset List ───────────────────────────────────────────── */}
      <div className="animate-slide-up delay-200">
        <h2 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-5 px-2">
          Profiled This Session ({datasets.length})
        </h2>

        {datasets.length === 0 ? (
          <p className="text-navy/40 font-light px-2">Nothing uploaded yet.</p>
        ) : (
          <div className="space-y-5">
            {datasets.map((ds) => (
              <div
                key={ds.id}
                className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-3xl p-7"
              >
                <div className="flex items-start gap-5">
                  <div className="w-14 h-14 bg-cream-dark/60 text-navy-muted rounded-2xl flex items-center justify-center shrink-0">
                    <FileIcon />
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className="font-bold text-navy text-base truncate mb-2">{ds.name}</p>
                    <span className="text-xs text-navy/40 font-light">
                      {ds.result.row_count.toLocaleString()} rows · {ds.result.column_count} cols · {ds.size}
                    </span>
                    {ds.result.warnings.length > 0 && (
                      <p className="text-xs text-peach mt-2 font-medium">
                        {ds.result.warnings.length} data-quality warning(s)
                      </p>
                    )}
                  </div>
                </div>

                {ds.expanded && (
                  <div className="mt-5 pt-5 border-t border-dusty-rose/15 space-y-3">
                    {ds.result.warnings.map((w, i) => (
                      <p key={i} className="text-xs text-navy/60">⚠ {w}</p>
                    ))}
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {ds.result.column_summary.map((col) => (
                        <div key={col.name} className="bg-cream/50 rounded-xl p-3">
                          <p className="text-xs font-bold text-navy truncate">{col.name}</p>
                          <p className="text-[10px] text-navy/40 uppercase tracking-wide">{col.dtype}</p>
                          <p className="text-[10px] text-navy/40 mt-1">{col.missing_count} missing</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Actions */}
                <div className="flex items-center gap-3 mt-5 pt-5 border-t border-dusty-rose/15">
                  <button
                    onClick={() => toggleExpanded(ds.id)}
                    className="text-xs font-medium text-navy/50 hover:text-navy bg-cream-dark/50 px-4 py-2 rounded-xl transition-colors"
                  >
                    {ds.expanded ? 'Hide Details' : 'Preview'}
                  </button>
                  <Link
                    href="/dashboard/analysis/new"
                    className="text-xs font-medium text-navy/50 hover:text-navy bg-cream-dark/50 px-4 py-2 rounded-xl transition-colors"
                  >
                    Analyze
                  </Link>
                  <button
                    onClick={() => removeDataset(ds.id)}
                    className="text-xs font-medium text-dusty-rose hover:text-red-500 bg-dusty-rose/10 px-4 py-2 rounded-xl transition-colors ml-auto"
                  >
                    Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
