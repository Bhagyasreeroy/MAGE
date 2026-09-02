'use client';

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { deleteDataset, fetchDatasets, type DatasetSummary } from '../../lib/api';
import { UPLOAD_ACCEPT, uploadAnyFile } from '../../lib/upload-routing';

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


interface DatasetEntry {
  id: string;
  name: string;
  rowCount: number | null;
  columnCount: number | null;
  warnings: string[];
}

export default function DatasetsPage() {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoadingList, setIsLoadingList] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchDatasets()
      .then((list: DatasetSummary[]) => {
        setDatasets(
          list.map((d) => ({
            id: d.id,
            name: d.filename,
            rowCount: d.row_count,
            columnCount: d.column_count,
            warnings: [],
          })),
        );
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load datasets'))
      .finally(() => setIsLoadingList(false));
  }, []);

  async function ingestFile(file: File) {
    setIsUploading(true);
    setError(null);
    try {
      // Images and PDFs are OCR'd into a dataset; everything else is
      // ingested directly. The caller does not need to know which happened.
      const result = await uploadAnyFile(file);
      setDatasets((prev) => [
        {
          id: result.datasetId || crypto.randomUUID(),
          name: result.filename,
          rowCount: result.rowCount,
          columnCount: result.columnCount,
          warnings: result.warnings,
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

  async function removeDataset(id: string) {
    const previous = datasets;
    setDatasets((prev) => prev.filter((d) => d.id !== id));
    try {
      await deleteDataset(id);
    } catch (err) {
      setDatasets(previous);
      setError(err instanceof Error ? err.message : 'Failed to delete dataset');
    }
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
            Every file you upload is saved to your account — visible here on any device you sign into.
          </p>
        </div>
      </div>

      {/* ── Upload Zone ────────────────────────────────────────────── */}
      <input
        ref={fileInputRef}
        type="file"
        accept={UPLOAD_ACCEPT}
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
          {isUploading ? 'Reading your file…' : 'Drag & drop your file here'}
        </p>
        <p className="text-navy/40 font-light mb-6">
          CSV, TSV, JSON, Parquet and Excel — or a photo, scan or PDF of a table, which is read with OCR
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
          Your Datasets ({datasets.length})
        </h2>

        {isLoadingList ? (
          <p className="text-navy/40 font-light px-2">Loading…</p>
        ) : datasets.length === 0 ? (
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
                      {ds.rowCount?.toLocaleString() ?? '?'} rows · {ds.columnCount ?? '?'} cols
                    </span>
                    {ds.warnings.length > 0 && (
                      <p className="text-xs text-peach mt-2 font-medium">
                        {ds.warnings.length} data-quality warning(s)
                      </p>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-3 mt-5 pt-5 border-t border-dusty-rose/15">
                  <Link
                    href={`/dashboard/datasets/${ds.id}`}
                    className="text-xs font-medium text-navy/50 hover:text-navy bg-cream-dark/50 px-4 py-2 rounded-xl transition-colors"
                  >
                    Open
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
