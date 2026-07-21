'use client';

import { useState } from 'react';

const FileIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
  </svg>
);

const JsonIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
  </svg>
);

const PackageIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
  </svg>
);

const UploadIcon = () => (
  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
  </svg>
);

const MOCK_DATASETS = [
  { id: '1', name: 'customer_churn.csv', type: 'CSV', rows: 7043, cols: 21, size: '1.2 MB', date: 'Jul 18, 2026' },
  { id: '2', name: 'sales_2025.parquet', type: 'Parquet', rows: 15200, cols: 34, size: '4.8 MB', date: 'Jul 15, 2026' },
  { id: '3', name: 'marketing_data.json', type: 'JSON', rows: 3200, cols: 12, size: '890 KB', date: 'Jul 10, 2026' },
  { id: '4', name: 'user_behavior.csv', type: 'CSV', rows: 42000, cols: 28, size: '6.1 MB', date: 'Jul 5, 2026' },
];

const TYPE_COLORS: Record<string, string> = {
  CSV: 'bg-sage-light/40 text-sage border-sage/20',
  Parquet: 'bg-lavender-light/50 text-navy-muted border-lavender/30',
  JSON: 'bg-peach-light/30 text-peach border-peach/20',
};

const TYPE_ICONS: Record<string, React.ReactNode> = {
  CSV: <FileIcon />,
  Parquet: <PackageIcon />,
  JSON: <JsonIcon />,
};

export default function DatasetsPage() {
  const [isDragOver, setIsDragOver] = useState(false);

  return (
    <div className="max-w-5xl mx-auto">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-10 animate-fade-in">
        <div>
          <h1 className="font-[family-name:var(--font-serif)] text-4xl font-bold text-navy mb-2">
            Datasets
          </h1>
          <p className="text-navy/50 font-light text-lg">
            Manage your uploaded data files.
          </p>
        </div>
      </div>

      {/* ── Upload Zone ────────────────────────────────────────────── */}
      <div
        className={`mb-12 animate-slide-up delay-100 border-2 border-dashed rounded-[2rem] p-12 text-center transition-all duration-300 cursor-pointer ${
          isDragOver
            ? 'border-lavender bg-lavender-light/30 scale-[1.02]'
            : 'border-dusty-rose/30 bg-warm-white/50 hover:border-dusty-rose/50 hover:bg-warm-white/80'
        }`}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setIsDragOver(false); /* TODO: handle upload */ }}
      >
        <div className="w-20 h-20 bg-lavender-light/40 text-navy-muted rounded-[1.5rem] flex items-center justify-center mx-auto mb-5">
          <UploadIcon />
        </div>
        <p className="text-navy font-bold text-xl mb-2">
          Drag & drop your files here
        </p>
        <p className="text-navy/40 font-light mb-6">
          Supports CSV, JSON, and Parquet files
        </p>
        <button className="bg-navy text-cream font-medium px-8 py-3 rounded-2xl hover:bg-navy-light transition-all hover:-translate-y-0.5 shadow-lg shadow-navy/10">
          Browse Files
        </button>
      </div>

      {/* ── Dataset List ───────────────────────────────────────────── */}
      <div className="animate-slide-up delay-200">
        <h2 className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-5 px-2">
          Your Datasets ({MOCK_DATASETS.length})
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {MOCK_DATASETS.map((ds) => (
            <div
              key={ds.id}
              className="group bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-3xl p-7 hover:shadow-xl hover:shadow-navy/5 hover:border-dusty-rose/40 transition-all duration-300"
            >
              <div className="flex items-start gap-5">
                {/* File type icon */}
                <div className="w-14 h-14 bg-cream-dark/60 text-navy-muted rounded-2xl flex items-center justify-center shrink-0">
                  {TYPE_ICONS[ds.type]}
                </div>

                <div className="flex-1 min-w-0">
                  <p className="font-bold text-navy text-base truncate mb-2 group-hover:text-navy-light transition-colors">
                    {ds.name}
                  </p>
                  <div className="flex items-center gap-3 flex-wrap">
                    <span className={`text-xs font-bold uppercase tracking-wider px-2.5 py-1 rounded-full border ${TYPE_COLORS[ds.type]}`}>
                      {ds.type}
                    </span>
                    <span className="text-xs text-navy/40 font-light">
                      {ds.rows.toLocaleString()} rows · {ds.cols} cols · {ds.size}
                    </span>
                  </div>
                  <p className="text-xs text-navy/30 mt-3 font-light">{ds.date}</p>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-3 mt-5 pt-5 border-t border-dusty-rose/15">
                <button className="text-xs font-medium text-navy/50 hover:text-navy bg-cream-dark/50 px-4 py-2 rounded-xl transition-colors">
                  Preview
                </button>
                <button className="text-xs font-medium text-navy/50 hover:text-navy bg-cream-dark/50 px-4 py-2 rounded-xl transition-colors">
                  Analyze
                </button>
                <button className="text-xs font-medium text-dusty-rose hover:text-red-500 bg-dusty-rose/10 px-4 py-2 rounded-xl transition-colors ml-auto">
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
