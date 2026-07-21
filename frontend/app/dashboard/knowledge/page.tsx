'use client';

import { useEffect, useState } from 'react';

const DocumentIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
  </svg>
);

const InfoIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

interface KnowledgeSource {
  source: string;
  title: string;
  doc_type: string;
  chunk_count: number;
}

export default function KnowledgeBasePage() {
  const [sources, setSources] = useState<KnowledgeSource[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

  useEffect(() => {
    fetch(`${apiUrl}/analysis/knowledge-sources`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(setSources)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load'));
  }, [apiUrl]);

  return (
    <div className="max-w-4xl mx-auto">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="mb-10 animate-fade-in">
        <h1 className="font-[family-name:var(--font-serif)] text-4xl font-bold text-navy mb-2">
          Knowledge Base
        </h1>
        <p className="text-navy/50 font-light text-lg">
          The real EDA methodology sources every recommendation is grounded in and cited from.
        </p>
      </div>

      {/* ── Info Banner ────────────────────────────────────────────── */}
      <div className="bg-lavender-light/30 border border-lavender/30 rounded-3xl p-6 mb-10 flex items-start gap-4 animate-slide-up">
        <div className="text-navy-muted">
          <InfoIcon />
        </div>
        <div>
          <p className="text-sm font-bold text-navy mb-1">How Knowledge Base Works</p>
          <p className="text-sm text-navy/60 leading-relaxed font-light">
            These documents live in <code>data/knowledge_base/</code> and are embedded into the
            vector store on first use. Every recommendation MAGE produces cites one of these
            sources — this page shows exactly what it's grounded in.
          </p>
        </div>
      </div>

      {/* ── Sources List ───────────────────────────────────────────── */}
      {error && (
        <div className="bg-dusty-rose/10 border border-dusty-rose/30 rounded-2xl p-5 text-dusty-rose text-sm mb-6">
          Couldn&apos;t load knowledge sources: {error}
        </div>
      )}

      {sources === null && !error && (
        <p className="text-navy/40 font-light">Loading…</p>
      )}

      {sources && (
        <div className="space-y-4 animate-slide-up delay-100">
          {sources.map((source) => (
            <div
              key={source.source}
              className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/15 rounded-3xl p-6 flex items-center gap-5 hover:shadow-lg hover:shadow-navy/5 hover:border-dusty-rose/30 transition-all group"
            >
              <div className="w-12 h-12 bg-cream-dark/60 text-navy-muted rounded-xl flex items-center justify-center shrink-0">
                <DocumentIcon />
              </div>

              <div className="flex-1 min-w-0">
                <p className="font-bold text-base text-navy truncate group-hover:text-navy-light transition-colors mb-2">
                  {source.title}
                </p>
                <div className="flex items-center gap-4">
                  <span className="text-xs font-bold uppercase tracking-wider px-3 py-1 rounded-full border bg-sage-light/40 text-sage border-sage/30">
                    Embedded
                  </span>
                  {source.doc_type && (
                    <span className="text-xs text-navy/40 font-light">{source.doc_type}</span>
                  )}
                  <span className="text-xs text-navy/30 font-light">{source.chunk_count} chunks</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
