'use client';

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

const PlusIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
  </svg>
);

const MOCK_SOURCES = [
  { id: '1', name: 'churn_analysis_handbook.pdf', status: 'embedded', chunks: 142, date: 'Jul 18, 2026' },
  { id: '2', name: 'retention_best_practices.md', status: 'embedded', chunks: 56, date: 'Jul 15, 2026' },
  { id: '3', name: 'telecom_industry_report_2025.pdf', status: 'embedded', chunks: 208, date: 'Jul 12, 2026' },
  { id: '4', name: 'data_science_patterns.pdf', status: 'pending', chunks: 0, date: 'Jul 20, 2026' },
];

const STATUS_MAP: Record<string, { label: string; style: string; icon: React.ReactNode }> = {
  embedded: { 
    label: 'Embedded', 
    style: 'bg-sage-light/40 text-sage border-sage/30', 
    icon: <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg> 
  },
  pending: { 
    label: 'Pending', 
    style: 'bg-peach-light/30 text-peach border-peach/30', 
    icon: <svg className="w-3 h-3 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg> 
  },
  error: { 
    label: 'Error', 
    style: 'bg-dusty-rose/20 text-dusty-rose border-dusty-rose/30', 
    icon: <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg> 
  },
};

export default function KnowledgeBasePage() {
  return (
    <div className="max-w-4xl mx-auto">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-10 animate-fade-in">
        <div>
          <h1 className="font-[family-name:var(--font-serif)] text-4xl font-bold text-navy mb-2">
            Knowledge Base
          </h1>
          <p className="text-navy/50 font-light text-lg">
            Manage RAG knowledge sources for grounded recommendations.
          </p>
        </div>
        <button className="bg-navy text-cream font-medium px-6 py-3 rounded-2xl hover:bg-navy-light transition-all hover:-translate-y-0.5 shadow-lg shadow-navy/10 flex items-center gap-2">
          <PlusIcon />
          Add Source
        </button>
      </div>

      {/* ── Info Banner ────────────────────────────────────────────── */}
      <div className="bg-lavender-light/30 border border-lavender/30 rounded-3xl p-6 mb-10 flex items-start gap-4 animate-slide-up">
        <div className="text-navy-muted">
          <InfoIcon />
        </div>
        <div>
          <p className="text-sm font-bold text-navy mb-1">How Knowledge Base Works</p>
          <p className="text-sm text-navy/60 leading-relaxed font-light">
            Upload documents (PDF, Markdown, text) to create a knowledge base. MAGE will embed these
            into the vector store and use them to ground recommendations with real sources during analysis.
          </p>
        </div>
      </div>

      {/* ── Sources List ───────────────────────────────────────────── */}
      <div className="space-y-4 animate-slide-up delay-100">
        {MOCK_SOURCES.map((source) => {
          const status = STATUS_MAP[source.status];
          return (
            <div
              key={source.id}
              className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/15 rounded-3xl p-6 flex items-center gap-5 hover:shadow-lg hover:shadow-navy/5 hover:border-dusty-rose/30 transition-all group"
            >
              {/* Icon */}
              <div className="w-12 h-12 bg-cream-dark/60 text-navy-muted rounded-xl flex items-center justify-center shrink-0">
                <DocumentIcon />
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <p className="font-bold text-base text-navy truncate group-hover:text-navy-light transition-colors mb-2">
                  {source.name}
                </p>
                <div className="flex items-center gap-4">
                  <span className={`text-xs font-bold uppercase tracking-wider px-3 py-1 rounded-full border flex items-center gap-1.5 ${status.style}`}>
                    {status.icon} {status.label}
                  </span>
                  {source.chunks > 0 && (
                    <span className="text-xs text-navy/40 font-light">{source.chunks} chunks</span>
                  )}
                  <span className="text-xs text-navy/30 font-light">{source.date}</span>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-3 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                <button className="text-xs font-medium text-navy/50 hover:text-navy bg-cream-dark/50 px-4 py-2 rounded-xl transition-colors">
                  Re-embed
                </button>
                <button className="text-xs font-medium text-dusty-rose hover:text-red-500 bg-dusty-rose/10 px-4 py-2 rounded-xl transition-colors">
                  Remove
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
