'use client';

import { useEffect, useState } from 'react';
import { fetchKnowledgeSources, KnowledgeSource } from '@/app/lib/api';
import { Markdown } from '../../components/markdown';

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

const SearchIcon = () => (
  <svg className="w-5 h-5 text-navy/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
  </svg>
);

const BookOpenIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>
);

const CloseIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
  </svg>
);

export default function KnowledgeBasePage() {
  const [sources, setSources] = useState<KnowledgeSource[] | null>(null);
  const [selectedSource, setSelectedSource] = useState<KnowledgeSource | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchKnowledgeSources()
      .then(setSources)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load knowledge base'));
  }, []);

  const filteredSources = sources?.filter((source) => {
    const matchesSearch =
      source.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      source.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      source.section.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesType = filterType === 'all' || source.doc_type === filterType;
    return matchesSearch && matchesType;
  });

  const categories = sources
    ? Array.from(new Set(sources.map((s) => s.doc_type).filter(Boolean)))
    : [];

  return (
    <div className="max-w-6xl mx-auto pb-12">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="mb-8 animate-fade-in flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="font-[family-name:var(--font-serif)] text-4xl font-bold text-navy mb-2">
            RAG Knowledge Base
          </h1>
          <p className="text-navy/50 font-light text-lg">
            Inspect the statistical methodologies and domain rules grounding MAGE&apos;s recommendations.
          </p>
        </div>
        <div className="text-xs font-semibold px-4 py-2 bg-sage-light/30 text-sage border border-sage/30 rounded-2xl flex items-center gap-2 self-start md:self-auto">
          <BookOpenIcon />
          <span>{sources?.length ?? 0} Methodology Documents Loaded</span>
        </div>
      </div>

      {/* ── Info Banner ────────────────────────────────────────────── */}
      <div className="bg-lavender-light/30 border border-lavender/30 rounded-3xl p-6 mb-8 flex items-start gap-4 animate-slide-up">
        <div className="text-navy-muted mt-0.5">
          <InfoIcon />
        </div>
        <div>
          <p className="text-sm font-bold text-navy mb-1">How Retrieval-Augmented Generation (RAG) Uses This</p>
          <p className="text-sm text-navy/60 leading-relaxed font-light">
            These documents live in <code>data/knowledge_base/</code> and are chunked & embedded into vector store index.
            Whenever you run an analysis, <strong>MAGE queries this knowledge base</strong> to find exact statistical rules, thresholds, and best practices relevant to your data, citing them in every recommendation card. Click any document below to inspect its full grounding methodology.
          </p>
        </div>
      </div>

      {/* ── Search & Filters ───────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row items-center gap-4 mb-6">
        <div className="relative flex-1 w-full">
          <div className="absolute inset-y-0 left-4 flex items-center pointer-events-none">
            <SearchIcon />
          </div>
          <input
            type="text"
            placeholder="Search methodologies, rules, or algorithms..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-11 pr-4 py-3 bg-warm-white/80 border border-dusty-rose/20 rounded-2xl text-navy text-sm placeholder:text-navy/30 focus:outline-none focus:border-navy/40 focus:ring-2 focus:ring-navy/5 transition-all"
          />
        </div>

        {categories.length > 0 && (
          <div className="flex items-center gap-2 self-start sm:self-auto overflow-x-auto pb-1 sm:pb-0 w-full sm:w-auto">
            <button
              onClick={() => setFilterType('all')}
              className={`px-4 py-2.5 rounded-2xl text-xs font-semibold whitespace-nowrap transition-all ${
                filterType === 'all'
                  ? 'bg-navy text-warm-white shadow-sm'
                  : 'bg-warm-white/60 text-navy/60 hover:bg-warm-white border border-dusty-rose/15'
              }`}
            >
              All Types
            </button>
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setFilterType(cat)}
                className={`px-4 py-2.5 rounded-2xl text-xs font-semibold uppercase tracking-wider whitespace-nowrap transition-all ${
                  filterType === cat
                    ? 'bg-navy text-warm-white shadow-sm'
                    : 'bg-warm-white/60 text-navy/60 hover:bg-warm-white border border-dusty-rose/15'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* ── Main Content Grid ──────────────────────────────────────── */}
      {error && (
        <div className="bg-dusty-rose/10 border border-dusty-rose/30 rounded-2xl p-5 text-dusty-rose text-sm mb-6">
          Couldn&apos;t load knowledge base: {error}
        </div>
      )}

      {sources === null && !error && (
        <div className="text-center py-12 text-navy/40 font-light">Loading knowledge base documents…</div>
      )}

      {filteredSources && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-slide-up">
          {filteredSources.map((source) => (
            <div
              key={source.source}
              onClick={() => setSelectedSource(source)}
              className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/15 rounded-3xl p-6 flex flex-col justify-between hover:shadow-lg hover:shadow-navy/5 hover:border-navy/30 transition-all cursor-pointer group"
            >
              <div>
                <div className="flex items-start justify-between gap-4 mb-3">
                  <div className="w-10 h-10 bg-cream-dark/60 text-navy-muted rounded-xl flex items-center justify-center shrink-0 group-hover:bg-navy group-hover:text-warm-white transition-colors">
                    <DocumentIcon />
                  </div>
                  <span className="text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full border bg-sage-light/40 text-sage border-sage/30">
                    {source.chunk_count} Chunks Indexed
                  </span>
                </div>

                <h3 className="font-bold text-lg text-navy mb-2 group-hover:text-navy-light transition-colors line-clamp-1">
                  {source.title}
                </h3>
                
                <p className="text-xs text-navy/60 font-light line-clamp-3 mb-4 leading-relaxed">
                  {source.content.replace(/^---[\s\S]*?---\n/, '').trim().substring(0, 160)}...
                </p>
              </div>

              <div className="flex items-center justify-between pt-3 border-t border-dusty-rose/10 text-xs text-navy/40">
                <span className="font-mono text-[11px]">{source.filename}</span>
                <span className="font-medium text-navy/70 group-hover:translate-x-0.5 transition-transform flex items-center gap-1">
                  Inspect Document &rarr;
                </span>
              </div>
            </div>
          ))}
          {filteredSources.length === 0 && (
            <div className="col-span-full text-center py-12 bg-warm-white/40 rounded-3xl border border-dashed border-dusty-rose/20 text-navy/50">
              No methodology documents found matching &quot;{searchQuery}&quot;
            </div>
          )}
        </div>
      )}

      {/* ── Document Inspection Modal ───────────────────────────────── */}
      {selectedSource && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-navy/40 backdrop-blur-sm animate-fade-in">
          <div className="bg-warm-white border border-dusty-rose/30 rounded-3xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden animate-scale-up">
            {/* Modal Header */}
            <div className="p-6 border-b border-dusty-rose/15 flex items-start justify-between gap-4 bg-cream/40">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-xs font-mono px-2.5 py-0.5 rounded-lg bg-navy/5 text-navy/60 border border-navy/10">
                    {selectedSource.filename}
                  </span>
                  {selectedSource.doc_type && (
                    <span className="text-xs font-bold uppercase tracking-wider text-sage">
                      {selectedSource.doc_type}
                    </span>
                  )}
                </div>
                <h2 className="font-bold text-xl text-navy">{selectedSource.title}</h2>
              </div>
              <button
                onClick={() => setSelectedSource(null)}
                className="p-2 text-navy/40 hover:text-navy hover:bg-navy/5 rounded-full transition-colors"
              >
                <CloseIcon />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 overflow-y-auto space-y-4 text-sm text-navy/80 leading-relaxed bg-warm-white selection:bg-lavender/30">
              <div className="bg-navy/5 rounded-2xl p-4 mb-4 border border-navy/10 text-navy/70 font-sans text-xs">
                <p className="font-bold text-navy mb-1">Vector Store Indexing Info:</p>
                <p>Path: <code>{selectedSource.source}</code> • Indexed into {selectedSource.chunk_count} chunk(s) with overlapping boundaries for semantic similarity search.</p>
              </div>
              <div className="bg-cream/30 p-6 rounded-2xl border border-dusty-rose/15 font-sans text-navy/85">
                <Markdown text={selectedSource.content.replace(/^---[\s\S]*?---\n/, '').trim()} />
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-dusty-rose/15 flex justify-end bg-cream/20">
              <button
                onClick={() => setSelectedSource(null)}
                className="px-6 py-2.5 bg-navy text-warm-white rounded-2xl font-bold text-xs hover:bg-navy-light transition-colors"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
