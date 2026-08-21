'use client';

import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import { fetchSharedThread } from '../../lib/api';
import { Markdown } from '../../components/markdown';

interface SharedRun {
  goal: string;
  recommendations: string[];
  rag_sources: string[];
  summary: string;
  run_id?: string | null;
}

const LogoIcon = () => (
  <svg className="w-8 h-8 text-cream" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
  </svg>
);

const DocumentIcon = () => (
  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

export default function SharedConversationPage() {
  const params = useParams<{ runId: string }>();
  const [thread, setThread] = useState<SharedRun[] | null | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    fetchSharedThread(params.runId)
      .then((data) => {
        if (!cancelled) setThread(data as SharedRun[]);
      })
      .catch(() => {
        if (!cancelled) setThread(null);
      });
    return () => {
      cancelled = true;
    };
  }, [params.runId]);

  return (
    <div className="min-h-screen bg-cream">
      <header className="border-b border-dusty-rose/15 bg-warm-white/80 backdrop-blur-sm">
        <div className="max-w-3xl mx-auto px-6 py-5 flex items-center gap-3">
          <span className="font-[family-name:var(--font-serif)] text-xl font-bold text-navy">MAGE</span>
          <span className="text-navy/40 font-light text-sm ml-1">Shared analysis</span>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-12">
        {thread === undefined && null}

        {thread === null && (
          <div className="text-center py-20">
            <h1 className="font-[family-name:var(--font-serif)] text-3xl font-bold text-navy mb-4">
              This link isn&apos;t available
            </h1>
            <p className="text-navy/50 font-light mb-8">
              This conversation is no longer shared, or the link is wrong.
            </p>
            <Link
              href="/"
              className="inline-block bg-navy text-cream font-semibold px-8 py-4 rounded-2xl hover:bg-navy-light transition-all"
            >
              Go to MAGE
            </Link>
          </div>
        )}

        {thread && thread.length > 0 && (
          <div className="space-y-8">
            {thread.map((run, i) => (
              <div
                key={run.run_id ?? i}
                className="bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8 shadow-sm shadow-navy/5"
              >
                <p className="text-xs font-bold text-navy/40 uppercase tracking-widest mb-2">
                  {i === 0 ? 'Goal' : 'Follow-up'}
                </p>
                <p className="text-navy font-medium text-lg mb-6">{run.goal}</p>

                {run.recommendations.length > 0 ? (
                  <ul className="space-y-3 mb-6">
                    {run.recommendations.map((rec, idx) => (
                      <li key={idx} className="flex gap-3 text-navy/70 font-light">
                        <span className="text-navy-muted bg-cream-dark w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5">
                          {idx + 1}
                        </span>
                        <Markdown text={rec} className="text-sm flex-1 min-w-0" />
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-navy/40 font-light text-sm mb-6">No recommendations were grounded for this goal.</p>
                )}

                {run.rag_sources.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {run.rag_sources.map((src, idx) => (
                      <span
                        key={idx}
                        className="bg-cream-dark/60 border border-dusty-rose/20 rounded-xl px-3 py-1.5 text-xs text-navy/60 font-light flex items-center gap-1.5"
                      >
                        <DocumentIcon /> {src}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        <div className="text-center mt-14 pt-10 border-t border-dusty-rose/15">
          <p className="text-navy/50 font-light mb-4">Sign in to run your own analysis with MAGE.</p>
          <Link
            href="/signin"
            className="inline-block bg-navy text-cream font-semibold px-8 py-3.5 rounded-2xl hover:bg-navy-light transition-all"
          >
            Sign In
          </Link>
        </div>
      </main>
    </div>
  );
}
