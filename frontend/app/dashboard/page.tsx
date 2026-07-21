'use client';

import Link from 'next/link';
import { useAuth } from '../lib/auth-context';

const RECENT_ANALYSES = [
  {
    id: '1',
    goal: 'Identify top factors driving customer churn in Q4 2025',
    status: 'complete',
    expertise: 'intermediate',
    date: '2 hours ago',
  },
  {
    id: '2',
    goal: 'Analyze revenue trends across product categories',
    status: 'complete',
    expertise: 'expert',
    date: 'Yesterday',
  },
  {
    id: '3',
    goal: 'Find correlations between marketing spend and conversions',
    status: 'running',
    expertise: 'beginner',
    date: 'Just now',
  },
];

const STATUS_STYLES: Record<string, string> = {
  complete: 'bg-sage-light/40 text-sage border-sage/20',
  running: 'bg-peach-light/30 text-peach border-peach/20',
  failed: 'bg-dusty-rose/20 text-dusty-rose border-dusty-rose/30',
};

const ChartIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

const FolderIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
  </svg>
);

const LightbulbIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

const ArrowRightIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
  </svg>
);

const LightningIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
  </svg>
);

const UploadIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
  </svg>
);

export default function DashboardPage() {
  const { user } = useAuth();
  const firstName = user?.full_name.split(' ')[0] ?? '';

  return (
    <div className="max-w-5xl mx-auto">
      {/* ── Welcome ────────────────────────────────────────────────── */}
      <div className="mb-12 animate-fade-in">
        <h1 className="font-[family-name:var(--font-serif)] text-4xl font-bold text-navy mb-2">
          Welcome back, {firstName}
        </h1>
        <p className="text-navy/50 text-lg font-light">
          Here&apos;s what&apos;s happening with your analyses.
        </p>
      </div>

      {/* ── Stat Cards ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        {[
          { label: 'Total Analyses', value: '12', icon: <ChartIcon />, change: '+3 this week' },
          { label: 'Datasets', value: '5', icon: <FolderIcon />, change: '2 new' },
          { label: 'Recommendations', value: '48', icon: <LightbulbIcon />, change: 'Generated' },
        ].map((stat, idx) => (
          <div
            key={stat.label}
            className={`bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-[2rem] p-8 hover:shadow-xl hover:shadow-navy/5 transition-all duration-300 animate-slide-up delay-${(idx + 1) * 100}`}
          >
            <div className="flex items-center justify-between mb-6">
              <span className="text-navy-muted bg-lavender-light/30 p-3 rounded-2xl">
                {stat.icon}
              </span>
              <span className="text-xs text-navy/40 font-medium bg-cream-dark px-3 py-1 rounded-full uppercase tracking-wider">
                {stat.change}
              </span>
            </div>
            <p className="font-[family-name:var(--font-serif)] text-4xl font-bold text-navy mb-2">
              {stat.value}
            </p>
            <p className="text-sm text-navy/50 font-medium">{stat.label}</p>
          </div>
        ))}
      </div>

      {/* ── Quick Actions ──────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12 animate-fade-in delay-300">
        <Link
          href="/dashboard/analysis/new"
          className="group bg-navy text-cream rounded-[2rem] p-8 flex items-center gap-6 hover:bg-navy-light transition-all hover:-translate-y-1 shadow-xl shadow-navy/10"
        >
          <div className="w-16 h-16 bg-cream/10 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform text-peach-light">
            <LightningIcon />
          </div>
          <div>
            <p className="font-[family-name:var(--font-serif)] text-2xl font-bold mb-1">New Analysis</p>
            <p className="text-cream/50 text-sm font-light">Start a goal-conditioned EDA</p>
          </div>
          <div className="ml-auto text-cream/30 group-hover:text-cream/60 group-hover:translate-x-1 transition-all">
            <ArrowRightIcon />
          </div>
        </Link>
        <Link
          href="/dashboard/datasets"
          className="group bg-warm-white/80 backdrop-blur-sm border-2 border-dashed border-dusty-rose/30 rounded-[2rem] p-8 flex items-center gap-6 hover:border-dusty-rose/60 hover:bg-warm-white transition-all hover:-translate-y-1"
        >
          <div className="w-16 h-16 bg-lavender-light/50 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform text-navy-muted">
            <UploadIcon />
          </div>
          <div>
            <p className="font-[family-name:var(--font-serif)] text-2xl font-bold text-navy mb-1">Upload Dataset</p>
            <p className="text-navy/50 text-sm font-light">CSV, JSON, or Parquet files</p>
          </div>
          <div className="ml-auto text-navy/20 group-hover:text-navy/40 group-hover:translate-x-1 transition-all">
            <ArrowRightIcon />
          </div>
        </Link>
      </div>

      {/* ── Recent Analyses ────────────────────────────────────────── */}
      <div className="animate-fade-in delay-400">
        <div className="flex items-center justify-between mb-6">
          <h2 className="font-[family-name:var(--font-serif)] text-2xl font-bold text-navy">
            Recent Analyses
          </h2>
          <button className="text-sm text-navy/40 hover:text-navy/60 font-medium transition-colors flex items-center gap-1">
            View All <ArrowRightIcon />
          </button>
        </div>

        <div className="space-y-4">
          {RECENT_ANALYSES.map((analysis) => (
            <Link
              key={analysis.id}
              href={`/dashboard/analysis/${analysis.id}`}
              className="group block bg-warm-white/80 backdrop-blur-sm border border-dusty-rose/20 rounded-3xl p-6 hover:shadow-lg hover:shadow-navy/5 hover:border-dusty-rose/40 transition-all"
            >
              <div className="flex items-center gap-5">
                <div className="flex-1 min-w-0">
                  <p className="text-base font-semibold text-navy group-hover:text-navy-light transition-colors truncate">
                    {analysis.goal}
                  </p>
                  <div className="flex items-center gap-3 mt-3">
                    <span className={`text-xs font-semibold px-3 py-1 rounded-full border uppercase tracking-wider ${STATUS_STYLES[analysis.status]}`}>
                      {analysis.status === 'running' && '● '}
                      {analysis.status}
                    </span>
                    <span className="text-xs text-navy/40 uppercase tracking-wider">{analysis.expertise}</span>
                  </div>
                </div>
                <span className="text-sm text-navy/40 shrink-0 font-light">{analysis.date}</span>
                <div className="text-navy/20 group-hover:text-navy/40 shrink-0 transition-colors">
                  <ArrowRightIcon />
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
