'use client';

import Link from 'next/link';

const FEATURES = [
  {
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
      </svg>
    ),
    title: 'Upload Your Data',
    description: 'Drag & drop CSV, JSON, or Parquet files. MAGE handles the rest.',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
      </svg>
    ),
    title: 'Define Your Goal',
    description: 'Tell MAGE what you want to discover. It adapts the entire pipeline.',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
      </svg>
    ),
    title: 'Agents Analyze',
    description: '4 specialist agents work in concert — ingestion, mining, visualization, recommendations.',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    title: 'Get Insights',
    description: 'RAG-grounded recommendations tailored to your expertise level.',
  },
];

const STATS = [
  { value: '4', label: 'Specialist Agents' },
  { value: '3', label: 'Expertise Levels' },
  { value: '∞', label: 'Data Formats' },
  { value: 'RAG', label: 'Grounded' },
];

const TECH_PILLS = [
  'FastAPI', 'Next.js', 'ReAct Agents', 'ChromaDB', 'LangChain', 'Pandas', 'Docker',
];

const LogoIcon = () => (
  <svg className="w-6 h-6 text-navy" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
  </svg>
);

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-cream overflow-hidden">
      {/* ── Decorative Background ─────────────────────────────────────── */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute -top-[20%] -right-[10%] w-[70vw] h-[70vw] max-w-[800px] max-h-[800px] bg-lavender/30 rounded-full blur-[140px] animate-drift" />
        <div className="absolute -bottom-[20%] -left-[10%] w-[60vw] h-[60vw] max-w-[700px] max-h-[700px] bg-dusty-rose/20 rounded-full blur-[120px] animate-drift delay-500" />
        <div className="absolute top-[30%] left-[20%] w-[50vw] h-[50vw] max-w-[600px] max-h-[600px] bg-mist/20 rounded-full blur-[160px] animate-drift delay-1000" />
      </div>

      {/* ── Nav ────────────────────────────────────────────────────────── */}
      <nav className="relative z-20 flex items-center justify-between max-w-7xl mx-auto px-8 py-8">
        <Link href="/" className="flex items-center gap-3">
          <LogoIcon />
          <span className="font-[family-name:var(--font-serif)] text-2xl font-bold text-navy tracking-tight">
            MAGE
          </span>
        </Link>
        <div className="flex items-center gap-6">
          <Link
            href="/signin"
            className="text-sm font-medium text-navy/70 hover:text-navy transition-colors"
          >
            Sign In
          </Link>
          <Link
            href="/signup"
            className="text-sm font-medium bg-navy text-cream px-6 py-2.5 rounded-full hover:bg-navy-light transition-all hover:-translate-y-0.5 shadow-lg shadow-navy/10"
          >
            Get Started
          </Link>
        </div>
      </nav>

      {/* ── Hero ───────────────────────────────────────────────────────── */}
      <section className="relative z-10 max-w-5xl mx-auto px-8 pt-16 pb-24 text-center">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 bg-warm-white/60 border border-dusty-rose/20 rounded-full px-5 py-2 text-xs uppercase tracking-widest text-navy-muted font-medium mb-10 animate-fade-in backdrop-blur-md">
          <span className="w-1.5 h-1.5 bg-peach rounded-full" />
          Multi-Agent AI · Exploratory Data Analysis
        </div>

        {/* Heading */}
        <h1 className="font-[family-name:var(--font-serif)] text-6xl md:text-7xl font-bold text-navy leading-[1.15] mb-8 animate-fade-in delay-100">
          Unlock Insights with{' '}
          <span className="bg-gradient-to-r from-navy-muted via-dusty-rose to-peach bg-clip-text text-transparent animate-gradient">
            AI-Powered
          </span>{' '}
          Analysis
        </h1>

        {/* Subtitle */}
        <p className="text-lg md:text-xl text-navy/60 max-w-2xl mx-auto leading-relaxed mb-12 animate-fade-in delay-200 font-light">
          Describe your analytical goal. MAGE&apos;s specialist agents will ingest your data,
          mine patterns, and deliver{' '}
          <span className="text-navy font-medium">RAG-grounded recommendations</span>{' '}
          tailored to your expertise.
        </p>

        {/* CTA Buttons */}
        <div className="flex items-center justify-center gap-4 animate-fade-in delay-300">
          <Link
            href="/dashboard/analysis/new"
            className="bg-navy text-cream font-medium px-8 py-4 rounded-2xl hover:bg-navy-light transition-all hover:-translate-y-0.5 shadow-xl shadow-navy/10 flex items-center gap-3 text-base"
          >
            Start Analyzing
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </Link>
          <Link
            href="/dashboard"
            className="font-medium text-navy bg-warm-white/50 backdrop-blur-md border border-navy/10 px-8 py-4 rounded-2xl hover:bg-warm-white hover:border-navy/20 transition-all text-base"
          >
            View Dashboard
          </Link>
        </div>
      </section>

      {/* ── Stats Row ──────────────────────────────────────────────────── */}
      <section className="relative z-10 max-w-4xl mx-auto px-8 mb-28">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {STATS.map((stat, idx) => (
            <div
              key={stat.label}
              className={`bg-warm-white/60 backdrop-blur-md border border-dusty-rose/20 rounded-3xl p-6 text-center animate-slide-up delay-${(idx + 1) * 100}`}
            >
              <p className="font-[family-name:var(--font-serif)] text-4xl font-bold text-navy mb-2">
                {stat.value}
              </p>
              <p className="text-xs uppercase tracking-widest text-navy/50 font-medium">{stat.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── How It Works ───────────────────────────────────────────────── */}
      <section className="relative z-10 max-w-6xl mx-auto px-8 mb-32">
        <div className="text-center mb-16">
          <p className="text-xs font-medium text-dusty-rose uppercase tracking-widest mb-4">
            Process
          </p>
          <h2 className="font-[family-name:var(--font-serif)] text-4xl md:text-5xl font-bold text-navy">
            Four Steps to Insight
          </h2>
        </div>

        <div className="grid md:grid-cols-4 gap-6">
          {FEATURES.map((feature, idx) => (
            <div
              key={feature.title}
              className={`group bg-warm-white/60 backdrop-blur-md border border-dusty-rose/15 rounded-[2rem] p-8 hover:shadow-2xl hover:shadow-navy/5 hover:border-dusty-rose/30 transition-all duration-500 animate-slide-up delay-${(idx + 1) * 100}`}
            >
              <div className="flex items-center gap-4 mb-6">
                <span className="w-10 h-10 bg-lavender-light/50 rounded-2xl flex items-center justify-center text-navy-muted">
                  {feature.icon}
                </span>
                <span className="text-sm font-bold text-navy/30">
                  {String(idx + 1).padStart(2, '0')}
                </span>
              </div>
              <h3 className="font-[family-name:var(--font-serif)] text-xl font-bold text-navy mb-3">
                {feature.title}
              </h3>
              <p className="text-sm text-navy/60 leading-relaxed font-light">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Tech Pills ─────────────────────────────────────────────────── */}
      <section className="relative z-10 max-w-4xl mx-auto px-8 pb-32 text-center">
        <p className="text-xs text-navy/40 uppercase tracking-widest font-medium mb-6">
          Powered By
        </p>
        <div className="flex flex-wrap justify-center gap-3">
          {TECH_PILLS.map((tech) => (
            <span
              key={tech}
              className="bg-warm-white/60 backdrop-blur-sm border border-dusty-rose/15 rounded-full px-5 py-2.5 text-xs text-navy/60 font-medium hover:border-dusty-rose/40 hover:text-navy hover:bg-warm-white transition-all cursor-default"
            >
              {tech}
            </span>
          ))}
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────────────────────── */}
      <footer className="relative z-10 bg-navy text-cream/50 py-16">
        <div className="max-w-5xl mx-auto px-8 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3 text-cream">
            <LogoIcon />
            <span className="font-[family-name:var(--font-serif)] font-bold text-lg">MAGE</span>
          </div>
          <p className="text-sm font-light">Multi-Agent Goal-conditioned EDA</p>
          <div className="flex gap-8 text-sm">
            <Link href="/dashboard" className="hover:text-cream transition-colors">Dashboard</Link>
            <Link href="/signin" className="hover:text-cream transition-colors">Sign In</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
