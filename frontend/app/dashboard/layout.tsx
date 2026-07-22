'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';
import { fetchCurrentUser, logout, type UserProfile } from '../lib/api';

const DashboardIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zm10 0a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zm10 0a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
  </svg>
);

const AnalysisIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
  </svg>
);

const DatasetsIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
  </svg>
);

const KnowledgeIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>
);

const SettingsIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

const LogoIcon = () => (
  <svg className="w-7 h-7 text-cream" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
  </svg>
);

const BellIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
  </svg>
);

const NAV_ITEMS = [
  { href: '/dashboard', label: 'Dashboard', icon: <DashboardIcon /> },
  { href: '/dashboard/analysis/new', label: 'New Analysis', icon: <AnalysisIcon /> },
  { href: '/dashboard/datasets', label: 'Datasets', icon: <DatasetsIcon /> },
  { href: '/dashboard/knowledge', label: 'Knowledge Base', icon: <KnowledgeIcon /> },
  { href: '/dashboard/settings', label: 'Settings', icon: <SettingsIcon /> },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <div className="h-screen bg-cream flex overflow-hidden">
      {/* ── Decorative Background ─────────────────────────────────────── */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-[10%] left-[20%] w-[50vw] h-[50vw] max-w-[600px] max-h-[600px] bg-lavender/20 rounded-full blur-[140px] animate-drift" />
        <div className="absolute bottom-[10%] right-[10%] w-[60vw] h-[60vw] max-w-[700px] max-h-[700px] bg-dusty-rose/15 rounded-full blur-[120px] animate-drift delay-500" />
      </div>

      {/* ── Sidebar ────────────────────────────────────────────────── */}
      <aside className="w-64 bg-navy flex flex-col shrink-0 sticky top-0 h-screen z-30">
        {/* Logo */}
        <div className="px-6 py-7 border-b border-white/5">
          <Link href="/" className="flex items-center gap-3">
            <LogoIcon />
            <span className="font-[family-name:var(--font-serif)] text-xl font-bold text-cream tracking-tight">
              MAGE
            </span>
          </Link>
        </div>

        {/* Nav Items */}
        <nav className="flex-1 px-4 py-6 space-y-2">
          {NAV_ITEMS.map((item) => {
            const isActive =
              item.href === '/dashboard'
                ? pathname === '/dashboard'
                : pathname.startsWith(item.href);

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-4 px-4 py-3.5 rounded-2xl text-sm font-medium transition-all duration-300 ${
                  isActive
                    ? 'bg-warm-white/10 text-cream shadow-sm'
                    : 'text-cream/40 hover:text-cream/80 hover:bg-white/5'
                }`}
              >
                <div className={`${isActive ? 'text-lavender-light' : ''}`}>
                  {item.icon}
                </div>
                {item.label}
                {isActive && (
                  <span className="ml-auto w-1.5 h-1.5 bg-peach rounded-full" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* User */}
        <div className="px-4 py-6 border-t border-white/5">
          <div className="flex items-center gap-4 px-3">
            <div className="w-10 h-10 bg-gradient-to-br from-lavender to-dusty-rose rounded-xl flex items-center justify-center text-sm font-bold text-navy">
              N
            </div>
            <div>
              <p className="text-sm font-medium text-cream/90">Neha</p>
              <p className="text-xs text-cream/40">Free Plan</p>
            </div>
          </div>
        </div>
      </aside>

      {/* ── Main Content ───────────────────────────────────────────── */}
      <main className="flex-1 overflow-auto relative z-20">
        {/* Top bar */}
        <header className="sticky top-0 z-20 bg-cream/80 backdrop-blur-xl border-b border-dusty-rose/10 px-8 py-4 flex items-center justify-between">
          <div />
          <div className="flex items-center gap-4">
            <button className="w-10 h-10 bg-warm-white/60 border border-dusty-rose/20 rounded-xl flex items-center justify-center text-navy/60 hover:text-navy hover:bg-warm-white hover:border-dusty-rose/40 transition-all">
              <BellIcon />
            </button>
            <Link
              href="/dashboard/analysis/new"
              className="bg-navy text-cream text-sm font-medium px-5 py-2.5 rounded-xl hover:bg-navy-light transition-all hover:-translate-y-0.5 shadow-md shadow-navy/10 flex items-center gap-2"
            >
              <AnalysisIcon />
              New Analysis
            </Link>
          </div>
        </header>

        {/* Page Content */}
        <div className="p-8">
          {children}
        </div>
      </main>
    </div>
  );
}
