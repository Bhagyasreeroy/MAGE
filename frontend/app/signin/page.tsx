'use client';

import Link from 'next/link';
import { useState } from 'react';

const LogoIcon = () => (
  <svg className="w-10 h-10 text-cream" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
  </svg>
);

export default function SignInPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      window.location.href = '/dashboard';
    }, 1000);
  }

  return (
    <div className="min-h-screen bg-cream flex">
      {/* ── Decorative Background ──────────────────────────────────── */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 right-0 w-[600px] h-[600px] bg-lavender/30 rounded-full blur-[140px] animate-drift" />
        <div className="absolute bottom-0 -left-20 w-[500px] h-[500px] bg-dusty-rose/20 rounded-full blur-[120px] animate-drift delay-500" />
      </div>

      {/* ── Left Panel — Branding ──────────────────────────────────── */}
      <div className="hidden lg:flex w-1/2 bg-navy relative overflow-hidden items-center justify-center p-16">
        <div className="absolute inset-0">
          <div className="absolute top-10 -left-10 w-[400px] h-[400px] bg-lavender/10 rounded-full blur-[100px] animate-drift" />
          <div className="absolute bottom-20 right-10 w-[300px] h-[300px] bg-peach/10 rounded-full blur-[80px] animate-drift delay-1000" />
        </div>
        <div className="relative z-10 text-center">
          <div className="flex justify-center mb-8">
            <LogoIcon />
          </div>
          <h2 className="font-[family-name:var(--font-serif)] text-5xl font-bold text-cream mb-6">
            MAGE
          </h2>
          <p className="text-cream/60 text-lg max-w-sm mx-auto leading-relaxed font-light">
            Your AI-powered data analysis companion. Define a goal, let agents do the rest.
          </p>
        </div>
      </div>

      {/* ── Right Panel — Form ─────────────────────────────────────── */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8 relative z-10">
        <div className="w-full max-w-md animate-fade-in">
          {/* Mobile logo */}
          <div className="lg:hidden text-center mb-10">
            <div className="flex justify-center text-navy mb-4">
              <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
              </svg>
            </div>
            <h1 className="font-[family-name:var(--font-serif)] text-3xl font-bold text-navy">MAGE</h1>
          </div>

          <h1 className="font-[family-name:var(--font-serif)] text-3xl font-bold text-navy mb-2">
            Welcome back
          </h1>
          <p className="text-navy/50 mb-10 font-light">
            Sign in to continue to your dashboard.
          </p>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="email" className="block text-xs font-semibold text-navy/70 mb-2 uppercase tracking-wide">
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full bg-warm-white/80 border border-dusty-rose/30 rounded-2xl px-5 py-4 text-navy placeholder:text-navy/30 focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all text-sm"
                required
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-xs font-semibold text-navy/70 mb-2 uppercase tracking-wide">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-warm-white/80 border border-dusty-rose/30 rounded-2xl px-5 py-4 text-navy placeholder:text-navy/30 focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all text-sm"
                required
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-navy text-cream font-medium py-4 rounded-2xl hover:bg-navy-light disabled:opacity-60 disabled:cursor-not-allowed transition-all hover:-translate-y-0.5 shadow-xl shadow-navy/10 flex items-center justify-center gap-3 mt-4"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Signing in…
                </>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          <p className="text-center text-sm text-navy/50 mt-10">
            Don&apos;t have an account?{' '}
            <Link href="/signup" className="text-navy font-semibold hover:text-navy-light transition-colors">
              Create one
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
