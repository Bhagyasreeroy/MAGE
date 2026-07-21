'use client';

import Link from 'next/link';
import { useState, type FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import { loginUser } from '../lib/api';

const LogoIcon = () => (
  <svg className="w-10 h-10 text-cream" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
  </svg>
);

export default function SignInPage() {
  const router = useRouter();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      await loginUser({ email, password });
      router.push('/dashboard');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed.');
    } finally {
      setIsLoading(false);
    }
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
          <p className="text-navy/50 mb-6 font-light">
            Sign in to continue to your dashboard.
          </p>

          {/* Error banner */}
          {error && (
            <div className="bg-dusty-rose/10 border border-dusty-rose/30 rounded-2xl p-4 mb-6 text-dusty-rose text-sm flex items-start gap-3">
              <svg className="w-5 h-5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{error}</span>
            </div>
          )}

          {/* Google Sign-In */}
          <a
            href="http://localhost:8000/auth/google/login"
            className="w-full flex items-center justify-center gap-3 bg-warm-white/80 border border-dusty-rose/30 rounded-2xl px-5 py-4 text-navy text-sm font-medium hover:bg-warm-white hover:border-dusty-rose/50 transition-all hover:-translate-y-0.5 shadow-sm"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"/>
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
            </svg>
            Continue with Google
          </a>

          {/* Divider */}
          <div className="flex items-center gap-4 my-2">
            <div className="flex-1 h-px bg-dusty-rose/20" />
            <span className="text-xs text-navy/40 font-medium uppercase tracking-widest">or</span>
            <div className="flex-1 h-px bg-dusty-rose/20" />
          </div>

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
