'use client';

import Link from 'next/link';
import { useState, type FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import { registerUser, loginUser } from '../lib/api';

const GOOGLE_LOGIN_URL = `${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/auth/google/login`;

const LogoIcon = () => (
  <svg className="w-10 h-10 text-cream" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
  </svg>
);

const CheckIcon = () => (
  <svg className="w-4 h-4 text-cream/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
  </svg>
);

// Synced with the backend Pydantic validator in backend/schemas/auth.py
const PASSWORD_RULES = [
  { label: 'At least 8 characters', test: (p: string) => p.length >= 8 },
  { label: 'One uppercase letter', test: (p: string) => /[A-Z]/.test(p) },
  { label: 'One lowercase letter', test: (p: string) => /[a-z]/.test(p) },
  { label: 'One digit', test: (p: string) => /\d/.test(p) },
  { label: 'One special character', test: (p: string) => /[!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?]/.test(p) },
];

export default function SignUpPage() {
  const router = useRouter();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const allRulesMet = PASSWORD_RULES.every((r) => r.test(password));
  const canSubmit = name.trim().length >= 2 && email.includes('@') && allRulesMet;

  const passwordStrength = password.length === 0 ? 0 : allRulesMet ? 3 : password.length >= 8 ? 2 : 1;
  const strengthLabels = ['', 'Weak', 'Good', 'Strong'];
  const strengthColors = ['', 'bg-red-400', 'bg-peach', 'bg-sage'];

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError('');

    if (!allRulesMet) {
      setError('Password does not meet the complexity requirements below.');
      return;
    }

    setIsLoading(true);
    try {
      await registerUser({ email, full_name: name, password });
      await loginUser({ email, password });
      router.push('/dashboard');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed.');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-cream flex">
      {/* ── Decorative Background ──────────────────────────────────── */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 right-0 w-[600px] h-[600px] bg-lavender/30 rounded-full blur-[140px] animate-drift delay-200" />
        <div className="absolute bottom-0 -left-20 w-[500px] h-[500px] bg-peach/20 rounded-full blur-[120px] animate-drift delay-700" />
      </div>

      {/* ── Left Panel — Branding ──────────────────────────────────── */}
      <div className="hidden lg:flex w-1/2 bg-navy relative overflow-hidden items-center justify-center p-16">
        <div className="absolute inset-0">
          <div className="absolute top-10 -left-10 w-[400px] h-[400px] bg-peach/10 rounded-full blur-[100px] animate-drift" />
          <div className="absolute bottom-20 right-10 w-[300px] h-[300px] bg-lavender/10 rounded-full blur-[80px] animate-drift delay-1000" />
        </div>
        <div className="relative z-10 text-center">
          <div className="flex justify-center mb-8">
            <LogoIcon />
          </div>
          <h2 className="font-[family-name:var(--font-serif)] text-5xl font-bold text-cream mb-6">
            MAGE
          </h2>
          <p className="text-cream/60 text-lg max-w-sm mx-auto leading-relaxed font-light">
            Join the future of AI-powered exploratory data analysis.
          </p>
          <div className="mt-12 space-y-4 text-left max-w-xs mx-auto">
            {[
              'Multi-agent orchestration',
              'RAG-grounded recommendations',
              'Goal-conditioned analysis',
            ].map((item, i) => (
              <div key={i} className="flex items-center gap-4 text-cream/50 text-sm font-light">
                <CheckIcon />
                <span>{item}</span>
              </div>
            ))}
          </div>
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
            Create account
          </h1>
          <p className="text-navy/50 mb-6 font-light">
            Get started with MAGE in seconds.
          </p>

          {error && (
            <div
              role="alert"
              className="flex items-center gap-2 mb-6 px-4 py-3 rounded-2xl text-sm text-red-600 bg-red-50 border border-red-200"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="shrink-0">
                <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
                <path d="M8 4.5v4M8 10.5v.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
              {error}
            </div>
          )}

          {/* Google Sign-Up */}
          <a
            href={GOOGLE_LOGIN_URL}
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
          <div className="flex items-center gap-4 my-6">
            <div className="flex-1 h-px bg-dusty-rose/20" />
            <span className="text-xs text-navy/40 font-medium uppercase tracking-widest">or</span>
            <div className="flex-1 h-px bg-dusty-rose/20" />
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="name" className="block text-xs font-semibold text-navy/70 mb-2 uppercase tracking-wide">
                Full Name
              </label>
              <input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your name"
                className="w-full bg-warm-white/80 border border-dusty-rose/30 rounded-2xl px-5 py-4 text-navy placeholder:text-navy/30 focus:outline-none focus:ring-2 focus:ring-lavender focus:border-lavender transition-all text-sm"
                required
                minLength={2}
                maxLength={200}
                autoComplete="name"
              />
            </div>

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
                autoComplete="email"
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
                minLength={8}
                maxLength={128}
                autoComplete="new-password"
              />
              {/* Password strength */}
              {password.length > 0 && (
                <>
                  <div className="mt-3 flex items-center gap-3">
                    <div className="flex-1 flex gap-1.5">
                      {[1, 2, 3].map((level) => (
                        <div
                          key={level}
                          className={`h-1.5 flex-1 rounded-full transition-colors ${
                            passwordStrength >= level ? strengthColors[passwordStrength] : 'bg-navy/10'
                          }`}
                        />
                      ))}
                    </div>
                    <span className="text-xs text-navy/40 font-medium w-12 text-right">
                      {strengthLabels[passwordStrength]}
                    </span>
                  </div>
                  <ul className="mt-3 space-y-1">
                    {PASSWORD_RULES.map((rule) => {
                      const met = rule.test(password);
                      return (
                        <li
                          key={rule.label}
                          className={`flex items-center gap-2 text-xs ${met ? 'text-sage' : 'text-navy/40'}`}
                        >
                          <span className="w-3 text-center font-bold">{met ? '✓' : '·'}</span>
                          {rule.label}
                        </li>
                      );
                    })}
                  </ul>
                </>
              )}
            </div>

            <button
              type="submit"
              disabled={isLoading || !canSubmit}
              className="w-full bg-navy text-cream font-medium py-4 rounded-2xl hover:bg-navy-light disabled:opacity-60 disabled:cursor-not-allowed transition-all hover:-translate-y-0.5 shadow-xl shadow-navy/10 flex items-center justify-center gap-3 mt-4"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Creating account…
                </>
              ) : (
                <>
                  Create Account
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </>
              )}
            </button>
          </form>

          <p className="text-center text-sm text-navy/50 mt-10">
            Already have an account?{' '}
            <Link href="/signin" className="text-navy font-semibold hover:text-navy-light transition-colors">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
