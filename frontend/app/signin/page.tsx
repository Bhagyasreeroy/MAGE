"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { loginUser } from "../lib/api";

export default function SignInPage() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await loginUser({ email, password });
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        {/* ── Header ──────────────────────────────────────────── */}
        <div className="auth-header">
          <div className="auth-logo">🧙</div>
          <h1 className="auth-title">Welcome back</h1>
          <p className="auth-subtitle">Sign in to your MAGE account</p>
        </div>

        {/* ── Error banner ────────────────────────────────────── */}
        {error && (
          <div className="auth-error" role="alert" id="login-error">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
              <path d="M8 4.5v4M8 10.5v.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            {error}
          </div>
        )}

        {/* ── Form ────────────────────────────────────────────── */}
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label htmlFor="signin-email" className="form-label">
              Email
            </label>
            <input
              id="signin-email"
              type="email"
              className="form-input"
              placeholder="you@example.com"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label htmlFor="signin-password" className="form-label">
              Password
            </label>
            <div className="password-wrapper">
              <input
                id="signin-password"
                type={showPassword ? "text" : "password"}
                className="form-input"
                placeholder="••••••••"
                required
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                    <line x1="1" y1="1" x2="23" y2="23" />
                  </svg>
                ) : (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                )}
              </button>
            </div>
          </div>

          <button
            id="signin-submit"
            type="submit"
            className="auth-button"
            disabled={loading}
          >
            {loading ? (
              <span className="button-spinner" />
            ) : (
              "Sign in"
            )}
          </button>
        </form>

        {/* ── Footer ──────────────────────────────────────────── */}
        <p className="auth-footer">
          Don&apos;t have an account?{" "}
          <Link href="/register" className="auth-link" id="goto-register">
            Create one
          </Link>
        </p>
      </div>
    </div>
  );
}
