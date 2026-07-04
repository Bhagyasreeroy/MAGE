"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { registerUser, loginUser } from "../lib/api";

/**
 * Password strength rules — synced with the backend Pydantic validator.
 */
const PASSWORD_RULES = [
  { label: "At least 8 characters", test: (p: string) => p.length >= 8 },
  { label: "One uppercase letter", test: (p: string) => /[A-Z]/.test(p) },
  { label: "One lowercase letter", test: (p: string) => /[a-z]/.test(p) },
  { label: "One digit", test: (p: string) => /\d/.test(p) },
  {
    label: "One special character",
    test: (p: string) => /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(p),
  },
];

export default function RegisterPage() {
  const router = useRouter();

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const passwordsMatch = password === confirmPassword;
  const allRulesMet = PASSWORD_RULES.every((r) => r.test(password));
  const canSubmit =
    fullName.trim().length >= 2 &&
    email.includes("@") &&
    allRulesMet &&
    passwordsMatch &&
    confirmPassword.length > 0;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");

    if (!passwordsMatch) {
      setError("Passwords do not match.");
      return;
    }
    if (!allRulesMet) {
      setError("Password does not meet complexity requirements.");
      return;
    }

    setLoading(true);

    try {
      await registerUser({ email, full_name: fullName, password });
      // Auto-login after successful registration
      await loginUser({ email, password });
      router.push("/");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Registration failed.",
      );
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
          <h1 className="auth-title">Create account</h1>
          <p className="auth-subtitle">
            Get started with MAGE
          </p>
        </div>

        {/* ── Error banner ────────────────────────────────────── */}
        {error && (
          <div className="auth-error" role="alert" id="register-error">
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
            <label htmlFor="register-name" className="form-label">
              Full name
            </label>
            <input
              id="register-name"
              type="text"
              className="form-input"
              placeholder="John Doe"
              required
              minLength={2}
              maxLength={200}
              autoComplete="name"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label htmlFor="register-email" className="form-label">
              Email
            </label>
            <input
              id="register-email"
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
            <label htmlFor="register-password" className="form-label">
              Password
            </label>
            <div className="password-wrapper">
              <input
                id="register-password"
                type={showPassword ? "text" : "password"}
                className="form-input"
                placeholder="••••••••"
                required
                minLength={8}
                maxLength={128}
                autoComplete="new-password"
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

            {/* ── Password strength checklist ──────────────────── */}
            {password.length > 0 && (
              <ul className="password-rules" id="password-rules">
                {PASSWORD_RULES.map((rule) => (
                  <li
                    key={rule.label}
                    className={
                      rule.test(password) ? "rule-pass" : "rule-fail"
                    }
                  >
                    <span className="rule-icon">
                      {rule.test(password) ? "✓" : "✗"}
                    </span>
                    {rule.label}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="register-confirm" className="form-label">
              Confirm password
            </label>
            <input
              id="register-confirm"
              type={showPassword ? "text" : "password"}
              className="form-input"
              placeholder="••••••••"
              required
              autoComplete="new-password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
            />
            {confirmPassword.length > 0 && !passwordsMatch && (
              <p className="field-error" id="confirm-mismatch">
                Passwords do not match.
              </p>
            )}
          </div>

          <button
            id="register-submit"
            type="submit"
            className="auth-button"
            disabled={!canSubmit || loading}
          >
            {loading ? (
              <span className="button-spinner" />
            ) : (
              "Create account"
            )}
          </button>
        </form>

        {/* ── Footer ──────────────────────────────────────────── */}
        <p className="auth-footer">
          Already have an account?{" "}
          <Link href="/signin" className="auth-link" id="goto-signin">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
