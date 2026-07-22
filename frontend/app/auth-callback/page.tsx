'use client';

import { useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { storeAccessTokenOnly } from '../../lib/api';

/**
 * /auth-callback
 * ──────────────
 * Landing page for the Google OAuth redirect.
 * Reads the `token` query param, stores it, then sends the user to the dashboard.
 */
export default function AuthCallbackPage() {
  const router = useRouter();
  const params = useSearchParams();

  useEffect(() => {
    const token = params.get('token');
    if (token) {
      storeAccessTokenOnly(token);
      router.replace('/dashboard');
    } else {
      // No token — something went wrong, send back to signin
      router.replace('/signin');
    }
  }, [params, router]);

  return (
    <div className="min-h-screen bg-cream flex items-center justify-center">
      <div className="text-center animate-fade-in">
        <div className="w-12 h-12 border-2 border-navy/20 border-t-navy rounded-full animate-spin mx-auto mb-4" />
        <p className="text-navy/60 font-light text-sm">Signing you in…</p>
      </div>
    </div>
  );
}
