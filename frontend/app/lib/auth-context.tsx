'use client';

import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react';
import { useRouter } from 'next/navigation';
import {
  fetchCurrentUser,
  getAccessToken,
  logout as clearSession,
  storeAccessTokenOnly,
  type UserProfile,
} from './api';

/**
 * The Google OAuth callback (backend/routers/oauth.py) redirects to
 * `/dashboard?token=<jwt>` rather than going through loginUser(), since it
 * issues the token itself instead of the frontend submitting credentials.
 * Pick that token up here so an OAuth sign-in actually results in a stored
 * session, then strip it from the URL so it doesn't linger in history.
 */
function consumeOAuthTokenFromUrl(): void {
  if (typeof window === 'undefined') return;
  const params = new URLSearchParams(window.location.search);
  const token = params.get('token');
  if (!token) return;

  storeAccessTokenOnly(token);
  params.delete('token');
  const query = params.toString();
  window.history.replaceState({}, '', window.location.pathname + (query ? `?${query}` : ''));
}

interface AuthContextValue {
  user: UserProfile | null;
  isLoading: boolean;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

/**
 * Gates its children behind a valid session: redirects to /signin if there's
 * no access token or the token is rejected by /auth/me, otherwise exposes
 * the current user via useAuth().
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    consumeOAuthTokenFromUrl();

    if (!getAccessToken()) {
      router.replace('/signin');
      return;
    }

    fetchCurrentUser()
      .then((profile) => {
        if (!cancelled) setUser(profile);
      })
      .catch(() => {
        if (!cancelled) router.replace('/signin');
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [router]);

  function logout() {
    clearSession();
    setUser(null);
    router.replace('/signin');
  }

  return (
    <AuthContext.Provider value={{ user, isLoading, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

/** Must be called from a descendant of AuthProvider. */
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth() must be used within an <AuthProvider>.');
  }
  return ctx;
}
