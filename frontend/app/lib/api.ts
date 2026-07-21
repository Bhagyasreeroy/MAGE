/**
 * lib/api.ts
 * ──────────
 * Centralised API client for the MAGE backend.
 *
 * Handles:
 *  • Base URL configuration via env var
 *  • Automatic Authorization header injection
 *  • Token storage in localStorage
 *  • Automatic token refresh on 401 responses
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * Extract a human-readable message from a FastAPI error response body.
 * `detail` is a string for HTTPException, or an array of
 * {loc, msg, type} for Pydantic validation errors (422s) — never assume
 * one shape without checking, since `??` won't fall through a truthy array.
 */
export function parseApiError(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => (typeof d === "object" && d && "msg" in d ? String(d.msg) : String(d))).join(", ");
  }
  if (typeof detail === "string") {
    return detail;
  }
  return `Request failed (${status})`;
}

// ── Token storage ────────────────────────────────────────────────────────────

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("mage_access_token");
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("mage_refresh_token");
}

export function storeTokens(access: string, refresh: string): void {
  localStorage.setItem("mage_access_token", access);
  localStorage.setItem("mage_refresh_token", refresh);
}

/**
 * Store an access token without a refresh token — used for the OAuth
 * redirect flow, where the backend only issues a MAGE access token
 * (see backend/routers/oauth.py). A session started this way won't
 * auto-refresh on 401 and will require a full re-login once it expires.
 */
export function storeAccessTokenOnly(access: string): void {
  localStorage.setItem("mage_access_token", access);
}

export function clearTokens(): void {
  localStorage.removeItem("mage_access_token");
  localStorage.removeItem("mage_refresh_token");
}

// ── Generic fetch wrapper ────────────────────────────────────────────────────

interface FetchOptions extends RequestInit {
  /** If true, attach the Bearer token automatically. */
  auth?: boolean;
}

export async function apiFetch<T = unknown>(
  path: string,
  options: FetchOptions = {},
): Promise<T> {
  const { auth = false, headers: extraHeaders, ...rest } = options;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(extraHeaders as Record<string, string>),
  };

  if (auth) {
    const token = getAccessToken();
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
  }

  let response = await fetch(`${API_BASE}${path}`, { headers, ...rest });

  // If 401 and we have a refresh token, try to refresh and retry once
  if (response.status === 401 && auth) {
    const refreshed = await tryRefreshToken();
    if (refreshed) {
      headers["Authorization"] = `Bearer ${getAccessToken()}`;
      response = await fetch(`${API_BASE}${path}`, { headers, ...rest });
    }
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(parseApiError(body, response.status));
  }

  return response.json() as Promise<T>;
}

// ── Token refresh ────────────────────────────────────────────────────────────

async function tryRefreshToken(): Promise<boolean> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return false;

  try {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!res.ok) {
      clearTokens();
      return false;
    }

    const data = await res.json();
    storeTokens(data.access_token, data.refresh_token);
    return true;
  } catch {
    clearTokens();
    return false;
  }
}

// ── Auth API helpers ─────────────────────────────────────────────────────────

export interface RegisterPayload {
  email: string;
  full_name: string;
  password: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface UserProfile {
  id: string;
  email: string;
  full_name: string;
  is_active: boolean;
  created_at: string;
}

export async function registerUser(
  payload: RegisterPayload,
): Promise<UserProfile> {
  return apiFetch<UserProfile>("/auth/register", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function loginUser(
  payload: LoginPayload,
): Promise<TokenResponse> {
  const data = await apiFetch<TokenResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  storeTokens(data.access_token, data.refresh_token);
  return data;
}

export async function fetchCurrentUser(): Promise<UserProfile> {
  return apiFetch<UserProfile>("/auth/me", { auth: true });
}

export function logout(): void {
  clearTokens();
}
