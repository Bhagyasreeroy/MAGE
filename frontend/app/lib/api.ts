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

  const response = await fetchWithAuthRetry(path, headers, auth, rest);

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(parseApiError(body, response.status));
  }

  return response.json() as Promise<T>;
}

/**
 * Like apiFetch, but for multipart/form-data bodies (file uploads) — never
 * sets Content-Type itself, since the browser must set it (with the
 * multipart boundary) when the body is a FormData instance. Always attaches
 * the auth token, since every multipart endpoint in this app requires it.
 */
export async function authFetchFormData<T = unknown>(
  path: string,
  formData: FormData,
): Promise<T> {
  const response = await fetchWithAuthRetry(path, {}, true, { method: "POST", body: formData });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(parseApiError(body, response.status));
  }

  return response.json() as Promise<T>;
}

async function fetchWithAuthRetry(
  path: string,
  headers: Record<string, string>,
  auth: boolean,
  rest: RequestInit,
): Promise<Response> {
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

  return response;
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
  default_expertise_level: string;
  created_at: string;
}

export interface UpdateProfilePayload {
  full_name?: string;
  email?: string;
  default_expertise_level?: string;
}

export interface AnalysisRunSummary {
  id: string;
  goal: string;
  expertise_level: string;
  status: string;
  summary: string;
  dataset_id: string | null;
  created_at: string;
}

export interface DatasetSummary {
  id: string;
  filename: string;
  row_count: number | null;
  column_count: number | null;
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

export async function updateProfile(payload: UpdateProfilePayload): Promise<UserProfile> {
  return apiFetch<UserProfile>("/auth/me", {
    method: "PATCH",
    auth: true,
    body: JSON.stringify(payload),
  });
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  await apiFetch("/auth/change-password", {
    method: "POST",
    auth: true,
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
}

export async function deleteAccount(): Promise<void> {
  await apiFetch("/auth/me", { method: "DELETE", auth: true });
}

export function logout(): void {
  clearTokens();
}

// ── Analysis history / datasets ─────────────────────────────────────────────

export async function fetchAnalysisHistory(): Promise<AnalysisRunSummary[]> {
  return apiFetch<AnalysisRunSummary[]>("/analysis/history", { auth: true });
}

export async function fetchAnalysisRun(runId: string): Promise<unknown> {
  return apiFetch(`/analysis/history/${runId}`, { auth: true });
}

export async function fetchDatasets(): Promise<DatasetSummary[]> {
  return apiFetch<DatasetSummary[]>("/analysis/datasets", { auth: true });
}

export async function deleteDataset(datasetId: string): Promise<void> {
  await apiFetch(`/analysis/datasets/${datasetId}`, { method: "DELETE", auth: true });
}
