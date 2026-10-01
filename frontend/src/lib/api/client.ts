/**
 * Single configured Axios instance for the FastAPI backend.
 *
 * Every API module imports this client instead of calling axios
 * directly, so base URL, defaults and error normalization live
 * in exactly one place.
 *
 * The current Supabase access token is attached centrally as
 * `Authorization: Bearer <token>`. Individual API functions never
 * set the header themselves. Only public Supabase configuration
 * is read here; the frontend never holds service-role
 * credentials, and signed asset URLs are produced by the backend.
 */

import axios, { type AxiosInstance } from "axios";

import {
  getSupabaseClient,
  isSupabaseConfigured,
} from "../supabase/client";

// ============================================================
// BASE URL
// ============================================================

/**
 * Base URL of the FastAPI backend, injected at build time from
 * NEXT_PUBLIC_API_BASE_URL. The localhost fallback only covers a
 * missing local .env.local during development.
 */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// ============================================================
// TYPED API ERROR
// ============================================================

/**
 * Preserves the useful parts of a failed request: HTTP status
 * and the backend `detail` payload when one was returned.
 *
 * `detail` is intentionally `unknown`: FastAPI returns a string
 * for simple HTTPException cases and an object (for example
 * `{ message, assessment_id }`) for the endpoints in this API.
 */
export class ApiError extends Error {
  readonly status: number | null;
  readonly detail: unknown;

  constructor(message: string, status: number | null, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function detailMessage(detail: unknown): string | null {
  if (typeof detail === "string" && detail.length > 0) {
    return detail;
  }

  if (
    detail !== null &&
    typeof detail === "object" &&
    !Array.isArray(detail) &&
    "message" in detail &&
    typeof (detail as { message: unknown }).message === "string"
  ) {
    return (detail as { message: string }).message;
  }

  return null;
}

/**
 * Normalizes anything thrown by the client into an ApiError so
 * callers never have to inspect Axios internals and errors are
 * never silently swallowed.
 */
export function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) {
    return error;
  }

  if (axios.isAxiosError(error)) {
    const status = error.response?.status ?? null;

    const data: unknown = error.response?.data;

    const detail =
      data !== null && typeof data === "object" && "detail" in data
        ? (data as { detail: unknown }).detail
        : null;

    const message =
      detailMessage(detail) ??
      error.message ??
      "Request to the assessment API failed.";

    return new ApiError(message, status, detail);
  }

  if (error instanceof Error) {
    return new ApiError(error.message, null, null);
  }

  return new ApiError("Unknown API error.", null, null);
}

// ============================================================
// CLIENT
// ============================================================

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    Accept: "application/json",
  },
});

// Content-Type is deliberately not set globally: Axios derives
// application/json for plain objects and multipart/form-data
// (with its boundary) for FormData uploads.

// ============================================================
// AUTHENTICATION
// ============================================================

/**
 * Read the current Supabase access token, or null when there is
 * no session or Supabase is not configured.
 *
 * Concurrent callers share one in-flight lookup so a burst of
 * requests cannot trigger repeated session reads. No manual
 * refresh is performed: supabase-js refreshes an expired token
 * internally, so there is no refresh loop here.
 */
let inFlightToken: Promise<string | null> | null = null;

async function getAccessToken(): Promise<string | null> {
  if (!isSupabaseConfigured) {
    return null;
  }

  if (!inFlightToken) {
    inFlightToken = getSupabaseClient()
      .auth.getSession()
      .then(({ data }) => data.session?.access_token ?? null)
      .catch(() => null)
      .finally(() => {
        inFlightToken = null;
      });
  }

  return inFlightToken;
}

apiClient.interceptors.request.use(async (config) => {
  const token = await getAccessToken();

  // A public endpoint such as /health still works: no token
  // means no Authorization header at all, never a fake one.
  if (token) {
    config.headers.set("Authorization", `Bearer ${token}`);
  }

  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error: unknown) => Promise.reject(toApiError(error)),
);
