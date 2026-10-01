/**
 * GET /health on the FastAPI backend.
 *
 * Small, dependency-free way to confirm the API client can reach
 * the backend. Used by the development connectivity check.
 */

import { apiClient } from "./client";
import type { HealthResponse } from "./types";

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await apiClient.get<HealthResponse>("/health");

  return data;
}
