/**
 * Typed wrappers for the assessment endpoints exposed by
 * backend/api/assessment.py.
 *
 * Only API communication lives here. Errors propagate as
 * ApiError from the shared client, so callers keep the HTTP
 * status and the backend `detail` payload.
 */

import { apiClient } from "./client";
import type {
  AssessmentAssetsResponse,
  AssessmentDetail,
  AssessmentHistoryResponse,
  AssessmentResponse,
  AssessmentUploadResponse,
} from "./types";

// ============================================================
// UPLOAD
//
// POST /api/assessment/upload
// multipart/form-data fields: before_image, after_image
// Validates the two images without running inference.
// ============================================================

export async function uploadAssessment(
  beforeImage: File,
  afterImage: File,
): Promise<AssessmentUploadResponse> {
  const formData = new FormData();
  formData.append("before_image", beforeImage);
  formData.append("after_image", afterImage);

  const { data } = await apiClient.post<AssessmentUploadResponse>(
    "/api/assessment/upload",
    formData,
  );

  return data;
}

// ============================================================
// ANALYZE
//
// POST /api/assessment/analyze
// multipart/form-data fields: before_image, after_image
// Runs the damage assessment pipeline and returns the full
// AssessmentResponse.
// ============================================================

export async function analyzeAssessment(
  beforeImage: File,
  afterImage: File,
): Promise<AssessmentResponse> {
  const formData = new FormData();
  formData.append("before_image", beforeImage);
  formData.append("after_image", afterImage);

  const { data } = await apiClient.post<AssessmentResponse>(
    "/api/assessment/analyze",
    formData,
  );

  return data;
}

// ============================================================
// HISTORY
//
// GET /api/assessment/history?limit=<1..100>
// Newest first. `limit` is omitted when not supplied, matching
// the optional backend query parameter.
// ============================================================

export async function getAssessmentHistory(
  limit?: number,
): Promise<AssessmentHistoryResponse> {
  const { data } = await apiClient.get<AssessmentHistoryResponse>(
    "/api/assessment/history",
    limit === undefined ? undefined : { params: { limit } },
  );

  return data;
}

// ============================================================
// DETAIL
//
// GET /api/assessment/{assessment_id}
// The id is a string on the backend, not a validated UUID, so
// it is only URL-encoded here.
// ============================================================

export async function getAssessment(
  assessmentId: string,
): Promise<AssessmentDetail> {
  const { data } = await apiClient.get<AssessmentDetail>(
    `/api/assessment/${encodeURIComponent(assessmentId)}`,
  );

  return data;
}

// ============================================================
// ASSETS
//
// GET /api/assessment/{assessment_id}/assets
// Returns short-lived signed URLs for the private before image,
// after image, prediction mask and PDF report. Missing assets
// are null.
// ============================================================

export async function getAssessmentAssets(
  assessmentId: string,
): Promise<AssessmentAssetsResponse> {
  const { data } = await apiClient.get<AssessmentAssetsResponse>(
    `/api/assessment/${encodeURIComponent(assessmentId)}/assets`,
  );

  return data;
}
