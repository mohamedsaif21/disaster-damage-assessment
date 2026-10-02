/**
 * User-facing messages for a failed assessment request.
 *
 * Only the HTTP status and the backend `detail` message already
 * carried by `ApiError` are inspected. Nothing is logged, and no
 * stack trace, raw Axios error or internal identifier is ever
 * surfaced.
 */

import axios from "axios";

import { toApiError } from "@/lib/api/client";

const SESSION_EXPIRED = "Your session has expired. Please sign in again.";

const NOT_PERMITTED =
  "You do not have permission to perform this assessment.";

const NOT_FOUND = "The requested assessment resource was not found.";

const INVALID_IMAGES = "Please check the selected images and try again.";

const ANALYSIS_FAILED =
  "The assessment could not be completed. Please try again.";

const SERVER_UNREACHABLE =
  "Unable to reach the assessment server. Please check that the " +
  "backend is running and try again.";

const UNKNOWN_FAILURE =
  "Something went wrong while running the assessment.";

const STATUS_MESSAGES: Record<number, string> = {
  401: SESSION_EXPIRED,
  403: NOT_PERMITTED,
  404: NOT_FOUND,
  422: INVALID_IMAGES,
};

/**
 * Turn a failed analyze request into one concise sentence.
 *
 * A 400 or 413 carries a validation detail that the backend writes
 * for the operator ("Before image must be PNG or JPEG. ..."), so
 * that text is preferred over a generic message.
 *
 * Without an HTTP status the request never produced a response. An
 * Axios failure of that kind means the server could not be reached
 * or timed out; anything else is a genuine unknown and is reported
 * as such instead of blaming the backend.
 */
export function toAnalysisErrorMessage(error: unknown): string {
  const apiError = toApiError(error);

  if (apiError.status === null) {
    return axios.isAxiosError(error) ? SERVER_UNREACHABLE : UNKNOWN_FAILURE;
  }

  if (apiError.status === 400 || apiError.status === 413) {
    return apiError.message;
  }

  if (apiError.status >= 500) {
    return ANALYSIS_FAILED;
  }

  return STATUS_MESSAGES[apiError.status] ?? UNKNOWN_FAILURE;
}
