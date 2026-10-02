/**
 * User-facing messages for a failed assessment request.
 *
 * Only the HTTP status and the backend `detail` message already
 * carried by `ApiError` are inspected. Nothing is logged, and no
 * stack trace, raw Axios error or internal identifier is ever
 * surfaced.
 */

import type { ApiError } from "@/lib/api/client";

const SESSION_EXPIRED =
  "Your session has expired. Please sign in again.";

const NOT_PERMITTED =
  "You do not have permission to perform this assessment.";

const NOT_FOUND = "The requested assessment resource was not found.";

const INVALID_IMAGES =
  "Please check the selected images and try again.";

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
 * for the operator ("Before image must be PNG or JPEG. …"), so that
 * text is preferred over a generic message. A missing HTTP status
 * means the request never reached the backend, which is reported as
 * an unreachable server rather than as a rejection.
 */
export function toAnalysisErrorMessage(error: ApiError): string {
  if (error.status === null) {
    return SERVER_UNREACHABLE;
  }

  if (error.status === 400 || error.status === 413) {
    return error.message;
  }

  if (error.status >= 500) {
    return ANALYSIS_FAILED;
  }

  return STATUS_MESSAGES[error.status] ?? UNKNOWN_FAILURE;
}