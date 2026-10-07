/**
 * User-facing messages for a failed assessment detail request.
 *
 * Only the HTTP status carried by `ApiError` is inspected. The
 * backend `detail` payload, any stack trace and the raw Axios
 * error are never surfaced, so no internal identifier can leak
 * into the result page. Nothing is logged.
 *
 * The wording matches the messages agreed for the detail route:
 * 401 is an expired session, 403 an ownership refusal, 404 an
 * unknown id, and every other outcome (5xx, network failure, no
 * response at all) is reported as a load failure that can be
 * retried.
 */

import { toApiError } from "@/lib/api/client";

const SESSION_EXPIRED = "Your session has expired. Please sign in again.";

const NOT_PERMITTED =
  "You do not have permission to view this assessment.";

const NOT_FOUND = "Assessment not found.";

const LOAD_FAILED =
  "Unable to load this assessment. Please try again.";

export function toDetailErrorMessage(error: unknown): string {
  const apiError = toApiError(error);

  if (apiError.status === 401) {
    return SESSION_EXPIRED;
  }

  if (apiError.status === 403) {
    return NOT_PERMITTED;
  }

  if (apiError.status === 404) {
    return NOT_FOUND;
  }

  return LOAD_FAILED;
}
