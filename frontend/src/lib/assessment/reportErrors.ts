/**
 * User-facing messages for a failed PDF report download.
 *
 * Only the HTTP status carried by `ApiError` is inspected. The
 * raw Axios error, the backend/Storage response body and the
 * signed URL are never surfaced or logged, so no token, storage
 * path or internal detail can leak into the result page.
 *
 * The statuses come from the signed Storage URL rather than the
 * FastAPI report route: an expired or tampered signature is a
 * 400 or 403, a removed object is a 404, and anything else (5xx,
 * network failure, no response) is reported as a retryable
 * download failure.
 */

import { toApiError } from "@/lib/api/client";

const SESSION_EXPIRED = "Your session has expired. Please sign in again.";

const REPORT_LINK_EXPIRED =
  "The report link has expired. Please try again to prepare a new one.";

const REPORT_UNAVAILABLE = "The PDF report is no longer available.";

const DOWNLOAD_FAILED =
  "Unable to download the PDF report. Please try again.";

export function toReportErrorMessage(error: unknown): string {
  const apiError = toApiError(error);

  if (apiError.status === 401) {
    return SESSION_EXPIRED;
  }

  if (apiError.status === 400 || apiError.status === 403) {
    return REPORT_LINK_EXPIRED;
  }

  if (apiError.status === 404) {
    return REPORT_UNAVAILABLE;
  }

  return DOWNLOAD_FAILED;
}
