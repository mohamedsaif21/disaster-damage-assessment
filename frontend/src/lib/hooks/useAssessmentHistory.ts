"use client";

/**
 * Loads the authenticated user's assessment history.
 *
 * This hook owns request state only. It calls the existing typed
 * API function, so the request flows through the shared Axios
 * client, which attaches the Supabase access token centrally. No
 * `user_id` is sent: the backend derives ownership from the
 * verified token.
 *
 * No state-management or data-fetching library is used; plain
 * React state is enough for a single endpoint.
 */

import { useCallback, useEffect, useState } from "react";

import { getAssessmentHistory } from "@/lib/api/assessment";
import type { AssessmentHistoryResponse } from "@/lib/api/types";
import { toApiError, type ApiError } from "@/lib/api/client";

export interface UseAssessmentHistoryResult {
  data: AssessmentHistoryResponse | null;
  isLoading: boolean;
  error: ApiError | null;
  refetch: () => void;
}

interface HistoryState {
  data: AssessmentHistoryResponse | null;
  error: ApiError | null;
  /**
   * The request this state belongs to. A request counts as
   * settled only once its own response has been recorded, so the
   * loading flag is derived rather than set imperatively.
   */
  settledRequestId: number;
}

const INITIAL_STATE: HistoryState = {
  data: null,
  error: null,
  settledRequestId: -1,
};

export function useAssessmentHistory(): UseAssessmentHistoryResult {
  const [state, setState] = useState<HistoryState>(INITIAL_STATE);
  const [requestId, setRequestId] = useState(0);

  const refetch = useCallback(() => {
    setRequestId((current) => current + 1);
  }, []);

  useEffect(() => {
    // Cleanup marks the run as stale, so a slow response from a
    // superseded request can never overwrite fresher state.
    let active = true;

    // `limit` is intentionally omitted. The backend sets
    // `count` to the number of rows it returned, so a limited
    // request would report a partial count and the dashboard
    // could not show a truthful total.
    getAssessmentHistory()
      .then((response) => {
        if (!active) {
          return;
        }

        setState({
          data: response,
          error: null,
          settledRequestId: requestId,
        });
      })
      .catch((thrown: unknown) => {
        if (!active) {
          return;
        }

        // Normalized so the view can rely on `ApiError` even when
        // a failure happened outside the response interceptor.
        setState({
          data: null,
          error: toApiError(thrown),
          settledRequestId: requestId,
        });
      });

    return () => {
      active = false;
    };
  }, [requestId]);

  return {
    data: state.data,
    isLoading: state.settledRequestId !== requestId,
    error: state.error,
    refetch,
  };
}
