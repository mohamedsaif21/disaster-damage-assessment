"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { RefreshCw, TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { getAssessment, getAssessmentAssets } from "@/lib/api/assessment";
import type {
  AssessmentAssetsResponse,
  AssessmentDetail,
} from "@/lib/api/types";
import { toDetailErrorMessage } from "@/lib/assessment/detailErrors";
import { AssessmentImages } from "./AssessmentImages";
import { AssessmentResultError } from "./AssessmentResultError";
import { AssessmentResultSkeleton } from "./AssessmentResultSkeleton";
import { AssessmentSummary } from "./AssessmentSummary";
import { DamageStatistics } from "./DamageStatistics";
import { ModelInformation } from "./ModelInformation";

/**
 * Client container for the assessment result page.
 *
 * Data flow, in order:
 * 1. `GET /assessment/{id}` (detail) is mandatory. A failure maps
 *    to the agreed message (401 / 403 / 404 / network) and shows
 *    the error card; Retry repeats the request.
 * 2. Only after the detail succeeds does it request
 *    `GET /assessment/{id}/assets`. The signed URLs are handed to
 *    `AssessmentImages` and never rendered as text or logged.
 *
 * If the assets request fails, the assessment result is still
 * shown: a notice explains that images are unavailable and offers
 * a retry, while each figure falls back to its own unavailable
 * state. Retry re-runs the whole sequence.
 *
 * Both requests go through the shared Axios client, so the
 * Supabase bearer token is attached centrally; no token, header,
 * `user_id` or direct Supabase query appears here. The detail id
 * is passed through unchanged (it is treated as an opaque string
 * by the backend). Nothing is logged, and no state survives a
 * component unmount: the `active` flag discards responses from a
 * superseded effect, and `requestId` gives Retry a fresh key.
 */

type Phase = "loading" | "error" | "ready";

interface ResultState {
  phase: Phase;
  detail: AssessmentDetail | null;
  assets: AssessmentAssetsResponse | null;
  assetsError: string | null;
  message: string | null;
}

const INITIAL_STATE: ResultState = {
  phase: "loading",
  detail: null,
  assets: null,
  assetsError: null,
  message: null,
};

export interface AssessmentResultViewProps {
  assessmentId: string;
}

export function AssessmentResultView({
  assessmentId,
}: AssessmentResultViewProps) {
  const [requestId, setRequestId] = useState(0);
  const [state, setState] = useState<ResultState>(INITIAL_STATE);

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const detail = await getAssessment(assessmentId);

        if (!active) {
          return;
        }

        let assets: AssessmentAssetsResponse | null = null;
        let assetsError: string | null = null;

        try {
          assets = await getAssessmentAssets(assessmentId);
        } catch (error) {
          assetsError = toDetailErrorMessage(error);
        }

        if (!active) {
          return;
        }

        setState({
          phase: "ready",
          detail,
          assets,
          assetsError,
          message: null,
        });
      } catch (error) {
        if (!active) {
          return;
        }

        setState({
          phase: "error",
          detail: null,
          assets: null,
          assetsError: null,
          message: toDetailErrorMessage(error),
        });
      }
    }

    void load();

    return () => {
      active = false;
    };
  }, [assessmentId, requestId]);

  const retry = useCallback(() => {
    setState(INITIAL_STATE);
    setRequestId((current) => current + 1);
  }, []);

  if (state.phase === "loading") {
    return <AssessmentResultSkeleton />;
  }

  if (state.phase === "error" || state.detail === null) {
    return (
      <AssessmentResultError
        message={
          state.message ?? "Unable to load this assessment. Please try again."
        }
        onRetry={retry}
      />
    );
  }

  const { detail, assets, assetsError } = state;

  return (
    <div>
      <header className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-slate-900">
            Assessment Result
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            The stored assessment result, with imagery, damage statistics and
            model details.
          </p>
        </div>

        <Link
          href="/dashboard"
          className="inline-flex h-9 items-center justify-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 focus-visible:ring-offset-2"
        >
          Back to Dashboard
        </Link>
      </header>

      <div className="space-y-6">
        <AssessmentSummary detail={detail} />

        {assetsError ? (
          <div
            role="alert"
            className="flex flex-wrap items-center gap-x-4 gap-y-3 rounded-md border border-amber-200 bg-amber-50 px-4 py-3"
          >
            <div className="flex min-w-0 items-start gap-2">
              <TriangleAlert
                aria-hidden="true"
                className="mt-0.5 h-4 w-4 shrink-0 text-amber-700"
              />
              <div className="min-w-0 text-sm">
                <p className="font-semibold text-amber-900">
                  Images could not be loaded.
                </p>
                <p className="mt-0.5 text-amber-800">{assetsError}</p>
              </div>
            </div>

            <Button
              variant="secondary"
              size="sm"
              onClick={retry}
              className="ml-auto"
            >
              <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />
              Retry
            </Button>
          </div>
        ) : null}

        <AssessmentImages detail={detail} assets={assets} />

        <DamageStatistics statistics={detail.assessment_class_statistics} />

        <ModelInformation model={detail.assessment_models} />
      </div>
    </div>
  );
}
