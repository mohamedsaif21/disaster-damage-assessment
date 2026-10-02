"use client";

/**
 * Dashboard view.
 *
 * This is the client boundary for the dashboard: the Supabase
 * session lives in the browser, so the history request can only be
 * issued client side. The page component stays a thin server
 * component that renders this view inside the existing AppShell.
 *
 * All API communication happens in `useAssessmentHistory`; this
 * component only decides what to render.
 */

import { DashboardEmptyState } from "./DashboardEmptyState";
import { DashboardErrorState } from "./DashboardErrorState";
import { DashboardSkeleton } from "./DashboardSkeleton";
import { DashboardSummary } from "./DashboardSummary";
import { RecentAssessments } from "./RecentAssessments";
import { useAssessmentHistory } from "@/lib/hooks/useAssessmentHistory";

/**
 * How many assessments the recent list shows. The backend already
 * returns newest first, so the list is a slice, not a re-sort.
 */
const RECENT_LIMIT = 5;

export function DashboardView() {
  const { data, isLoading, error, refetch } = useAssessmentHistory();

  const assessments = data?.assessments ?? [];
  const total = data?.count ?? 0;
  const isEmpty = !isLoading && !error && assessments.length === 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-slate-900">
          Dashboard
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          An overview of your damage assessments, newest first.
        </p>
      </div>

      {isLoading && <DashboardSkeleton />}

      {!isLoading && error && (
        <DashboardErrorState error={error} onRetry={refetch} />
      )}

      {!isLoading && !error && isEmpty && <DashboardEmptyState />}

      {!isLoading && !error && !isEmpty && (
        <>
          <DashboardSummary assessments={assessments} total={total} />

          <RecentAssessments
            assessments={assessments.slice(0, RECENT_LIMIT)}
            total={total}
            onRefresh={refetch}
          />
        </>
      )}
    </div>
  );
}
