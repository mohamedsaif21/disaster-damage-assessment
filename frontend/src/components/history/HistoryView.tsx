"use client";

import { useCallback, useMemo, useState } from "react";

import { DashboardEmptyState } from "@/components/dashboard/DashboardEmptyState";
import { DashboardErrorState } from "@/components/dashboard/DashboardErrorState";
import { useAssessmentHistory } from "@/lib/hooks/useAssessmentHistory";
import { HistoryControls } from "./HistoryControls";
import { HistoryList } from "./HistoryList";
import { HistoryNoResults } from "./HistoryNoResults";
import { HistorySkeleton } from "./HistorySkeleton";
import {
  buildDamageLevelOptions,
  buildStatusOptions,
  DEFAULT_HISTORY_QUERY,
  filterHistory,
  type HistoryQuery,
} from "./historyFilters";

/**
 * Client container for the assessment history page.
 *
 * It reuses `useAssessmentHistory`, which owns the single history
 * request and flows through the shared Axios client, so the
 * Supabase bearer token is attached centrally and no `user_id` is
 * ever sent from the frontend. This component only derives the
 * filter options, applies client-side search/filter/sort and
 * decides which state to render.
 *
 * The raw records are never modified: `filterHistory` returns a new
 * array for display.
 */
export function HistoryView() {
  const { data, isLoading, error, refetch } = useAssessmentHistory();
  const [query, setQuery] = useState<HistoryQuery>(DEFAULT_HISTORY_QUERY);

  const assessments = useMemo(() => data?.assessments ?? [], [data]);
  const total = data?.count ?? 0;

  const isEmpty = !isLoading && !error && assessments.length === 0;

  const damageLevelOptions = useMemo(
    () => buildDamageLevelOptions(assessments),
    [assessments],
  );

  const statusOptions = useMemo(
    () => buildStatusOptions(assessments),
    [assessments],
  );

  const visible = useMemo(
    () => filterHistory(assessments, query),
    [assessments, query],
  );

  const updateQuery = useCallback((patch: Partial<HistoryQuery>) => {
    setQuery((current) => ({ ...current, ...patch }));
  }, []);

  const resetQuery = useCallback(() => {
    setQuery(DEFAULT_HISTORY_QUERY);
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-slate-900">
          Assessment History
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          Every damage assessment saved to your account, newest first. Search,
          filter and sort the records, then open one to see its full result.
        </p>
      </div>

      {isLoading && <HistorySkeleton />}

      {!isLoading && error && (
        <DashboardErrorState error={error} onRetry={refetch} />
      )}

      {!isLoading && !error && isEmpty && <DashboardEmptyState />}

      {!isLoading && !error && !isEmpty && (
        <>
          <HistoryControls
            query={query}
            damageLevelOptions={damageLevelOptions}
            statusOptions={statusOptions}
            shownCount={visible.length}
            totalCount={total}
            onChange={updateQuery}
            onReset={resetQuery}
          />

          {visible.length === 0 ? (
            <HistoryNoResults onClear={resetQuery} />
          ) : (
            <HistoryList assessments={visible} />
          )}
        </>
      )}
    </div>
  );
}
