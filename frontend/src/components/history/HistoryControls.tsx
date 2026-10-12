"use client";

import { RotateCcw, Search } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { formatCount } from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  hasActiveQuery,
  SORT_OPTIONS,
  type FilterOption,
  type HistoryQuery,
  type HistorySortKey,
} from "./historyFilters";

/**
 * Search, filter and sort controls for the history page.
 *
 * Every option is derived from the records the backend actually
 * returned, so a filter can never offer a value that does not exist
 * in the current data. All controls are labelled and the result
 * count is announced politely as the user types.
 */

export interface HistoryControlsProps {
  query: HistoryQuery;
  damageLevelOptions: FilterOption[];
  statusOptions: FilterOption[];
  shownCount: number;
  totalCount: number;
  onChange: (patch: Partial<HistoryQuery>) => void;
  onReset: () => void;
}

const labelClass =
  "block text-xs font-medium uppercase tracking-wide text-slate-500";

const fieldClass =
  "mt-1.5 h-9 w-full rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-900 " +
  "transition-colors focus-visible:border-slate-500 focus-visible:outline-none " +
  "focus-visible:ring-2 focus-visible:ring-slate-500 focus-visible:ring-offset-1";

export function HistoryControls({
  query,
  damageLevelOptions,
  statusOptions,
  shownCount,
  totalCount,
  onChange,
  onReset,
}: HistoryControlsProps) {
  const isFiltered = hasActiveQuery(query);

  return (
    <Card>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <div className="sm:col-span-2 xl:col-span-1">
            <label htmlFor="history-search" className={labelClass}>
              Search
            </label>
            <div className="relative mt-1.5">
              <Search
                aria-hidden="true"
                className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
              />
              <input
                id="history-search"
                type="search"
                autoComplete="off"
                value={query.search}
                onChange={(event) => onChange({ search: event.target.value })}
                placeholder="ID, damage level, status or model"
                className={cn(fieldClass, "pl-9")}
              />
            </div>
          </div>

          <div>
            <label htmlFor="history-damage-level" className={labelClass}>
              Damage level
            </label>
            <select
              id="history-damage-level"
              value={query.damageLevel}
              onChange={(event) =>
                onChange({ damageLevel: event.target.value })
              }
              className={fieldClass}
            >
              <option value="">All levels</option>
              {damageLevelOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="history-status" className={labelClass}>
              Status
            </label>
            <select
              id="history-status"
              value={query.status}
              onChange={(event) => onChange({ status: event.target.value })}
              className={fieldClass}
            >
              <option value="">All statuses</option>
              {statusOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="history-sort" className={labelClass}>
              Sort by
            </label>
            <select
              id="history-sort"
              value={query.sort}
              onChange={(event) =>
                onChange({ sort: event.target.value as HistorySortKey })
              }
              className={fieldClass}
            >
              {SORT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 pt-4">
          <p className="text-sm text-slate-600" aria-live="polite">
            Showing{" "}
            <span className="font-medium text-slate-900">
              {formatCount(shownCount)}
            </span>{" "}
            of{" "}
            <span className="font-medium text-slate-900">
              {formatCount(totalCount)}
            </span>{" "}
            {totalCount === 1 ? "assessment" : "assessments"}
          </p>

          <Button
            variant="secondary"
            size="sm"
            onClick={onReset}
            disabled={!isFiltered}
          >
            <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
            Reset filters
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
