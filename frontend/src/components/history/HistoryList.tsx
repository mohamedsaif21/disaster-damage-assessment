import { ChevronRight } from "lucide-react";
import Link from "next/link";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import {
  formatDamageLevel,
  formatDateTime,
  formatPercentage,
  formatStatus,
  orDash,
  shortId,
} from "@/lib/format";
import type { AssessmentHistoryItem } from "@/lib/api/types";

/**
 * The filtered and sorted assessment records.
 *
 * Only fields present on `AssessmentHistoryItem` are rendered. The
 * id is shown shortened with the full value in `title`, and every
 * row links to the existing Step 8.10 result page. A card list is
 * used instead of a wide table so the layout stays readable on
 * mobile without horizontal scrolling.
 *
 * `user_id` is deliberately never displayed, matching the dashboard.
 */

export interface HistoryListProps {
  assessments: AssessmentHistoryItem[];
}

export function HistoryList({ assessments }: HistoryListProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Assessments</CardTitle>
        <p className="mt-1 text-sm text-slate-500">
          Select a record to open its full result.
        </p>
      </CardHeader>

      <CardContent className="px-0 pb-0">
        <ul className="divide-y divide-slate-200 border-t border-slate-200">
          {assessments.map((item) => (
            <li key={item.id}>
              <HistoryRow item={item} />
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

interface HistoryRowProps {
  item: AssessmentHistoryItem;
}

function HistoryRow({ item }: HistoryRowProps) {
  const model = item.assessment_models;

  return (
    <Link
      href={`/assessment/${item.id}`}
      className="group flex flex-col gap-3 px-5 py-4 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-slate-500 sm:flex-row sm:items-center sm:justify-between sm:gap-6 sm:px-6"
    >
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <span
            className="font-mono text-sm font-medium text-slate-900"
            title={item.id}
          >
            {shortId(item.id)}
          </span>

          <span
            className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs font-medium text-slate-700"
            title={item.damage_level ?? "No damage level recorded"}
          >
            {formatDamageLevel(item.damage_level)}
          </span>

          <span className="text-sm font-semibold tabular-nums text-slate-900">
            {formatPercentage(item.damage_percentage)}
          </span>
        </div>

        <dl className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-sm text-slate-600">
          <div className="flex gap-1.5">
            <dt className="text-slate-500">Created</dt>
            <dd>
              <time dateTime={item.created_at}>
                {formatDateTime(item.created_at)}
              </time>
            </dd>
          </div>

          <div className="flex gap-1.5">
            <dt className="text-slate-500">Updated</dt>
            <dd>
              <time dateTime={item.updated_at}>
                {formatDateTime(item.updated_at)}
              </time>
            </dd>
          </div>

          <div className="flex gap-1.5">
            <dt className="text-slate-500">Status</dt>
            <dd>{formatStatus(item.status)}</dd>
          </div>

          <div className="flex min-w-0 gap-1.5">
            <dt className="text-slate-500">Model</dt>
            <dd className="truncate">{orDash(model?.name)}</dd>
          </div>

          {model?.architecture ? (
            <div className="flex min-w-0 gap-1.5">
              <dt className="text-slate-500">Architecture</dt>
              <dd className="truncate">{model.architecture}</dd>
            </div>
          ) : null}
        </dl>
      </div>

      <span className="inline-flex shrink-0 items-center gap-1 text-sm font-medium text-slate-500 transition-colors group-hover:text-slate-700">
        Open
        <ChevronRight aria-hidden="true" className="h-4 w-4" />
      </span>
    </Link>
  );
}
