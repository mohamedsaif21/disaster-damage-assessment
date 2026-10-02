import { ChevronRight } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/Button";
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
 * Recent assessments, newest first.
 *
 * The ordering is the one the backend already applies
 * (`created_at` descending); the list is not re-sorted or
 * re-labelled client side. Only fields present on
 * `AssessmentHistoryItem` are shown.
 */

export interface RecentAssessmentsProps {
  assessments: AssessmentHistoryItem[];
  total: number;
  onRefresh: () => void;
}

export function RecentAssessments({
  assessments,
  total,
  onRefresh,
}: RecentAssessmentsProps) {
  return (
    <section aria-label="Recent assessments">
      <Card>
        <CardHeader className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle>Recent Assessments</CardTitle>
            <p className="mt-1 text-sm text-slate-500">
              Newest first, as returned by the assessment history.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-sm text-slate-500">
              {assessments.length} of {total}
            </span>
            <Button variant="secondary" size="sm" onClick={onRefresh}>
              Refresh
            </Button>
          </div>
        </CardHeader>

        <CardContent className="px-0 pb-0">
          <ul className="divide-y divide-slate-200 border-t border-slate-200">
            {assessments.map((item) => (
              <li key={item.id}>
                <AssessmentRow item={item} />
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </section>
  );
}

interface AssessmentRowProps {
  item: AssessmentHistoryItem;
}

function AssessmentRow({ item }: AssessmentRowProps) {
  const modelName = item.assessment_models?.name ?? null;

  return (
    <Link
      href={`/assessment/${item.id}`}
      className="group flex items-start justify-between gap-4 px-6 py-4 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-slate-500"
    >
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="font-mono text-sm font-medium text-slate-900">
            {shortId(item.id)}
          </span>
          <span
            className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs font-medium text-slate-700"
            title={item.damage_level ?? "No damage level recorded"}
          >
            {formatDamageLevel(item.damage_level)}
          </span>
          <span className="text-sm font-medium text-slate-900">
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
            <dt className="text-slate-500">Status</dt>
            <dd>{formatStatus(item.status)}</dd>
          </div>

          <div className="flex gap-1.5">
            <dt className="text-slate-500">Model</dt>
            <dd className="truncate">{orDash(modelName)}</dd>
          </div>
        </dl>
      </div>

      <ChevronRight
        aria-hidden="true"
        className="mt-1 h-4 w-4 shrink-0 text-slate-400 transition-colors group-hover:text-slate-600"
      />
    </Link>
  );
}
