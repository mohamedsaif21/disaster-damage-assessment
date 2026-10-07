import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import {
  formatCount,
  formatDateTime,
  formatDamageLevel,
  formatPercentage,
  formatStatus,
} from "@/lib/format";
import type { AssessmentDetail } from "@/lib/api/types";

/**
 * Primary result of one assessment.
 *
 * Every value comes straight from `AssessmentDetail`; nothing is
 * recalculated in the browser. `damage_level` and
 * `damage_percentage` are the backend's authoritative result and
 * are shown as the most prominent figures on the page. Nullable
 * fields fall back to the shared em dash, so no "null" or
 * "undefined" can ever render.
 */

export interface AssessmentSummaryProps {
  detail: AssessmentDetail;
}

export function AssessmentSummary({ detail }: AssessmentSummaryProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Damage Summary</CardTitle>
        <p className="mt-1 text-sm text-slate-500">
          The assessment result as stored by the backend.
        </p>
      </CardHeader>

      <CardContent>
        <div className="flex flex-wrap items-end gap-x-10 gap-y-4">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Damage level
            </p>
            <p
              className="mt-1 text-3xl font-semibold tracking-tight text-slate-900"
              title={detail.damage_level ?? "No damage level recorded"}
            >
              {formatDamageLevel(detail.damage_level)}
            </p>
          </div>

          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Damage percentage
            </p>
            <p className="mt-1 text-3xl font-semibold tracking-tight text-slate-900">
              {formatPercentage(detail.damage_percentage)}
            </p>
          </div>
        </div>

        <dl className="mt-6 grid grid-cols-1 gap-x-6 gap-y-4 border-t border-slate-200 pt-4 sm:grid-cols-2 lg:grid-cols-3">
          <div>
            <dt className="text-xs font-medium text-slate-500">Status</dt>
            <dd className="mt-0.5 text-sm font-medium text-slate-900">
              {formatStatus(detail.status)}
            </dd>
          </div>

          <div>
            <dt className="text-xs font-medium text-slate-500">
              Total pixels
            </dt>
            <dd className="mt-0.5 text-sm font-medium text-slate-900">
              {detail.total_pixels !== null
                ? formatCount(detail.total_pixels)
                : "\u2014"}
            </dd>
          </div>

          <div>
            <dt className="text-xs font-medium text-slate-500">
              Damage pixels
            </dt>
            <dd className="mt-0.5 text-sm font-medium text-slate-900">
              {detail.damage_pixels !== null
                ? formatCount(detail.damage_pixels)
                : "\u2014"}
            </dd>
          </div>

          <div>
            <dt className="text-xs font-medium text-slate-500">Created</dt>
            <dd className="mt-0.5 text-sm font-medium text-slate-900">
              <time dateTime={detail.created_at}>
                {formatDateTime(detail.created_at)}
              </time>
            </dd>
          </div>

          <div>
            <dt className="text-xs font-medium text-slate-500">Updated</dt>
            <dd className="mt-0.5 text-sm font-medium text-slate-900">
              <time dateTime={detail.updated_at}>
                {formatDateTime(detail.updated_at)}
              </time>
            </dd>
          </div>

          <div className="sm:col-span-2 lg:col-span-3">
            <dt className="text-xs font-medium text-slate-500">
              Assessment ID
            </dt>
            <dd className="mt-0.5 break-all font-mono text-sm text-slate-900">
              {detail.id}
            </dd>
          </div>
        </dl>
      </CardContent>
    </Card>
  );
}
