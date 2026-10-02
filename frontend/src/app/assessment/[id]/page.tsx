import Link from "next/link";

import { Card, CardContent } from "@/components/ui/Card";

/**
 * Assessment detail placeholder.
 *
 * The dashboard links here so an assessment id has a real,
 * structurally correct destination. This page deliberately
 * performs no request: the detail view, its assets and its
 * statistics are not part of this step.
 */
export default async function AssessmentDetailPage(
  props: PageProps<"/assessment/[id]">,
) {
  const { id } = await props.params;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-slate-900">
          Assessment Detail
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          The assessment detail view is not implemented yet.
        </p>
      </div>

      <Card>
        <CardContent className="space-y-4">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Assessment ID
            </p>
            <p className="mt-1 break-all font-mono text-sm text-slate-900">
              {id}
            </p>
          </div>

          <p className="text-sm text-slate-600">
            Images, statistics and the generated report for this assessment
            will be shown here in a later step.
          </p>

          <Link
            href="/dashboard"
            className="inline-flex h-9 items-center justify-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 focus-visible:ring-offset-2"
          >
            Back to Dashboard
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}
