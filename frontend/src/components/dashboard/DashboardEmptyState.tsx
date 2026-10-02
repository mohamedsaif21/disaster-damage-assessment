import { FilePlus } from "lucide-react";
import Link from "next/link";

import { Card, CardContent } from "@/components/ui/Card";

/**
 * Shown when the authenticated user has no assessments yet.
 *
 * The action navigates to the existing New Assessment route. No
 * assessment form is implemented at this stage.
 */
export function DashboardEmptyState() {
  return (
    <Card>
      <CardContent className="flex flex-col items-center px-6 py-14 text-center">
        <span
          aria-hidden="true"
          className="flex h-10 w-10 items-center justify-center rounded-md bg-slate-100 text-slate-500"
        >
          <FilePlus className="h-5 w-5" />
        </span>

        <h3 className="mt-4 text-base font-semibold text-slate-900">
          No assessments yet
        </h3>

        <p className="mt-2 max-w-md text-sm text-slate-600">
          You have not run any damage assessments. Start one by uploading a
          before and after image pair to see results appear here.
        </p>

        <Link
          href="/assessment/new"
          className="mt-6 inline-flex h-9 items-center justify-center gap-2 rounded-md bg-slate-900 px-4 text-sm font-medium text-white transition-colors hover:bg-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 focus-visible:ring-offset-2"
        >
          <FilePlus aria-hidden="true" className="h-4 w-4" />
          New Assessment
        </Link>
      </CardContent>
    </Card>
  );
}
