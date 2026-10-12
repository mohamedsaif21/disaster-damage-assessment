import { Card, CardContent, CardHeader } from "@/components/ui/Card";

/**
 * Loading placeholder for the history page.
 *
 * Tailwind only, in the same restrained style as
 * `DashboardSkeleton`: the shapes mirror the controls and list so
 * the page does not jump when the data arrives. The controls are
 * not rendered with placeholder values, so nothing looks
 * interactive before the real records exist.
 */

function ShimmerBlock({ className }: { className: string }) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse rounded bg-slate-200 ${className}`}
    />
  );
}

export function HistorySkeleton() {
  return (
    <div aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading your assessment history.</span>

      <Card>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <div className="space-y-2 sm:col-span-2 xl:col-span-1">
              <ShimmerBlock className="h-3 w-16" />
              <ShimmerBlock className="h-9 w-full" />
            </div>
            <div className="space-y-2">
              <ShimmerBlock className="h-3 w-24" />
              <ShimmerBlock className="h-9 w-full" />
            </div>
            <div className="space-y-2">
              <ShimmerBlock className="h-3 w-16" />
              <ShimmerBlock className="h-9 w-full" />
            </div>
            <div className="space-y-2">
              <ShimmerBlock className="h-3 w-20" />
              <ShimmerBlock className="h-9 w-full" />
            </div>
          </div>

          <div className="flex items-center justify-between border-t border-slate-200 pt-4">
            <ShimmerBlock className="h-4 w-44" />
            <ShimmerBlock className="h-8 w-28" />
          </div>
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardHeader>
          <ShimmerBlock className="h-4 w-32" />
          <ShimmerBlock className="mt-2 h-3 w-56 max-w-full" />
        </CardHeader>

        <div className="divide-y divide-slate-200 border-t border-slate-200">
          {Array.from({ length: 6 }, (_, index) => (
            <div key={index} className="space-y-2 px-6 py-4">
              <div className="flex items-center gap-3">
                <ShimmerBlock className="h-4 w-20" />
                <ShimmerBlock className="h-5 w-16 rounded-full" />
                <ShimmerBlock className="h-4 w-12" />
              </div>
              <div className="flex flex-wrap gap-4">
                <ShimmerBlock className="h-3 w-40" />
                <ShimmerBlock className="h-3 w-24" />
                <ShimmerBlock className="h-3 w-32" />
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
