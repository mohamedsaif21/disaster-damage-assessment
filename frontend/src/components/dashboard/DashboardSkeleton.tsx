import { Card, CardContent, CardHeader } from "@/components/ui/Card";

/**
 * Loading placeholder for the dashboard.
 *
 * Tailwind only: no skeleton library. The shapes mirror the real
 * summary and list layout so the page does not jump when data
 * arrives, and no empty state is shown while loading.
 */

function ShimmerBlock({ className }: { className: string }) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse rounded bg-slate-200 ${className}`}
    />
  );
}

export function DashboardSkeleton() {
  return (
    <div aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading your assessment history.</span>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, index) => (
          <Card key={index}>
            <CardContent className="flex items-start gap-3">
              <ShimmerBlock className="h-8 w-8 shrink-0 rounded-md" />
              <div className="w-full space-y-2">
                <ShimmerBlock className="h-3 w-24" />
                <ShimmerBlock className="h-7 w-20" />
                <ShimmerBlock className="h-3 w-full max-w-[9rem]" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="mt-6">
        <CardHeader className="flex flex-wrap items-center justify-between gap-3">
          <div className="space-y-2">
            <ShimmerBlock className="h-4 w-44" />
            <ShimmerBlock className="h-3 w-64 max-w-full" />
          </div>
          <ShimmerBlock className="h-8 w-20" />
        </CardHeader>

        <div className="divide-y divide-slate-200 border-t border-slate-200">
          {Array.from({ length: 5 }, (_, index) => (
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
