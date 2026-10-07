import { Card, CardContent, CardHeader } from "@/components/ui/Card";

/**
 * Loading placeholder for the assessment result page.
 *
 * Tailwind only, in the same restrained style as
 * `DashboardSkeleton`: the shapes mirror the real result layout
 * so the page does not jump when the data arrives. This state
 * means "the completed result is being retrieved" and is
 * deliberately distinct from the Step 8.9 analysis spinner, which
 * means "a request is running". No progress value is shown.
 */

function ShimmerBlock({ className }: { className: string }) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse rounded bg-slate-200 ${className}`}
    />
  );
}

export function AssessmentResultSkeleton() {
  return (
    <div aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading the assessment result.</span>

      <Card>
        <CardHeader>
          <ShimmerBlock className="h-4 w-32" />
        </CardHeader>
        <CardContent className="grid gap-6 sm:grid-cols-2">
          <div className="space-y-3">
            <ShimmerBlock className="h-4 w-24" />
            <ShimmerBlock className="h-8 w-40" />
            <ShimmerBlock className="h-8 w-32" />
          </div>
          <div className="space-y-3">
            <ShimmerBlock className="h-4 w-full" />
            <ShimmerBlock className="h-4 w-full" />
            <ShimmerBlock className="h-4 w-2/3" />
          </div>
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardHeader>
          <ShimmerBlock className="h-4 w-24" />
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {Array.from({ length: 3 }, (_, index) => (
              <div key={index} className="space-y-2">
                <ShimmerBlock className="h-4 w-28" />
                <ShimmerBlock className="aspect-[4/3] w-full rounded-md" />
                <ShimmerBlock className="h-3 w-40" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <ShimmerBlock className="h-4 w-36" />
          </CardHeader>
          <CardContent className="space-y-3">
            {Array.from({ length: 5 }, (_, index) => (
              <ShimmerBlock key={index} className="h-4 w-full" />
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <ShimmerBlock className="h-4 w-36" />
          </CardHeader>
          <CardContent className="space-y-3">
            {Array.from({ length: 4 }, (_, index) => (
              <ShimmerBlock key={index} className="h-4 w-full" />
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
