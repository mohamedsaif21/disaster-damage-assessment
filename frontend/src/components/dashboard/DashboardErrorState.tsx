import { CircleAlert, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import type { ApiError } from "@/lib/api/client";

/**
 * Error state for a failed history request.
 *
 * Only a safe, human-readable message is rendered. The raw Axios
 * error, the backend `detail` payload and the HTTP status are
 * never shown, so no token, key or internal trace can leak into
 * the UI. Nothing is logged either.
 */

export interface DashboardErrorStateProps {
  error: ApiError;
  onRetry: () => void;
}

export function DashboardErrorState({
  error,
  onRetry,
}: DashboardErrorStateProps) {
  const isUnauthenticated = error.status === 401;

  const description = isUnauthenticated
    ? "Your session is missing or has expired, so your assessment history " +
      "could not be retrieved. Sign in again and retry."
    : "We couldn't retrieve your assessment history. Check that the " +
      "assessment service is reachable, then try again.";

  return (
    <Card>
      <CardContent className="flex flex-col items-center px-6 py-14 text-center">
        <span
          aria-hidden="true"
          className="flex h-10 w-10 items-center justify-center rounded-md border border-amber-200 bg-amber-50 text-amber-700"
        >
          <CircleAlert className="h-5 w-5" />
        </span>

        <h3 className="mt-4 text-base font-semibold text-slate-900">
          Unable to load assessments
        </h3>

        <p className="mt-2 max-w-md text-sm text-slate-600">{description}</p>

        <Button className="mt-6" onClick={onRetry}>
          <RefreshCw aria-hidden="true" className="h-4 w-4" />
          Try Again
        </Button>
      </CardContent>
    </Card>
  );
}
