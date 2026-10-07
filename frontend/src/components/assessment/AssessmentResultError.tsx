import { CircleAlert, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";

/**
 * Error state for a failed assessment detail request.
 *
 * Only the mapped, human-readable message is rendered: the HTTP
 * status, the backend `detail` payload and the raw Axios error
 * are never shown, so no token, storage path or internal trace
 * can leak into the UI. Nothing is logged.
 *
 * The block is an `alert`, so the failure is announced when it
 * appears, and the retry repeats the actual API request instead
 * of reloading the page.
 */

export interface AssessmentResultErrorProps {
  message: string;
  onRetry: () => void;
}

export function AssessmentResultError({
  message,
  onRetry,
}: AssessmentResultErrorProps) {
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
          Assessment unavailable
        </h3>

        <p role="alert" className="mt-2 max-w-md text-sm text-slate-600">
          {message}
        </p>

        <Button className="mt-6" onClick={onRetry}>
          <RefreshCw aria-hidden="true" className="h-4 w-4" />
          Retry
        </Button>
      </CardContent>
    </Card>
  );
}
