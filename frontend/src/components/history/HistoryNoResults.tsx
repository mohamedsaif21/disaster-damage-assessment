import { RotateCcw, SearchX } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";

/**
 * Shown when the history has records but none match the current
 * search or filters. Distinct from the "no assessments yet" empty
 * state, which the dashboard empty state covers.
 */

export interface HistoryNoResultsProps {
  onClear: () => void;
}

export function HistoryNoResults({ onClear }: HistoryNoResultsProps) {
  return (
    <Card>
      <CardContent className="flex flex-col items-center px-6 py-14 text-center">
        <span
          aria-hidden="true"
          className="flex h-10 w-10 items-center justify-center rounded-md bg-slate-100 text-slate-500"
        >
          <SearchX className="h-5 w-5" />
        </span>

        <h3 className="mt-4 text-base font-semibold text-slate-900">
          No matching assessments
        </h3>

        <p className="mt-2 max-w-md text-sm text-slate-600">
          No assessments match the current search and filters. Adjust the
          criteria, or clear them to see every assessment.
        </p>

        <Button className="mt-6" onClick={onClear}>
          <RotateCcw aria-hidden="true" className="h-4 w-4" />
          Clear filters
        </Button>
      </CardContent>
    </Card>
  );
}
