"use client";

import { useCallback, useState } from "react";
import { FileDown, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { downloadAssessmentReport } from "@/lib/api/assessment";
import { toReportErrorMessage } from "@/lib/assessment/reportErrors";

/**
 * "Download PDF Report" action for a completed assessment.
 *
 * The report is delivered by the backend as a short-lived signed
 * URL in the assessment assets response, so this component never
 * builds a storage URL: it downloads the exact URL it is given
 * (`reportUrl`) through the shared Axios client and saves the bytes
 * as a file. Nothing is logged and no URL is persisted beyond the
 * generated object URL, which is revoked as soon as the download
 * is triggered.
 *
 * When `reportUrl` is null the assessment has no stored report.
 * The action is disabled rather than hidden so the availability is
 * explicit, and "Check again" re-runs the existing assets request
 * (via the parent) instead of fabricating a report.
 *
 * While a download is in flight the button is disabled, so a second
 * click cannot start a duplicate request.
 */

type ReportPhase = "idle" | "loading" | "error";

function reportFilename(assessmentId: string): string {
  const safe = assessmentId.replace(/[^a-zA-Z0-9_-]/g, "");
  return `damage-assessment-${safe || "report"}.pdf`;
}

export interface AssessmentReportActionProps {
  assessmentId: string;
  reportUrl: string | null;
  onRefresh: () => void;
}

export function AssessmentReportAction({
  assessmentId,
  reportUrl,
  onRefresh,
}: AssessmentReportActionProps) {
  const [phase, setPhase] = useState<ReportPhase>("idle");
  const [error, setError] = useState<string | null>(null);

  const handleDownload = useCallback(async () => {
    if (!reportUrl || phase === "loading") {
      return;
    }

    setError(null);
    setPhase("loading");

    try {
      const blob = await downloadAssessmentReport(reportUrl);

      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = reportFilename(assessmentId);
      anchor.rel = "noopener";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(objectUrl);

      setPhase("idle");
    } catch (downloadError) {
      setError(toReportErrorMessage(downloadError));
      setPhase("error");
    }
  }, [assessmentId, phase, reportUrl]);

  if (!reportUrl) {
    return (
      <div className="flex flex-col items-start gap-1">
        <Button variant="secondary" disabled>
          <FileDown aria-hidden="true" className="h-4 w-4" />
          Download PDF Report
        </Button>
        <span className="text-xs text-slate-500">
          No PDF report is available for this assessment.{" "}
          <button
            type="button"
            onClick={onRefresh}
            className="rounded font-medium text-slate-700 underline underline-offset-2 hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 focus-visible:ring-offset-2"
          >
            Check again
          </button>
        </span>
      </div>
    );
  }

  const loading = phase === "loading";

  return (
    <div className="flex flex-col items-start gap-1">
      <Button
        variant="secondary"
        onClick={handleDownload}
        disabled={loading}
        aria-busy={loading}
      >
        {loading ? (
          <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
        ) : (
          <FileDown aria-hidden="true" className="h-4 w-4" />
        )}
        {loading ? "Preparing PDF report\u2026" : "Download PDF Report"}
      </Button>

      <span aria-live="polite" className="sr-only">
        {loading ? "Preparing PDF report." : ""}
      </span>

      {phase === "error" && error ? (
        <span className="text-xs">
          <span role="alert" className="text-red-600">
            {error}
          </span>{" "}
          <button
            type="button"
            onClick={onRefresh}
            className="rounded font-medium text-slate-700 underline underline-offset-2 hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 focus-visible:ring-offset-2"
          >
            Try again
          </button>
        </span>
      ) : null}
    </div>
  );
}
