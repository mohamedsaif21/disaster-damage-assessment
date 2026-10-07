"use client";

/**
 * Indeterminate waiting state for a running analysis request.
 *
 * The backend exposes no progress endpoint, so this component
 * never renders a percentage, a progress bar, a countdown or a
 * timer. The spinner only shows that the request the user started
 * is still open; the real request duration decides when the view
 * moves on.
 *
 * The workflow stages below the message are a static explanation
 * of what an analysis involves. They are not reported by the
 * backend, so nothing is marked as completed, highlighted or
 * animated as if progress were being received. The whole block is
 * a single polite live region, so the waiting state is announced
 * by text rather than by animation alone.
 */

import { LoaderCircle } from "lucide-react";

interface AnalysisStage {
  title: string;
  detail: string;
}

const ANALYSIS_STAGES: AnalysisStage[] = [
  {
    title: "Images validated",
    detail: "Both images already passed the browser-side checks.",
  },
  {
    title: "Running AI damage assessment",
    detail: "The backend compares the before and after images.",
  },
  {
    title: "Preparing assessment results",
    detail: "The assessment is stored for the results view.",
  },
];

export function AnalysisLoading() {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-busy="true"
      className="flex flex-col items-center px-4 py-10 text-center sm:py-14"
    >
      <span
        aria-hidden="true"
        className="flex h-11 w-11 items-center justify-center rounded-full border border-slate-200 bg-slate-50"
      >
        <LoaderCircle className="h-5 w-5 animate-spin text-slate-700" />
      </span>

      <h2 className="mt-4 text-lg font-semibold tracking-tight text-slate-900">
        Analyzing Disaster Damage
      </h2>

      <p className="mt-2 max-w-md text-sm text-slate-600">
        Comparing your before and after images and running the damage
        assessment.
      </p>

      <p className="mt-1 max-w-md text-xs text-slate-500">
        The request is running now. This page updates as soon as the
        result is ready.
      </p>

      <div className="mt-6 w-full max-w-md rounded-md border border-slate-200 bg-slate-50 px-4 py-3 text-left">
        <p className="text-xs font-medium text-slate-700">
          What the analysis does
        </p>

        <ol className="mt-2 space-y-2">
          {ANALYSIS_STAGES.map((stage, index) => (
            <li key={stage.title} className="flex items-start gap-2.5">
              <span
                aria-hidden="true"
                className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-slate-300 bg-white text-[10px] font-semibold text-slate-600"
              >
                {index + 1}
              </span>

              <span className="min-w-0">
                <span className="block text-xs font-medium text-slate-800">
                  {stage.title}
                </span>
                <span className="block text-xs leading-5 text-slate-500">
                  {stage.detail}
                </span>
              </span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
