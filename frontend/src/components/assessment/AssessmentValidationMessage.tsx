import { CircleAlert, TriangleAlert } from "lucide-react";

import type { ImageValidationIssue } from "@/lib/assessment/imageValidation";

/**
 * Renders one validation issue for an assessment image.
 *
 * The state is never communicated by colour alone: every issue shows
 * a distinct icon, a short title and an explanatory sentence. The
 * message is written for the operator, so no backend `detail`
 * payload, HTTP status or internal error is ever exposed.
 *
 * `role="alert"` announces the message when it appears, because it
 * is added after the user has already chosen a file.
 */

export interface AssessmentValidationMessageProps {
  issue: ImageValidationIssue;
  id?: string;
}

export function AssessmentValidationMessage({
  issue,
  id,
}: AssessmentValidationMessageProps) {
  const Icon =
    issue.code === "dimension_mismatch" ? TriangleAlert : CircleAlert;

  return (
    <div
      id={id}
      role="alert"
      className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-left"
    >
      <Icon
        aria-hidden="true"
        className="mt-0.5 h-4 w-4 shrink-0 text-amber-700"
      />

      <p className="text-xs leading-5 text-amber-900">
        <span className="font-semibold">{issue.title}.</span>{" "}
        {issue.description}
      </p>
    </div>
  );
}