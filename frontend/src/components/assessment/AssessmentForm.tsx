"use client";

/**
 * Presentational form for a new damage assessment.
 *
 * This component owns no state. It receives the current selection,
 * the validation outcome and the derived form status from
 * `NewAssessmentView`, and it reports intent back through
 * callbacks. Keeping it stateless is what lets the upload and
 * analyze workflow be added later without restructuring the form.
 */

import { ArrowDown, CircleAlert, Info, Layers } from "lucide-react";

import { AssessmentValidationMessage } from "./AssessmentValidationMessage";
import { ImageUploadCard } from "./ImageUploadCard";
import { Button } from "@/components/ui/Button";
import type {
  ImageValidationIssue,
  ValidatedImage,
} from "@/lib/assessment/imageValidation";
import type { AssessmentImageSlot } from "./ImageUploadCard";
import { cn } from "@/lib/utils";

/**
 * Derived state of the form.
 *
 * - `empty`: no image selected at all.
 * - `incomplete`: exactly one of the two images is selected.
 * - `invalid`: a validation rule is violated.
 * - `ready`: both images are present and valid.
 * - `submitting`: the analysis request is in progress.
 */
export type AssessmentFormStatus =
  | "empty"
  | "incomplete"
  | "invalid"
  | "ready"
  | "submitting";

const STATUS_HINTS: Record<AssessmentFormStatus, string> = {
  empty: "Select a before image and an after image to enable the analysis.",
  incomplete: "Select the remaining image to enable the analysis.",
  invalid: "Resolve the validation messages above to enable the analysis.",
  ready: "Both images passed client-side validation.",
  submitting: "Running the assessment. This can take a moment.",
};

export type AssessmentValidationState = Record<
  AssessmentImageSlot,
  boolean
>;

/**
 * A single message shown under the image grid.
 *
 * `error` is announced assertively because it reports a failed
 * request; `info` is only ever polite. Both carry an icon and text,
 * so severity is never communicated by colour alone.
 */
export interface AssessmentMessage {
  text: string;
  severity: "info" | "error";
}

export interface AssessmentFormProps {
  before: ValidatedImage | null;
  after: ValidatedImage | null;
  beforePreviewUrl: string | null;
  afterPreviewUrl: string | null;
  beforeIssue: ImageValidationIssue | null;
  afterIssue: ImageValidationIssue | null;
  pairIssue: ImageValidationIssue | null;
  validating: AssessmentValidationState;
  status: AssessmentFormStatus;
  message: AssessmentMessage | null;
  onSelect: (slot: AssessmentImageSlot, file: File) => void;
  onRemove: (slot: AssessmentImageSlot) => void;
  onClear: () => void;
  onSubmit: () => void;
}

export function AssessmentForm({
  before,
  after,
  beforePreviewUrl,
  afterPreviewUrl,
  beforeIssue,
  afterIssue,
  pairIssue,
  validating,
  status,
  message,
  onSelect,
  onRemove,
  onClear,
  onSubmit,
}: AssessmentFormProps) {
  const isSubmitting = status === "submitting";
  const isValidating = validating.before || validating.after;
  const hasSelection = before !== null || after !== null;
  const canSubmit = status === "ready" && !isValidating;

  return (
    <form
      className="space-y-4"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <div className="grid gap-4 lg:grid-cols-2 lg:gap-6">
        <ImageUploadCard
          slot="before"
          image={before}
          previewUrl={beforePreviewUrl}
          issue={beforeIssue}
          isBusy={validating.before || isSubmitting}
          onSelect={onSelect}
          onRemove={onRemove}
        />

        <div
          aria-hidden="true"
          className="flex items-center justify-center lg:hidden"
        >
          <ArrowDown className="h-4 w-4 text-slate-400" />
        </div>

        <ImageUploadCard
          slot="after"
          image={after}
          previewUrl={afterPreviewUrl}
          issue={afterIssue}
          isBusy={validating.after || isSubmitting}
          onSelect={onSelect}
          onRemove={onRemove}
        />
      </div>

      {pairIssue && <AssessmentValidationMessage issue={pairIssue} />}

      {message && (
        <div
          role={message.severity === "error" ? "alert" : "status"}
          className={cn(
            "flex items-start gap-2 rounded-md border px-3 py-2",
            message.severity === "error"
              ? "border-amber-200 bg-amber-50"
              : "border-slate-200 bg-white",
          )}
        >
          {message.severity === "error" ? (
            <CircleAlert
              aria-hidden="true"
              className="mt-0.5 h-4 w-4 shrink-0 text-amber-700"
            />
          ) : (
            <Info
              aria-hidden="true"
              className="mt-0.5 h-4 w-4 shrink-0 text-slate-500"
            />
          )}

          <p
            className={cn(
              "text-xs leading-5",
              message.severity === "error" ? "text-amber-900" : "text-slate-700",
            )}
          >
            {message.text}
          </p>
        </div>
      )}

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-xs text-slate-500" aria-live="polite">
          {STATUS_HINTS[status]}
        </p>

        <div className="flex shrink-0 items-center gap-2">
          <Button
            variant="secondary"
            onClick={onClear}
            disabled={!hasSelection || isSubmitting}
          >
            Clear
          </Button>

          <Button
            type="submit"
            disabled={!canSubmit}
            aria-busy={isSubmitting}
          >
            <Layers aria-hidden="true" className="h-4 w-4" />
            Analyze Images
          </Button>
        </div>
      </div>
    </form>
  );
}