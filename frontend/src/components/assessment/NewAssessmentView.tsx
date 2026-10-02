"use client";

/**
 * New Assessment view.
 *
 * This is the client boundary for the form: files are selected in
 * the browser, so the whole interaction lives client side while the
 * route itself stays a thin server component.
 *
 * The view owns everything stateful: the two selected images, their
 * local preview URLs, the validation outcome for each slot, the
 * cross-slot dimension rule and the derived form status. Nothing is
 * uploaded. Previews are browser object URLs, created only after a
 * file has been validated and revoked again as soon as the file is
 * replaced, removed, cleared, or the view unmounts.
 *
 * No `user_id` is ever read or sent, no Supabase table is queried
 * and no token is handled here; the shared API client attaches
 * authentication when a request is eventually dispatched.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  AssessmentForm,
  type AssessmentFormStatus,
  type AssessmentValidationState,
} from "./AssessmentForm";
import type { AssessmentImageSlot } from "./ImageUploadCard";
import { Card, CardContent } from "@/components/ui/Card";
import {
  validateImageDimensions,
  validateImageFile,
  type ImageValidationIssue,
  type ValidatedImage,
} from "@/lib/assessment/imageValidation";

// ============================================================
// STATE MODEL
// ============================================================

interface AssessmentSlotState {
  image: ValidatedImage | null;
  previewUrl: string | null;
  issue: ImageValidationIssue | null;
}

type AssessmentSlots = Record<AssessmentImageSlot, AssessmentSlotState>;

const EMPTY_SLOT: AssessmentSlotState = {
  image: null,
  previewUrl: null,
  issue: null,
};

const INITIAL_SLOTS: AssessmentSlots = {
  before: EMPTY_SLOT,
  after: EMPTY_SLOT,
};

const NOT_VALIDATING: AssessmentValidationState = {
  before: false,
  after: false,
};

/**
 * Shown after the action is activated. The analysis is not
 * dispatched yet, so the form reports readiness instead of
 * pretending a result exists.
 */
const ANALYSIS_NOT_DISPATCHED_NOTICE =
  "Both images are ready. Running the damage assessment is not enabled " +
  "in this build yet, so nothing has been submitted.";

// ============================================================
// VIEW
// ============================================================

export function NewAssessmentView() {
  const [slots, setSlots] = useState<AssessmentSlots>(INITIAL_SLOTS);
  const [validating, setValidating] =
    useState<AssessmentValidationState>(NOT_VALIDATING);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  /**
   * Per-slot validation run counter.
   *
   * Every selection bumps the counter for its own slot. A result is
   * applied only while it still matches, so a slow validation can
   * never overwrite a newer choice. Unmounting bumps both counters,
   * which stops an in-flight validation from writing state.
   */
  const runIdsRef = useRef<Record<AssessmentImageSlot, number>>({
    before: 0,
    after: 0,
  });

  /**
   * Every object URL this view has created and not yet revoked.
   *
   * The set is captured on mount, so unmount cleanup can release
   * the last live preview without depending on state that may
   * already be gone.
   */
  const livePreviewUrlsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const livePreviewUrls = livePreviewUrlsRef.current;

    return () => {
      livePreviewUrls.forEach((url) => {
        URL.revokeObjectURL(url);
      });

      livePreviewUrls.clear();
    };
  }, []);

  const before = slots.before.image;
  const after = slots.after.image;

  /**
   * Both upload endpoints require identical dimensions, so the pair
   * is checked as soon as both images are known.
   */
  const pairIssue = useMemo(
    () => (before && after ? validateImageDimensions(before, after) : null),
    [before, after],
  );

  const status: AssessmentFormStatus = useMemo(() => {
    if (isSubmitting) {
      return "submitting";
    }

    if (slots.before.issue || slots.after.issue || pairIssue) {
      return "invalid";
    }

    if (before && after) {
      return "ready";
    }

    if (before || after) {
      return "incomplete";
    }

    return "empty";
  }, [
    isSubmitting,
    slots.before.issue,
    slots.after.issue,
    pairIssue,
    before,
    after,
  ]);

  const releasePreviewUrl = useCallback((url: string | null) => {
    if (!url) {
      return;
    }

    URL.revokeObjectURL(url);
    livePreviewUrlsRef.current.delete(url);
  }, []);

  const handleSelect = useCallback(
    (slot: AssessmentImageSlot, file: File) => {
      runIdsRef.current[slot] += 1;
      const runId = runIdsRef.current[slot];

      setNotice(null);
      setValidating((current) => ({ ...current, [slot]: true }));

      void validateImageFile(file).then((result) => {
        if (runIdsRef.current[slot] !== runId) {
          return;
        }

        setValidating((current) => ({ ...current, [slot]: false }));

        // The previous preview is released before the new one is
        // created, so replacing an image never leaves an orphaned
        // object URL behind.
        releasePreviewUrl(slots[slot].previewUrl);

        // A rejected file never enters the state, so nothing
        // downstream can act on it. The slot is left empty with the
        // reason attached, which keeps a single issue describing
        // exactly the current contents of that slot.
        if (!result.ok) {
          setSlots((current) => ({
            ...current,
            [slot]: { image: null, previewUrl: null, issue: result.issue },
          }));

          return;
        }

        const previewUrl = URL.createObjectURL(file);

        livePreviewUrlsRef.current.add(previewUrl);

        setSlots((current) => ({
          ...current,
          [slot]: { image: result.image, previewUrl, issue: null },
        }));
      });
    },
    [releasePreviewUrl, slots],
  );

  const resetSlot = useCallback(
    (slot: AssessmentImageSlot) => {
      runIdsRef.current[slot] += 1;

      setValidating((current) => ({ ...current, [slot]: false }));

      releasePreviewUrl(slots[slot].previewUrl);

      setSlots((current) => ({ ...current, [slot]: EMPTY_SLOT }));
    },
    [releasePreviewUrl, slots],
  );

  const handleRemove = useCallback(
    (slot: AssessmentImageSlot) => {
      setNotice(null);
      resetSlot(slot);
    },
    [resetSlot],
  );

  const handleClear = useCallback(() => {
    setNotice(null);
    resetSlot("before");
    resetSlot("after");
  }, [resetSlot]);

  const handleSubmit = useCallback(() => {
    if (isSubmitting || !before || !after) {
      return;
    }

    if (slots.before.issue || slots.after.issue || pairIssue) {
      return;
    }

    setIsSubmitting(true);
    setNotice(null);

    // Dispatch point for the assessment workflow: the next step
    // awaits analyzeAssessment(before.file, after.file) from
    // @/lib/api/assessment here. That function posts the existing
    // before_image and after_image multipart fields through the
    // shared Axios client, which attaches the Supabase access token
    // and never sends a user id.
    void Promise.resolve()
      .then(() => {
        setNotice(ANALYSIS_NOT_DISPATCHED_NOTICE);
      })
      .finally(() => {
        setIsSubmitting(false);
      });
  }, [
    isSubmitting,
    before,
    after,
    slots.before.issue,
    slots.after.issue,
    pairIssue,
  ]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight text-slate-900">
          New Assessment
        </h2>
        <p className="mt-1 text-sm text-slate-600">
          Select a before and an after image of the same location. Both
          images are checked in your browser and are only sent when the
          analysis is started.
        </p>
      </div>

      <Card>
        <CardContent className="py-4 sm:py-6">
          <AssessmentForm
            before={before}
            after={after}
            beforePreviewUrl={slots.before.previewUrl}
            afterPreviewUrl={slots.after.previewUrl}
            beforeIssue={slots.before.issue}
            afterIssue={slots.after.issue}
            pairIssue={pairIssue}
            validating={validating}
            status={status}
            notice={notice}
            onSelect={handleSelect}
            onRemove={handleRemove}
            onClear={handleClear}
            onSubmit={handleSubmit}
          />
        </CardContent>
      </Card>
    </div>
  );
}