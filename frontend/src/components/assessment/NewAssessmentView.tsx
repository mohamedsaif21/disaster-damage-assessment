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
 * cross-slot dimension rule, the derived form status and the single
 * message shown after an attempt. Previews are browser object URLs,
 * created only after a file has been validated and revoked again as
 * soon as the file is replaced, removed, cleared, or the view
 * unmounts.
 *
 * Submitting hands both validated files to the shared analyze
 * wrapper and, on success, navigates to the assessment the backend
 * just persisted. No `user_id` is ever read or sent, no Supabase
 * table is queried and no token is handled here; the shared API
 * client attaches authentication.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import {
  AssessmentForm,
  type AssessmentFormStatus,
  type AssessmentMessage,
  type AssessmentValidationState,
} from "./AssessmentForm";
import type { AssessmentImageSlot } from "./ImageUploadCard";
import { Card, CardContent } from "@/components/ui/Card";
import { analyzeAssessment } from "@/lib/api/assessment";
import { toAnalysisErrorMessage } from "@/lib/assessment/analysisErrors";
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

// ============================================================
// VIEW
// ============================================================

export function NewAssessmentView() {
  const router = useRouter();

  const [slots, setSlots] = useState<AssessmentSlots>(INITIAL_SLOTS);
  const [validating, setValidating] =
    useState<AssessmentValidationState>(NOT_VALIDATING);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState<AssessmentMessage | null>(null);

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
   * Whether a request is in flight right now.
   *
   * `isSubmitting` is state, so it only reaches the next render. The
   * ref changes synchronously and therefore closes the window in
   * which a second activation could still read a stale `false` and
   * create a second assessment for the same pair of images.
   */
  const requestInFlightRef = useRef(false);

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

      setMessage(null);
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
      setMessage(null);
      resetSlot(slot);
    },
    [resetSlot],
  );

  const handleClear = useCallback(() => {
    setMessage(null);
    resetSlot("before");
    resetSlot("after");
  }, [resetSlot]);

  const handleSubmit = useCallback(async () => {
    if (requestInFlightRef.current || !before || !after) {
      return;
    }

    if (slots.before.issue || slots.after.issue || pairIssue) {
      return;
    }

    // A validation may still be settling against the previous file
    // in a slot that already holds a valid image, so the pair is
    // never submitted while any slot is mid-check.
    if (validating.before || validating.after) {
      return;
    }

    requestInFlightRef.current = true;
    setIsSubmitting(true);
    setMessage(null);

    try {
      // The wrapper posts the existing before_image and after_image
      // multipart fields through the shared Axios client, which
      // attaches the Supabase access token. No user id is sent.
      const assessment = await analyzeAssessment(before.file, after.file);

      // Both images stay selected and their previews stay intact
      // while navigating, so an unsubscribed promise from a double
      // activation cannot land on a discarded component.
      router.push(`/assessment/${encodeURIComponent(assessment.id)}`);
    } catch (error) {
      setMessage({
        text: toAnalysisErrorMessage(error),
        severity: "error",
      });
      setIsSubmitting(false);
      requestInFlightRef.current = false;
    }
  }, [
    before,
    after,
    slots.before.issue,
    slots.after.issue,
    pairIssue,
    validating.before,
    validating.after,
    router,
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
            message={message}
            onSelect={handleSelect}
            onRemove={handleRemove}
            onClear={handleClear}
            onSubmit={() => {
              void handleSubmit();
            }}
          />
        </CardContent>
      </Card>
    </div>
  );
}