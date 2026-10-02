"use client";

/**
 * One labelled image slot of the New Assessment form.
 *
 * The card is responsible for a single image only. It never talks
 * to the API: the selected File is handed back to the parent view,
 * which owns validation and the form state. The preview is a local
 * object URL, so choosing a file performs no network request.
 *
 * The card is deliberately explicit about whether it holds the
 * pre-disaster or the post-disaster image, and the two slots use
 * different wording, icons and badges so they cannot be confused.
 */

import { useId, useRef, useState, type DragEvent } from "react";
import { Camera, ImageUp, Siren, Trash2, type LucideIcon } from "lucide-react";

import { AssessmentValidationMessage } from "./AssessmentValidationMessage";
import { ImagePreview } from "./ImagePreview";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import {
  IMAGE_FILE_ACCEPT,
  MAX_IMAGE_SIZE_BYTES,
  MIN_IMAGE_HEIGHT,
  MIN_IMAGE_WIDTH,
  type ImageValidationIssue,
  type ValidatedImage,
} from "@/lib/assessment/imageValidation";
import { formatBytes, formatPixelDimensions } from "@/lib/format";
import { cn } from "@/lib/utils";

export type AssessmentImageSlot = "before" | "after";

interface SlotCopy {
  title: string;
  badge: string;
  description: string;
  icon: LucideIcon;
  selectLabel: string;
  replaceLabel: string;
  removeLabel: string;
  previewAlt: (filename: string) => string;
  emptyTitle: string;
  emptyHint: string;
}

const SLOT_COPY: Record<AssessmentImageSlot, SlotCopy> = {
  before: {
    title: "Before Disaster Image",
    badge: "Pre-disaster",
    description:
      "The reference image taken before the disaster. This is the " +
      "undamaged baseline the after image is compared against.",
    icon: Camera,
    selectLabel: "Select before image",
    replaceLabel: "Replace before image",
    removeLabel: "Remove before image",
    previewAlt: (filename) => `Preview of the before disaster image ${filename}`,
    emptyTitle: "No before image selected",
    emptyHint: "Drag a before image here, or choose a file from your device.",
  },
  after: {
    title: "After Disaster Image",
    badge: "Post-disaster",
    description:
      "The image taken after the disaster. This is the scene the " +
      "damage is measured in.",
    icon: Siren,
    selectLabel: "Select after image",
    replaceLabel: "Replace after image",
    removeLabel: "Remove after image",
    previewAlt: (filename) => `Preview of the after disaster image ${filename}`,
    emptyTitle: "No after image selected",
    emptyHint: "Drag an after image here, or choose a file from your device.",
  },
};

const MAX_IMAGE_SIZE_MEGABYTES = Math.round(
  MAX_IMAGE_SIZE_BYTES / (1024 * 1024),
);

export interface ImageUploadCardProps {
  slot: AssessmentImageSlot;
  image: ValidatedImage | null;
  previewUrl: string | null;
  issue: ImageValidationIssue | null;
  isBusy: boolean;
  onSelect: (slot: AssessmentImageSlot, file: File) => void;
  onRemove: (slot: AssessmentImageSlot) => void;
}

export function ImageUploadCard({
  slot,
  image,
  previewUrl,
  issue,
  isBusy,
  onSelect,
  onRemove,
}: ImageUploadCardProps) {
  const copy = SLOT_COPY[slot];
  const Icon = copy.icon;

  const inputId = useId();
  const hintId = useId();

  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepthRef = useRef(0);

  const [isDragging, setIsDragging] = useState(false);

  const openPicker = () => {
    inputRef.current?.click();
  };

  const handleInputChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.item(0);

    // Clearing the value lets the same file be chosen again after
    // it has been removed, because the browser fires no change
    // event when the value did not actually change.
    event.target.value = "";

    if (file) {
      onSelect(slot, file);
    }
  };

  const handleDragEnter = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();

    dragDepthRef.current += 1;

    if (!isBusy) {
      setIsDragging(true);
    }
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    // Required on the parent for a drop event to fire at all.
    event.preventDefault();
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();

    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);

    if (dragDepthRef.current === 0) {
      setIsDragging(false);
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();

    dragDepthRef.current = 0;
    setIsDragging(false);

    if (isBusy) {
      return;
    }

    const file = event.dataTransfer.files.item(0);

    if (file) {
      onSelect(slot, file);
    }
  };

  return (
    <Card className="flex h-full flex-col">
      <CardHeader>
        <div className="flex items-start gap-3">
          <span
            aria-hidden="true"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-slate-200 bg-slate-50 text-slate-600"
          >
            <Icon className="h-4 w-4" />
          </span>

          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <CardTitle>{copy.title}</CardTitle>
              <span className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs font-medium text-slate-700">
                {copy.badge}
              </span>
            </div>

            <p className="mt-1 text-sm text-slate-500">{copy.description}</p>
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col gap-3">
        <div
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            "flex flex-1 flex-col justify-center rounded-md border-2 border-dashed transition-colors",
            isDragging
              ? "border-slate-900 bg-slate-100"
              : "border-slate-300 bg-slate-50",
            isBusy && "opacity-60",
          )}
        >
          <input
            ref={inputRef}
            id={inputId}
            type="file"
            accept={IMAGE_FILE_ACCEPT}
            className="hidden"
            disabled={isBusy}
            onChange={handleInputChange}
          />

          {image && previewUrl ? (
            <ImagePreview
              previewUrl={previewUrl}
              alt={copy.previewAlt(image.file.name)}
              className="rounded-md"
            />
          ) : (
            <div className="flex flex-col items-center px-4 py-8 text-center">
              <span
                aria-hidden="true"
                className="flex h-9 w-9 items-center justify-center rounded-md bg-white text-slate-500 ring-1 ring-slate-200"
              >
                <ImageUp className="h-4 w-4" />
              </span>

              <p className="mt-3 text-sm font-medium text-slate-800">
                {isBusy ? "Checking image\u2026" : copy.emptyTitle}
              </p>

              <p className="mt-1 max-w-xs text-xs text-slate-500">
                {copy.emptyHint}
              </p>
            </div>
          )}
        </div>

        <p id={hintId} className="text-xs text-slate-500">
          PNG or JPEG, up to {MAX_IMAGE_SIZE_MEGABYTES} MB, at least{" "}
          {MIN_IMAGE_WIDTH} \u00d7 {MIN_IMAGE_HEIGHT} pixels. The before and
          after image must have identical dimensions.
        </p>

        {issue && (
          <AssessmentValidationMessage issue={issue} />
        )}

        {image && (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 rounded-md border border-slate-200 bg-white px-3 py-2 text-xs sm:grid-cols-3">
            <div className="col-span-2 min-w-0 sm:col-span-3">
              <dt className="text-slate-500">File</dt>
              <dd className="truncate font-medium text-slate-900" title={image.file.name}>
                {image.file.name}
              </dd>
            </div>

            <div>
              <dt className="text-slate-500">Size</dt>
              <dd className="font-medium text-slate-900">
                {formatBytes(image.file.size)}
              </dd>
            </div>

            <div>
              <dt className="text-slate-500">Dimensions</dt>
              <dd className="font-medium text-slate-900">
                {formatPixelDimensions(image.width, image.height)}
              </dd>
            </div>

            <div>
              <dt className="text-slate-500">Format</dt>
              <dd className="font-medium text-slate-900">{image.format}</dd>
            </div>
          </dl>
        )}

        <div className="mt-auto flex flex-wrap items-center gap-2">
          <Button
            variant={image ? "secondary" : "primary"}
            size="sm"
            disabled={isBusy}
            aria-describedby={hintId}
            onClick={openPicker}
          >
            <ImageUp aria-hidden="true" className="h-4 w-4" />
            {image ? copy.replaceLabel : copy.selectLabel}
          </Button>

          {image && (
            <Button
              variant="ghost"
              size="sm"
              disabled={isBusy}
              onClick={() => onRemove(slot)}
            >
              <Trash2 aria-hidden="true" className="h-4 w-4" />
              {copy.removeLabel}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}