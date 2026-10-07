import { useState } from "react";
import type { ReactNode } from "react";
import Image from "next/image";
import { Camera, ImageOff, ScanLine, Siren } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import {
  formatBytes,
  formatPixelDimensions,
} from "@/lib/format";
import type {
  AssessmentAssetsResponse,
  AssessmentDetail,
  AssessmentImageRecord,
} from "@/lib/api/types";

/**
 * Image comparison grid: before image, after image and the AI
 * prediction mask, each shown with the real metadata recorded by
 * the backend.
 *
 * The three `src` values are the signed URLs from
 * `GET /assessment/{id}/assets`. They are used only as image
 * sources: no `storage_path`, `mask_storage_path` or raw URL is
 * rendered, and the links are never re-hosted or rewritten. The
 * images are rendered with `next/image` in `unoptimized` mode,
 * which the docs recommend for authenticated sources — the browser
 * fetches the signed URL directly, so no `next.config` hostname
 * allow-list is needed and no copy is cached by the optimizer.
 *
 * Each figure degrades independently:
 * - no signed URL (asset missing or assets request failed) →
 *   explicit unavailable message, exact wording for the mask;
 * - a URL that fails to load (for example expired) → error
 *   message via `onError`;
 * - metadata that is absent → the caption is simply omitted.
 */

interface Figure {
  key: string;
  label: string;
  icon: ReactNode;
  url: string | null;
  alt: string;
  caption: string | null;
  aspectRatio: string;
  missingMessage: string;
  errorMessage: string;
}

const DEFAULT_ASPECT_RATIO = "4 / 3";

function imageCaption(record: AssessmentImageRecord | null): string | null {
  if (!record) {
    return null;
  }

  const parts: string[] = [];

  if (record.width !== null && record.height !== null) {
    parts.push(formatPixelDimensions(record.width, record.height));
  }

  if (record.format) {
    parts.push(record.format);
  }

  if (record.size_bytes !== null) {
    parts.push(formatBytes(record.size_bytes));
  }

  return parts.length > 0 ? parts.join(" \u00b7 ") : null;
}

function imageAspectRatio(
  record: AssessmentImageRecord | null,
): string {
  if (record && record.width !== null && record.height !== null && record.height > 0) {
    return `${record.width} / ${record.height}`;
  }

  return DEFAULT_ASPECT_RATIO;
}

function UnavailableFigure({ message }: { message: string }) {
  return (
    <div className="flex aspect-[4/3] flex-col items-center justify-center gap-2 rounded-md border border-dashed border-slate-300 bg-slate-50 p-4 text-center">
      <ImageOff aria-hidden="true" className="h-5 w-5 text-slate-400" />
      <p className="text-sm font-medium text-slate-500">{message}</p>
    </div>
  );
}

function ResultFigure({ figure }: { figure: Figure }) {
  const [failed, setFailed] = useState(false);
  const showImage = figure.url !== null && !failed;

  return (
    <figure className="flex flex-col gap-2">
      <figcaption className="flex items-center gap-2">
        <span aria-hidden="true" className="text-slate-500">
          {figure.icon}
        </span>
        <h3 className="text-sm font-medium text-slate-700">{figure.label}</h3>
      </figcaption>

      {showImage ? (
        <div
          className="relative w-full overflow-hidden bg-slate-100"
          style={{ aspectRatio: figure.aspectRatio }}
        >
          <Image
            src={figure.url ?? ""}
            alt={figure.alt}
            fill
            unoptimized
            sizes="(max-width: 768px) 100vw, 33vw"
            className="object-contain"
            onError={() => setFailed(true)}
          />
        </div>
      ) : (
        <UnavailableFigure
          message={failed ? figure.errorMessage : figure.missingMessage}
        />
      )}

      {figure.caption ? (
        <p className="text-xs text-slate-500">{figure.caption}</p>
      ) : null}
    </figure>
  );
}

export interface AssessmentImagesProps {
  detail: AssessmentDetail;
  assets: AssessmentAssetsResponse | null;
}

export function AssessmentImages({ detail, assets }: AssessmentImagesProps) {
  const beforeRecord =
    detail.assessment_images.find((image) => image.image_type === "before") ??
    null;
  const afterRecord =
    detail.assessment_images.find((image) => image.image_type === "after") ??
    null;

  const figures: Figure[] = [
    {
      key: "before",
      label: "Before Disaster",
      icon: <Camera aria-hidden="true" className="h-4 w-4" />,
      url: assets?.before_image_url ?? null,
      alt: "Before disaster image of the assessed area",
      caption: imageCaption(beforeRecord),
      aspectRatio: imageAspectRatio(beforeRecord),
      missingMessage: "Before image unavailable.",
      errorMessage: "Before image could not be loaded.",
    },
    {
      key: "after",
      label: "After Disaster",
      icon: <Siren aria-hidden="true" className="h-4 w-4" />,
      url: assets?.after_image_url ?? null,
      alt: "After disaster image of the assessed area",
      caption: imageCaption(afterRecord),
      aspectRatio: imageAspectRatio(afterRecord),
      missingMessage: "After image unavailable.",
      errorMessage: "After image could not be loaded.",
    },
    {
      key: "prediction",
      label: "AI Prediction",
      icon: <ScanLine aria-hidden="true" className="h-4 w-4" />,
      url: assets?.prediction_mask_url ?? null,
      alt: "AI damage prediction mask for the assessed area",
      caption: detail.assessment_predictions
        ? `Mask ${formatPixelDimensions(
            detail.assessment_predictions.mask_width,
            detail.assessment_predictions.mask_height,
          )}`
        : null,
      aspectRatio: detail.assessment_predictions
        ? `${detail.assessment_predictions.mask_width} / ${detail.assessment_predictions.mask_height}`
        : DEFAULT_ASPECT_RATIO,
      missingMessage: "Prediction visualization unavailable.",
      errorMessage: "Prediction visualization could not be loaded.",
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Imagery</CardTitle>
        <p className="mt-1 text-sm text-slate-500">
          The stored image pair and the AI prediction mask, loaded through
          temporary signed links.
        </p>
      </CardHeader>

      <CardContent>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {figures.map((figure) => (
            <ResultFigure key={figure.key} figure={figure} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
