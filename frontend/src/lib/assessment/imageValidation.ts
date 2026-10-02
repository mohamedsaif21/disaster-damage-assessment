/**
 * Client-side validation rules for before/after assessment images.
 *
 * Every rule here mirrors a rule the backend already enforces in
 * `validate_image()` and the upload/analyze routes of
 * `backend/api/assessment.py`:
 *
 * - MAX_FILE_SIZE = 10 MB, checked against the bytes that are
 *   actually read, not the reported name.
 * - ALLOWED_CONTENT_TYPES = { image/png, image/jpeg }. A file whose
 *   reported MIME type is anything else is rejected.
 * - ALLOWED_FORMATS = { PNG, JPEG }. The backend detects the real
 *   format with Pillow, so this module reads the leading signature
 *   bytes instead of trusting the file name.
 * - The file must decode as a real image.
 * - MIN_IMAGE_WIDTH = MIN_IMAGE_HEIGHT = 256.
 * - Before and after must have exactly matching dimensions.
 *
 * No new or stricter rule is introduced here, and no backend code
 * is changed: this module only fails fast so a user is told about a
 * problem in the browser instead of after an upload.
 *
 * Validation is local. It reads the selected File and never sends
 * it anywhere.
 */

import type { ImageDimensions } from "@/lib/api/types";

/** Mirrors `MAX_FILE_SIZE` in backend/api/assessment.py. */
export const MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024;

/** Mirrors `ALLOWED_CONTENT_TYPES` in backend/api/assessment.py. */
export const ALLOWED_IMAGE_MIME_TYPES = ["image/png", "image/jpeg"] as const;

export type AllowedImageMimeType = (typeof ALLOWED_IMAGE_MIME_TYPES)[number];

/**
 * Narrow an arbitrary string to one of the accepted MIME types.
 *
 * A `File` reports whatever the operating system told the browser,
 * so the reported type is only ever checked, never trusted.
 */
export function isAllowedImageMimeType(
  value: string,
): value is AllowedImageMimeType {
  return (ALLOWED_IMAGE_MIME_TYPES as readonly string[]).includes(value);
}

/**
 * `accept` attribute for the file inputs.
 *
 * Only PNG and JPEG are offered, matching the MIME types the
 * backend accepts. The extensions are listed as well so the native
 * file picker starts in the right place; the actual check uses the
 * MIME type, never the extension.
 */
export const IMAGE_FILE_ACCEPT = "image/png,image/jpeg,.png,.jpg,.jpeg";

/** Mirrors `MIN_IMAGE_WIDTH` in backend/api/assessment.py. */
export const MIN_IMAGE_WIDTH = 256;

/** Mirrors `MIN_IMAGE_HEIGHT` in backend/api/assessment.py. */
export const MIN_IMAGE_HEIGHT = 256;

/** Detected image format, mirroring `ALLOWED_FORMATS`. */
export type DetectedImageFormat = "PNG" | "JPEG";

export interface ValidatedImage {
  file: File;
  format: DetectedImageFormat;
  width: number;
  height: number;
}

export type ImageValidationIssueCode =
  | "empty"
  | "invalid_type"
  | "too_large"
  | "unsupported_format"
  | "unreadable"
  | "too_small"
  | "dimension_mismatch";

/**
 * A user-facing validation result.
 *
 * `title` and `description` are written for the operator, so the UI
 * never has to render a backend message or an internal error.
 */
export interface ImageValidationIssue {
  code: ImageValidationIssueCode;
  title: string;
  description: string;
}

export type ImageValidationResult =
  | { ok: true; image: ValidatedImage }
  | { ok: false; issue: ImageValidationIssue };

// ============================================================
// SIGNATURE BYTES
//
// The backend determines the real format by decoding the image
// with Pillow. In the browser the equivalent cheap check is the
// file signature, which is what this module uses so a mislabelled
// file is rejected before it is ever uploaded.
// ============================================================

const PNG_SIGNATURE: readonly number[] = [
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
];

const JPEG_SIGNATURE: readonly number[] = [0xff, 0xd8, 0xff];

function startsWith(header: Uint8Array, signature: readonly number[]) {
  return (
    header.length >= signature.length &&
    signature.every((byte, index) => header[index] === byte)
  );
}

/**
 * Read the leading bytes of the file and report the format they
 * announce. Returns null when the file is neither PNG nor JPEG.
 */
async function sniffImageFormat(
  file: File,
): Promise<DetectedImageFormat | null> {
  const header = new Uint8Array(await file.slice(0, 12).arrayBuffer());

  if (startsWith(header, PNG_SIGNATURE)) {
    return "PNG";
  }

  if (startsWith(header, JPEG_SIGNATURE)) {
    return "JPEG";
  }

  return null;
}

// ============================================================
// DIMENSIONS
// ============================================================

/**
 * Decode the image locally and read its natural size.
 *
 * Resolves to null when the browser cannot decode the file. No
 * temporary object URL is left behind: it is revoked on both the
 * success and the failure path.
 */
function readImageDimensions(file: File): Promise<ImageDimensions | null> {
  if (typeof createImageBitmap === "function") {
    return createImageBitmap(file)
      .then((bitmap) => {
        const dimensions = {
          width: bitmap.width,
          height: bitmap.height,
        };

        bitmap.close();

        return dimensions;
      })
      .catch(() => null);
  }

  return new Promise<ImageDimensions | null>((resolve) => {
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();

    const settle = (dimensions: ImageDimensions | null) => {
      URL.revokeObjectURL(objectUrl);
      resolve(dimensions);
    };

    image.onload = () => {
      settle({
        width: image.naturalWidth,
        height: image.naturalHeight,
      });
    };

    image.onerror = () => {
      settle(null);
    };

    image.src = objectUrl;
  });
}

// ============================================================
// ISSUES
// ============================================================

function unsupportedFormatIssue(): ImageValidationIssue {
  return {
    code: "unsupported_format",
    title: "Unsupported image format",
    description:
      "The file is not a PNG or JPEG image. Please select a PNG or JPEG file.",
  };
}

function tooSmallIssue(dimensions: ImageDimensions): ImageValidationIssue {
  return {
    code: "too_small",
    title: "Image is too small",
    description:
      `Images must be at least ${MIN_IMAGE_WIDTH} × ${MIN_IMAGE_HEIGHT} pixels. ` +
      `The selected image is ${dimensions.width} × ${dimensions.height} pixels.`,
  };
}

// ============================================================
// FILE VALIDATION
// ============================================================

/**
 * Validate one selected image against the backend contract.
 *
 * The checks run in the same order as `validate_image()`: MIME
 * type, emptiness, size, real format, decodability and finally the
 * minimum dimensions. A rejected file is never stored, so a later
 * step cannot accidentally act on it.
 */
export async function validateImageFile(
  file: File,
): Promise<ImageValidationResult> {
  const mimeType = file.type;

  if (!isAllowedImageMimeType(mimeType)) {
    return {
      ok: false,
      issue: {
        code: "invalid_type",
        title: "Invalid file type",
        description:
          "Please select a supported image file (PNG or JPEG).",
      },
    };
  }

  if (file.size === 0) {
    return {
      ok: false,
      issue: {
        code: "empty",
        title: "File is empty",
        description:
          "This file has no content. Please select an image file.",
      },
    };
  }

  if (file.size > MAX_IMAGE_SIZE_BYTES) {
    return {
      ok: false,
      issue: {
        code: "too_large",
        title: "File is too large",
        description:
          "Please select an image within the supported size limit of 10 MB.",
      },
    };
  }

  const format = await sniffImageFormat(file);

  if (!format) {
    return { ok: false, issue: unsupportedFormatIssue() };
  }

  const dimensions = await readImageDimensions(file);

  if (!dimensions) {
    return {
      ok: false,
      issue: {
        code: "unreadable",
        title: "Not a valid image",
        description:
          "This file could not be read as an image. Please select a " +
          "different file.",
      },
    };
  }

  if (
    dimensions.width < MIN_IMAGE_WIDTH ||
    dimensions.height < MIN_IMAGE_HEIGHT
  ) {
    return { ok: false, issue: tooSmallIssue(dimensions) };
  }

  return {
    ok: true,
    image: {
      file,
      format,
      width: dimensions.width,
      height: dimensions.height,
    },
  };
}

// ============================================================
// PAIR VALIDATION
// ============================================================

/**
 * Check the before/after pair.
 *
 * Both upload endpoints reject a pair whose dimensions differ, so
 * the mismatch is reported here instead of after an upload. Returns
 * null when the pair is consistent.
 */
export function validateImageDimensions(
  before: ValidatedImage,
  after: ValidatedImage,
): ImageValidationIssue | null {
  if (before.width === after.width && before.height === after.height) {
    return null;
  }

  return {
    code: "dimension_mismatch",
    title: "Dimensions do not match",
    description:
      "Before and after images must have identical dimensions. " +
      `The before image is ${before.width} × ${before.height} pixels and the ` +
      `after image is ${after.width} × ${after.height} pixels.`,
  };
}