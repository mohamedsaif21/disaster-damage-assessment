/**
 * Small display formatting helpers for assessment data.
 *
 * These are presentation concerns only. Stored values are never
 * modified, and no value is invented: every helper either renders
 * a field that the backend returned or falls back to an em dash.
 *
 * Date and number formatting uses the built-in `Intl` API, so no
 * date library is required.
 */

const EM_DASH = "\u2014";

const dateTimeFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

const integerFormatter = new Intl.NumberFormat();

/**
 * Format a backend timestamp for display.
 *
 * The value is rendered in the reader's local timezone, which is
 * the correct semantic for an operational tool. The stored
 * timestamp itself is untouched. An absent or unparseable value
 * becomes an em dash rather than "Invalid Date".
 */
export function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return EM_DASH;
  }

  const parsed = new Date(value);

  if (Number.isNaN(parsed.getTime())) {
    return EM_DASH;
  }

  return dateTimeFormatter.format(parsed);
}

/**
 * Format a count with locale-aware digit grouping.
 */
export function formatCount(value: number): string {
  return integerFormatter.format(value);
}

/**
 * Format the backend `damage_percentage` as a percentage.
 *
 * The backend already rounds this to four decimals, so two are
 * shown and trailing zeros are kept to preserve precision
 * consistently across rows.
 */
export function formatPercentage(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return EM_DASH;
  }

  return `${value.toFixed(2)}%`;
}

/**
 * Present a backend damage level as readable text.
 *
 * The backend emits NONE, LOW, MODERATE, HIGH or SEVERE. Any other
 * value is passed through unchanged rather than mapped, so an
 * unexpected level is still visible to the operator.
 */
export function formatDamageLevel(
  value: string | null | undefined,
): string {
  if (!value) {
    return EM_DASH;
  }

  const words = value.replace(/_/g, " ").trim().toLowerCase();

  if (words.length === 0) {
    return EM_DASH;
  }

  return words.charAt(0).toUpperCase() + words.slice(1);
}

/**
 * Present an assessment status as readable text.
 */
export function formatStatus(value: string | null | undefined): string {
  return formatDamageLevel(value);
}

const byteUnits = ["B", "KB", "MB", "GB"] as const;

/**
 * Format a byte count as a readable file size.
 *
 * Used for local file metadata in the browser, so the value is the
 * size of the selected File rather than a backend field. An invalid
 * size becomes an em dash instead of "NaN B".
 */
export function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return EM_DASH;
  }

  let size = value;
  let unit = 0;

  while (size >= 1024 && unit < byteUnits.length - 1) {
    size /= 1024;
    unit += 1;
  }

  const digits = unit === 0 || size >= 10 ? 0 : 1;

  return `${size.toFixed(digits)} ${byteUnits[unit]}`;
}

/**
 * Present pixel dimensions for display, for example
 * "1,920 × 1,080".
 */
export function formatPixelDimensions(
  width: number,
  height: number,
): string {
  return `${integerFormatter.format(width)} \u00d7 ${integerFormatter.format(height)}`;
}

/**
 * Shorten an assessment UUID for a compact row label. The full id
 * stays available to assistive technology through the title text.
 */
export function shortId(value: string | null | undefined): string {
  if (!value) {
    return EM_DASH;
  }

  return value.length > 8 ? value.slice(0, 8) : value;
}

/**
 * A neutral placeholder for any field the backend returned as
 * null, so the UI never renders an empty cell.
 */
export function orDash(value: string | null | undefined): string {
  if (!value) {
    return EM_DASH;
  }

  return value;
}
