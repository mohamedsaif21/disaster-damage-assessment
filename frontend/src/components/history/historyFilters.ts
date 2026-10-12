import type { AssessmentHistoryItem } from "@/lib/api/types";
import { formatDamageLevel, formatStatus } from "@/lib/format";

/**
 * Pure search, filter and sort logic for the assessment history page.
 *
 * Everything here operates on the records the backend already
 * returned (`AssessmentHistoryItem`). The list is never mutated:
 * `filterHistory` copies before sorting, and the source array is
 * left untouched. No field is invented and no value is estimated.
 *
 * The backend already orders the history newest first, so the
 * default sort simply keeps that order rather than re-querying.
 */

export type HistorySortKey =
  | "newest"
  | "oldest"
  | "damage_desc"
  | "damage_asc";

export interface HistoryQuery {
  search: string;
  /** "" means "any level"; otherwise a raw `damage_level` value. */
  damageLevel: string;
  /** "" means "any status"; otherwise a raw `status` value. */
  status: string;
  sort: HistorySortKey;
}

/**
 * Sentinel used for records whose `damage_level` or `status` is
 * null. It can never collide with a real backend value because the
 * backend uses plain words such as `NONE` or `completed`.
 */
export const NO_VALUE = "__none__";

export const DEFAULT_HISTORY_QUERY: HistoryQuery = {
  search: "",
  damageLevel: "",
  status: "",
  sort: "newest",
};

export interface FilterOption {
  value: string;
  label: string;
}

export const SORT_OPTIONS: { value: HistorySortKey; label: string }[] = [
  { value: "newest", label: "Newest first" },
  { value: "oldest", label: "Oldest first" },
  { value: "damage_desc", label: "Highest damage first" },
  { value: "damage_asc", label: "Lowest damage first" },
];

/**
 * Presentation order for the known backend damage levels. Unknown
 * values are still listed, after the known ones, so nothing is
 * hidden if the backend ever emits a new level.
 */
const DAMAGE_LEVEL_ORDER = ["NONE", "LOW", "MODERATE", "HIGH", "SEVERE"];

function damageLevelRank(value: string): number {
  const index = DAMAGE_LEVEL_ORDER.indexOf(value);
  return index === -1 ? DAMAGE_LEVEL_ORDER.length : index;
}

/** Unique damage levels present in the returned records. */
export function buildDamageLevelOptions(
  assessments: AssessmentHistoryItem[],
): FilterOption[] {
  const values = new Set<string>();
  let hasUnrecorded = false;

  for (const item of assessments) {
    if (item.damage_level) {
      values.add(item.damage_level);
    } else {
      hasUnrecorded = true;
    }
  }

  const options = [...values]
    .sort((a, b) => {
      const rank = damageLevelRank(a) - damageLevelRank(b);
      return rank !== 0 ? rank : a.localeCompare(b);
    })
    .map((value) => ({ value, label: formatDamageLevel(value) }));

  if (hasUnrecorded) {
    options.push({ value: NO_VALUE, label: "Not recorded" });
  }

  return options;
}

/** Unique statuses present in the returned records. */
export function buildStatusOptions(
  assessments: AssessmentHistoryItem[],
): FilterOption[] {
  const values = new Set<string>();
  let hasUnrecorded = false;

  for (const item of assessments) {
    if (item.status) {
      values.add(item.status);
    } else {
      hasUnrecorded = true;
    }
  }

  const options = [...values]
    .sort((a, b) => a.localeCompare(b))
    .map((value) => ({ value, label: formatStatus(value) }));

  if (hasUnrecorded) {
    options.push({ value: NO_VALUE, label: "Not recorded" });
  }

  return options;
}

export function hasActiveQuery(query: HistoryQuery): boolean {
  return (
    query.search.trim() !== "" ||
    query.damageLevel !== "" ||
    query.status !== "" ||
    query.sort !== DEFAULT_HISTORY_QUERY.sort
  );
}

function matchesSearch(item: AssessmentHistoryItem, needle: string): boolean {
  const haystack = [
    item.id,
    item.damage_level ?? "",
    formatDamageLevel(item.damage_level),
    item.status ?? "",
    formatStatus(item.status),
    item.assessment_models?.name ?? "",
    item.assessment_models?.architecture ?? "",
  ]
    .join(" ")
    .toLowerCase();

  return haystack.includes(needle);
}

function timeValue(value: string): number {
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function compareByCreated(
  a: AssessmentHistoryItem,
  b: AssessmentHistoryItem,
  direction: 1 | -1,
): number {
  const diff = timeValue(a.created_at) - timeValue(b.created_at);

  if (diff !== 0) {
    return direction * diff;
  }

  // Deterministic tie-break so equal timestamps never reorder.
  return a.id.localeCompare(b.id);
}

function sortHistory(
  assessments: AssessmentHistoryItem[],
  key: HistorySortKey,
): AssessmentHistoryItem[] {
  const sorted = [...assessments];

  switch (key) {
    case "oldest":
      sorted.sort((a, b) => compareByCreated(a, b, 1));
      break;

    case "damage_desc":
      sorted.sort(
        (a, b) =>
          b.damage_percentage - a.damage_percentage ||
          compareByCreated(a, b, -1),
      );
      break;

    case "damage_asc":
      sorted.sort(
        (a, b) =>
          a.damage_percentage - b.damage_percentage ||
          compareByCreated(a, b, -1),
      );
      break;

    case "newest":
    default:
      sorted.sort((a, b) => compareByCreated(a, b, -1));
      break;
  }

  return sorted;
}

/**
 * Apply search, filters and sort. Returns a new array; the input is
 * never modified.
 */
export function filterHistory(
  assessments: AssessmentHistoryItem[],
  query: HistoryQuery,
): AssessmentHistoryItem[] {
  const needle = query.search.trim().toLowerCase();

  const filtered = assessments.filter((item) => {
    if (needle && !matchesSearch(item, needle)) {
      return false;
    }

    if (query.damageLevel !== "") {
      const value = item.damage_level ?? NO_VALUE;
      if (value !== query.damageLevel) {
        return false;
      }
    }

    if (query.status !== "") {
      const value = item.status ?? NO_VALUE;
      if (value !== query.status) {
        return false;
      }
    }

    return true;
  });

  return sortHistory(filtered, query.sort);
}
