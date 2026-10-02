export type ClassValue =
  | string
  | number
  | null
  | undefined
  | false
  | ClassValue[]
  | Record<string, boolean | null | undefined>;

function toClassValue(value: ClassValue): string[] {
  if (!value) {
    return [];
  }

  if (typeof value === "string" || typeof value === "number") {
    return [String(value)];
  }

  if (Array.isArray(value)) {
    return value.flatMap(toClassValue);
  }

  return Object.entries(value)
    .filter(([, enabled]) => Boolean(enabled))
    .map(([key]) => key);
}

export function cn(...inputs: ClassValue[]): string {
  return inputs.flatMap(toClassValue).join(" ");
}
