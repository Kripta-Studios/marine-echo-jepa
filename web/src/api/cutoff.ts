export const cutoffOptions = [6, 12, 18].map(
  (hour) => `2020-02-17T${String(hour).padStart(2, "0")}:00:00Z`,
);

export function safeCutoff(value: string | null): string {
  return value !== null && cutoffOptions.includes(value)
    ? value
    : cutoffOptions[1];
}
