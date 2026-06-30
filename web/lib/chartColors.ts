// Centralized chart palette. Every chart/series color comes from here so a
// theme or design-system change is a token edit, not a hunt across components.
// Values are CSS custom properties defined in app/globals.css (:root), so they
// also flow through any future light/dark theming.

/** Ordered series colors for multi-line / multi-series charts. */
export const SERIES = [
  "var(--series-1)", // green (accent)
  "var(--series-2)", // blue
  "var(--series-3)", // amber
  "var(--series-4)", // red
  "var(--series-5)", // purple
  "var(--series-6)", // teal
] as const;

/** The brand accent, for primary single-series charts. */
export const ACCENT = "var(--accent)";

/** Three-stop heatmap ramp (low → mid → high) as [r, g, b] triples. Kept as
 *  numeric channels because the ramp is interpolated numerically in code. */
export const HEATMAP_STOPS: [number, number, number][] = [
  [13, 27, 42],
  [27, 94, 110],
  [0, 214, 143],
];
