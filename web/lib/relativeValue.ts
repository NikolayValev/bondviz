// Relative value: where today's spread sits within its own history. Pure statistics over the
// cached SpreadPoint series (1990→today), so the Signal page can say "2s10s is +1.3σ, 88th
// percentile of the last 10 years" instead of just quoting a level.

import type { SpreadPoint } from "@/lib/signal";

export interface RelativeValue {
  current: number;
  mean: number;
  /** Population standard deviation of the window. */
  std: number;
  /** (current − mean) / std; 0 when the window has no dispersion. */
  z: number;
  /** Share of window observations at or below current, 0–100. */
  percentile: number;
  min: number;
  max: number;
  n: number;
}

/** Stats of `current` (default: the last finite value) against the window `values`. */
export function relativeValue(values: number[], at?: number): RelativeValue | null {
  const xs = values.filter((v) => Number.isFinite(v));
  const current = at ?? xs[xs.length - 1];
  if (xs.length < 2 || current === undefined || !Number.isFinite(current)) return null;
  const n = xs.length;
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  let atOrBelow = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const v of xs) {
    if (v <= current) atOrBelow++;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  // Rounding leaves a flat series with std ~1e-17, which would blow z up; call that zero.
  const flat = std <= 1e-12 * Math.max(1, Math.abs(mean));
  return { current, mean, std: flat ? 0 : std, z: flat ? 0 : (current - mean) / std, percentile: (100 * atOrBelow) / n, min, max, n };
}

/**
 * Relative value of a spread's latest observation over a trailing window (calendar years back
 * from the latest date), or the whole history when `years` is null.
 */
export function spreadRelativeValue(
  points: SpreadPoint[],
  key: "s10y3m" | "s2s10s",
  years: number | null,
): RelativeValue | null {
  const valid = points.filter((p) => p[key] !== null);
  if (valid.length === 0) return null;
  const last = valid[valid.length - 1];
  let start = "";
  if (years !== null) {
    const d = new Date(`${last.date}T00:00:00Z`);
    d.setUTCFullYear(d.getUTCFullYear() - years);
    start = d.toISOString().slice(0, 10);
  }
  const window = valid.filter((p) => p.date >= start).map((p) => p[key] as number);
  return relativeValue(window, last[key] as number);
}
