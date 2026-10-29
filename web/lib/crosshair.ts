// Value lookup behind chart crosshairs: snap a pointer to the nearest data x, and read each
// series at that x the way its line is drawn (straight segments between points).

/** Index of the value in a sorted array closest to `x` (ties go left); -1 when empty. */
export function nearestIndex(sorted: number[], x: number): number {
  if (sorted.length === 0) return -1;
  let lo = 0;
  let hi = sorted.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] <= x) lo = mid;
    else hi = mid;
  }
  return Math.abs(x - sorted[lo]) <= Math.abs(sorted[hi] - x) ? lo : hi;
}

/**
 * A series' value at `x`: exact where it has a point, linear between its neighbours (what the
 * drawn line shows), and null outside its range, so a shorter series isn't extrapolated.
 * Points must be sorted by x.
 */
export function valueAt(points: [number, number][], x: number): number | null {
  const n = points.length;
  if (n === 0 || x < points[0][0] || x > points[n - 1][0]) return null;
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid][0] <= x) lo = mid;
    else hi = mid;
  }
  const [x0, y0] = points[lo];
  const [x1, y1] = points[hi];
  if (x === x0) return y0;
  if (x === x1) return y1;
  return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
}

/** Decimals that suit a readout given the data's span: 4.26 for yields, 310.84 for prices,
 *  0.312 for loadings, 1,234 for large values. */
export function readoutDecimals(span: number): number {
  const s = Math.abs(span);
  if (s < 0.5) return 3;
  if (s < 100) return 2;
  if (s < 1000) return 1;
  return 0;
}
