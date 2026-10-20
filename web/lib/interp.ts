/** Linear interpolation of y at x over sorted (xs, ys); flat-extrapolated
 *  beyond the endpoints. Shared by the curve bootstrap and carry/roll-down. */
export function interp(x: number, xs: number[], ys: number[]): number {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) {
      const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + (ys[i] - ys[i - 1]) * t;
    }
  }
  return ys[ys.length - 1];
}

/**
 * Monotone piecewise-cubic Hermite interpolation (Fritsch–Carlson, a.k.a. PCHIP), flat beyond
 * the endpoints like `interp`. The first derivative is continuous at every node and the curve
 * never overshoots the data (monotone stretches stay monotone), which is what a par curve needs
 * before bootstrapping: linear interpolation's slope jumps at each tenor become jumps in the
 * forward curve, magnified by maturity. Returns an evaluator so the slopes are computed once.
 */
export function makePchip(xs: number[], ys: number[]): (x: number) => number {
  const n = xs.length;
  if (n < 3) return (x) => interp(x, xs, ys);
  const h = xs.slice(1).map((x, i) => x - xs[i]);
  const delta = h.map((hi, i) => (ys[i + 1] - ys[i]) / hi);
  const d = new Array<number>(n).fill(0);
  for (let k = 1; k < n - 1; k++) {
    if (delta[k - 1] * delta[k] <= 0) continue; // local extremum or flat: zero slope
    const w1 = 2 * h[k] + h[k - 1];
    const w2 = h[k] + 2 * h[k - 1];
    d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k]);
  }
  // One-sided three-point endpoint slopes, limited so they keep the data's shape.
  const end = (h0: number, h1: number, d0: number, d1: number) => {
    const s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if (Math.sign(s) !== Math.sign(d0)) return 0;
    if (Math.sign(d0) !== Math.sign(d1) && Math.abs(s) > Math.abs(3 * d0)) return 3 * d0;
    return s;
  };
  d[0] = end(h[0], h[1], delta[0], delta[1]);
  d[n - 1] = end(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);

  return (x) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let k = 0;
    while (x > xs[k + 1]) k++;
    const t = (x - xs[k]) / h[k];
    const t2 = t * t;
    const t3 = t2 * t;
    return (
      (2 * t3 - 3 * t2 + 1) * ys[k] +
      (t3 - 2 * t2 + t) * h[k] * d[k] +
      (-2 * t3 + 3 * t2) * ys[k + 1] +
      (t3 - t2) * h[k] * d[k + 1]
    );
  };
}
