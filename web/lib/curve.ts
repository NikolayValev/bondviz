// Par → zero/forward bootstrap. Mirrors src/bondviz/visualizer.py
// (bootstrap_zeros_from_par): par yields are SEMIANNUAL par coupons; the
// resulting zero rates are ANNUAL-COMPOUNDED. Do not conflate the two
// conventions. The grid is regular (0.5Y steps), so discount factors are
// solved in order and every prior factor already exists (no interpolation
// of missing nodes is needed).

export interface ParPoint {
  years: number;
  yieldPct: number; // par yield in percent
}

export interface BootstrapResult {
  grid: number[];    // semiannual grid times (0.5, 1.0, …, maxT)
  df: number[];      // discount factor at each grid time
  zero: number[];    // annual-compounded zero rate (decimal) at each grid time
  forward: number[]; // 6-month implied forward (annual-compounded, decimal)
}

/** Linear interpolation of y at x given sorted (xs, ys), flat-extrapolated. */
function interp(x: number, xs: number[], ys: number[]): number {
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

export function bootstrapZeros(par: ParPoint[]): BootstrapResult {
  const empty: BootstrapResult = { grid: [], df: [], zero: [], forward: [] };
  const clean = par
    .filter((p) => Number.isFinite(p.years) && Number.isFinite(p.yieldPct))
    .sort((a, b) => a.years - b.years);
  if (clean.length === 0) return empty;

  const knownT = clean.map((p) => p.years);
  const knownR = clean.map((p) => p.yieldPct / 100); // to decimals
  const maxT = knownT[knownT.length - 1];

  const freq = 2;
  const grid: number[] = [];
  for (let t = 0.5; t <= maxT + 1e-9; t += 0.5) grid.push(Number(t.toFixed(10)));
  if (grid.length === 0) return empty;

  const rGrid = grid.map((t) => interp(t, knownT, knownR));

  const df: number[] = [];
  for (let i = 0; i < grid.length; i++) {
    const n = i + 1; // number of semiannual periods to this node
    const c = rGrid[i] / freq;
    if (n === 1) {
      df.push(1 / (1 + c));
    } else {
      let s = 0;
      for (let k = 0; k < n - 1; k++) s += df[k];
      df.push(Math.max(1e-9, (1 - c * s) / (1 + c)));
    }
  }

  const zero = grid.map((T, i) => Math.pow(df[i], -1 / T) - 1);
  const forward = grid.map((_, i) => {
    const dPrev = i === 0 ? 1 : df[i - 1];
    return Math.pow(dPrev / df[i], 1 / 0.5) - 1; // annualize the 0.5Y period
  });

  return { grid, df, zero, forward };
}
