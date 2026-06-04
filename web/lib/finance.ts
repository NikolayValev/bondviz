import { CurvePoint, Kpis, TenorLabel, YieldRow } from "@/lib/types";

export const TENOR_YEARS: Record<TenorLabel, number> = {
  "1M": 1 / 12, "2M": 2 / 12, "3M": 3 / 12, "6M": 6 / 12,
  "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30,
};

export const BC_TO_TENOR: Record<string, TenorLabel> = {
  BC_1MONTH: "1M", BC_2MONTH: "2M", BC_3MONTH: "3M", BC_6MONTH: "6M",
  BC_1YEAR: "1Y", BC_2YEAR: "2Y", BC_3YEAR: "3Y", BC_5YEAR: "5Y",
  BC_7YEAR: "7Y", BC_10YEAR: "10Y", BC_20YEAR: "20Y", BC_30YEAR: "30Y",
};

/** Convert a spread/level in percentage points to basis points. */
export const toBps = (pp: number): number => pp * 100;

// --- Bond pricing & risk analytics (continuous compounding) -----------------
// Mirrors src/bondviz/pricing.py and scenarios.py. The single source of truth
// for a bond's cashflows is bondCashflows; pricing and the duration/convexity
// analytics both build on it so they stay numerically consistent.

export interface Cashflows {
  cashflows: number[];
  times: number[];
}

/** Discrete coupon schedule: payment dates 1/freq … years and per-period
 *  coupons (face·coupon/freq) with the face value added to the final payment. */
export function bondCashflows(face: number, coupon: number, years: number, freq = 2): Cashflows {
  const n = Math.max(1, Math.round(years * freq));
  const couponPmt = (face * coupon) / freq;
  const times: number[] = [];
  const cashflows: number[] = [];
  for (let i = 1; i <= n; i++) {
    times.push(i / freq);
    cashflows.push(couponPmt);
  }
  cashflows[n - 1] += face;
  return { cashflows, times };
}

/** Present value of explicit cashflows under continuous compounding. */
export function priceFromCashflows(cashflows: number[], times: number[], yieldRate: number): number {
  let pv = 0;
  for (let i = 0; i < cashflows.length; i++) pv += cashflows[i] * Math.exp(-yieldRate * times[i]);
  return pv;
}

/** Present value of a fixed-coupon bond — discounts a discrete schedule so the
 *  price is consistent with the duration/convexity analytics below. */
export function pvContinuous(face: number, coupon: number, yieldRate: number, years: number, freq = 2): number {
  const { cashflows, times } = bondCashflows(face, coupon, years, freq);
  return priceFromCashflows(cashflows, times, yieldRate);
}

/** Macaulay duration = Σ(tᵢ·CFᵢ·e^(−y·tᵢ)) / price. */
export function macaulayDuration(cashflows: number[], times: number[], yieldRate: number): number {
  const price = priceFromCashflows(cashflows, times, yieldRate);
  let weighted = 0;
  for (let i = 0; i < cashflows.length; i++) weighted += times[i] * cashflows[i] * Math.exp(-yieldRate * times[i]);
  return weighted / price;
}

/** Modified duration. Under continuous compounding it equals Macaulay duration
 *  (no 1/(1+y/m) adjustment); kept as its own function to document intent. */
export function modifiedDuration(cashflows: number[], times: number[], yieldRate: number): number {
  return macaulayDuration(cashflows, times, yieldRate);
}

/** Convexity = Σ(tᵢ²·CFᵢ·e^(−y·tᵢ)) / price. */
export function convexity(cashflows: number[], times: number[], yieldRate: number): number {
  const price = priceFromCashflows(cashflows, times, yieldRate);
  let weighted = 0;
  for (let i = 0; i < cashflows.length; i++) weighted += times[i] * times[i] * cashflows[i] * Math.exp(-yieldRate * times[i]);
  return weighted / price;
}

export interface BondParams {
  face: number;
  coupon: number;
  yield_: number;
  years: number;
  freq?: number;
}

export interface ScenarioRow {
  shiftBps: number;
  newYield: number;
  newPrice: number;
  dollarChange: number;
  pctChange: number;
  approxPctChange: number;
}

/** Re-price a bond across parallel yield shifts (bps). Each row carries the
 *  full reprice and the duration+convexity approximation of the % change
 *  (ΔP/P ≈ −D·Δy + ½·C·Δy²) so the UI can show how the quadratic tracks. */
export function scenarioShift(params: BondParams, shiftsBps: number[]): ScenarioRow[] {
  const freq = params.freq ?? 2;
  const { cashflows, times } = bondCashflows(params.face, params.coupon, params.years, freq);
  const basePrice = priceFromCashflows(cashflows, times, params.yield_);
  const dur = modifiedDuration(cashflows, times, params.yield_);
  const conv = convexity(cashflows, times, params.yield_);

  return shiftsBps.map((shiftBps) => {
    const dy = shiftBps / 10_000;
    const newYield = params.yield_ + dy;
    const newPrice = priceFromCashflows(cashflows, times, newYield);
    const dollarChange = newPrice - basePrice;
    return {
      shiftBps,
      newYield,
      newPrice,
      dollarChange,
      pctChange: dollarChange / basePrice,
      approxPctChange: -dur * dy + 0.5 * conv * dy * dy,
    };
  });
}

/** Turn selected ± magnitudes (bps) into a sorted symmetric set including 0. */
export function symmetricShifts(magnitudes: number[]): number[] {
  const set = new Set<number>([0]);
  for (const m of magnitudes) {
    set.add(m);
    set.add(-m);
  }
  return [...set].sort((a, b) => a - b);
}

export function discountFactors(yieldRate: number, tenors: number[]): { t: number; df: number }[] {
  return tenors.map((t) => ({ t, df: Math.exp(-yieldRate * t) }));
}

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && !Number.isNaN(v) ? v : null;
}

export function computeCurveKpis(row: Record<string, unknown>): Kpis {
  const y10 = numOrNull(row.BC_10YEAR);
  const y2 = numOrNull(row.BC_2YEAR);
  const y3m = numOrNull(row.BC_3MONTH);
  return {
    tenYear: y10,
    twos10s: y10 !== null && y2 !== null ? y10 - y2 : null,
    threeM10Y: y10 !== null && y3m !== null ? y10 - y3m : null,
  };
}

export function rowToCurve(row: Record<string, unknown>): CurvePoint[] {
  const pts: CurvePoint[] = [];
  for (const [bc, label] of Object.entries(BC_TO_TENOR)) {
    const v = numOrNull(row[bc]);
    if (v !== null) pts.push({ label, years: TENOR_YEARS[label], yield: v });
  }
  return pts.sort((a, b) => a.years - b.years);
}

export function spreadSeries(rows: YieldRow[]): {
  twos10s: [number, number][];
  threeM10Y: [number, number][];
} {
  const twos10s: [number, number][] = [];
  const threeM10Y: [number, number][] = [];
  for (const r of rows) {
    const t = new Date(r.date).getTime();
    const y10 = numOrNull(r.BC_10YEAR);
    const y2 = numOrNull(r.BC_2YEAR);
    const y3m = numOrNull(r.BC_3MONTH);
    if (y10 !== null && y2 !== null) twos10s.push([t, y10 - y2]);
    if (y10 !== null && y3m !== null) threeM10Y.push([t, y10 - y3m]);
  }
  return { twos10s, threeM10Y };
}

export function describeCurve(curve: CurvePoint[]): string {
  const front = curve.find((p) => ["3M", "6M", "1Y", "2Y"].includes(p.label));
  const long = [...curve].reverse().find((p) => ["10Y", "20Y", "30Y"].includes(p.label));
  if (!front || !long) return "Incomplete tenor coverage in this snapshot.";
  const slope = long.yield - front.yield;
  if (slope < -0.1) return `Inverted: ${long.label} (${long.yield.toFixed(2)}%) below ${front.label} (${front.yield.toFixed(2)}%).`;
  if (slope < 0.1) return `Roughly flat between ${front.label} and ${long.label}.`;
  return `Upward sloping: ${long.label} (${long.yield.toFixed(2)}%) above ${front.label} (${front.yield.toFixed(2)}%).`;
}
