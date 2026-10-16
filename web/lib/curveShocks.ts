// Named yield-curve shocks: per-tenor shifts in basis points, applied to a curve.
//
// Every non-parallel shape is built from one coordinate, u(t): maturity on a log scale,
// −1 at the front end (1M and shorter), 0 at a 5Y pivot and +1 at 30Y (and beyond). Log
// maturity spreads the short tenors out the way the curve is quoted, so a "steepener" moves
// 2Y noticeably differently from 3M rather than lumping everything under a year together.

import { bondCashflows, convexity, modifiedDuration, priceFromCashflows } from "@/lib/finance";
import { CurvePoint } from "@/lib/types";

export type ShockKind =
  | "parallel-up"
  | "parallel-down"
  | "bull-steepener"
  | "bear-steepener"
  | "bull-flattener"
  | "bear-flattener"
  | "twist"
  | "butterfly";

export interface ShockSpec {
  kind: ShockKind;
  /** Magnitude in bps (positive); the shape decides where and which way it applies. */
  sizeBps: number;
}

export const SHOCKS: { kind: ShockKind; label: string; description: string }[] = [
  { kind: "parallel-up", label: "Parallel +", description: "Every tenor rises by the same amount." },
  { kind: "parallel-down", label: "Parallel −", description: "Every tenor falls by the same amount." },
  { kind: "bull-steepener", label: "Bull steepener", description: "Rates fall, the front end most (e.g. the market prices in cuts)." },
  { kind: "bear-steepener", label: "Bear steepener", description: "Rates rise, the long end most (e.g. term premium or supply)." },
  { kind: "bull-flattener", label: "Bull flattener", description: "Rates fall, the long end most (e.g. a flight to duration)." },
  { kind: "bear-flattener", label: "Bear flattener", description: "Rates rise, the front end most (e.g. a hawkish central bank)." },
  { kind: "twist", label: "Twist", description: "Front end falls and long end rises around a 5Y pivot (steepening twist)." },
  { kind: "butterfly", label: "Butterfly", description: "Wings rise while the 5Y belly falls (positive butterfly)." },
];

const T_SHORT = 1 / 12;
const T_PIVOT = 5;
const T_LONG = 30;

/** Log-maturity coordinate: −1 at ≤1M, 0 at the 5Y pivot, +1 at ≥30Y. */
export function curvePosition(years: number): number {
  const t = Math.min(T_LONG, Math.max(T_SHORT, years));
  return t <= T_PIVOT
    ? (Math.log(t) - Math.log(T_PIVOT)) / (Math.log(T_PIVOT) - Math.log(T_SHORT))
    : (Math.log(t) - Math.log(T_PIVOT)) / (Math.log(T_LONG) - Math.log(T_PIVOT));
}

/** Shift in bps at maturity `years` for a shock. */
export function shockBps({ kind, sizeBps: s }: ShockSpec, years: number): number {
  const u = curvePosition(years);
  const longW = (u + 1) / 2; // 0 at the front end → 1 at the long end
  switch (kind) {
    case "parallel-up":
      return s;
    case "parallel-down":
      return -s;
    case "bull-steepener":
      return -s * (1 - longW);
    case "bear-steepener":
      return s * longW;
    case "bull-flattener":
      return -s * longW;
    case "bear-flattener":
      return s * (1 - longW);
    case "twist":
      return s * u;
    case "butterfly":
      return s * (2 * Math.abs(u) - 1);
  }
}

/** The curve after a shock: each tenor's yield (percent) moves by its shift. */
export function applyShock(curve: CurvePoint[], spec: ShockSpec): CurvePoint[] {
  return curve.map((p) => ({ ...p, yield: p.yield + shockBps(spec, p.years) / 100 }));
}

export interface BenchmarkImpact {
  label: string;
  years: number;
  yieldBefore: number; // percent
  yieldAfter: number; // percent
  shiftBps: number;
  pctChange: number; // exact reprice, decimal
  approxPctChange: number; // −D·Δy + ½·C·Δy², decimal
}

/**
 * Reprice a benchmark bond at each requested tenor under a shock. Each benchmark pays a
 * semiannual coupon equal to today's yield at its tenor and is priced with the app's
 * continuous-compounding model, before and after its yield moves by the shock at that tenor.
 */
export function benchmarkImpacts(curve: CurvePoint[], spec: ShockSpec, labels: string[]): BenchmarkImpact[] {
  return labels.flatMap((label) => {
    const p = curve.find((c) => c.label === label);
    if (!p) return [];
    const y0 = p.yield / 100;
    const shift = shockBps(spec, p.years);
    const dy = shift / 10_000;
    const { cashflows, times } = bondCashflows(100, y0, p.years, 2);
    const before = priceFromCashflows(cashflows, times, y0);
    const after = priceFromCashflows(cashflows, times, y0 + dy);
    const d = modifiedDuration(cashflows, times, y0);
    const c = convexity(cashflows, times, y0);
    return [{
      label,
      years: p.years,
      yieldBefore: p.yield,
      yieldAfter: p.yield + shift / 100,
      shiftBps: shift,
      pctChange: after / before - 1,
      approxPctChange: -d * dy + 0.5 * c * dy * dy,
    }];
  });
}
