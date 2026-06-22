import { bondCashflows, modifiedDuration } from "@/lib/finance";

export interface CarryPoint {
  label: string;
  years: number;
  yieldPct: number; // y(T) in percent
  carryBps: number; // running-yield carry over the horizon, in bps of return
  rollBps: number; // roll-down return over the horizon, in bps of return
  totalBps: number; // carryBps + rollBps
  breakevenBps: number; // yield sell-off (bps) that zeroes the horizon return
  totalPct: number; // total horizon return, percent
}

export interface CurveInput {
  years: number;
  yieldPct: number;
  label: string;
}

/** Linear interpolation of y at x over sorted (xs, ys); flat-extrapolated.
 *  Mirrors the interp convention in lib/curve.ts. */
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

/**
 * Carry & roll-down per tenor over a holding horizon, assuming a STATIC curve.
 *   carry = y(T)·h                       (running-yield income over horizon)
 *   roll  = Dur(T−h)·(y(T) − y(T−h))     (price gain as the bond ages down the curve)
 * Dur is modified duration (continuous compounding) of a par bond at maturity T−h.
 * Breakeven = total / Dur = bps yields can rise over the horizon before return = 0.
 * Tenors with maturity ≤ horizon are excluded (a bond shorter than the horizon
 * cannot roll for the full period).
 */
export function carryRollDown(curve: CurveInput[], horizonYears: number): CarryPoint[] {
  const clean = curve
    .filter((p) => Number.isFinite(p.years) && Number.isFinite(p.yieldPct))
    .sort((a, b) => a.years - b.years);
  if (clean.length === 0) return [];

  const xs = clean.map((p) => p.years);
  const ys = clean.map((p) => p.yieldPct);
  const h = horizonYears;
  const eps = 1e-9;
  const out: CarryPoint[] = [];

  for (const p of clean) {
    if (p.years <= h + eps) continue;
    const rollMat = p.years - h;
    if (rollMat < xs[0]) continue; // exclude if rolled maturity is before the curve
    const yT = p.yieldPct / 100;
    const yRoll = interp(rollMat, xs, ys) / 100;

    // Modified duration of the rolled bond, priced as a par bond at its own yield.
    const { cashflows, times } = bondCashflows(100, yRoll, rollMat, 2);
    const dur = modifiedDuration(cashflows, times, yRoll);

    const carry = yT * h;
    const roll = dur * (yT - yRoll);
    const total = carry + roll;
    const breakeven = dur > eps ? total / dur : 0;

    out.push({
      label: p.label,
      years: p.years,
      yieldPct: p.yieldPct,
      carryBps: carry * 10_000,
      rollBps: roll * 10_000,
      totalBps: total * 10_000,
      breakevenBps: breakeven * 10_000,
      totalPct: total * 100,
    });
  }
  return out;
}
