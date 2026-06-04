import { describe, it, expect } from "vitest";
import {
  bondCashflows,
  convexity,
  dv01,
  modifiedDuration,
  pvContinuous,
} from "@/lib/finance";
import { portfolioMetrics, portfolioScenario } from "@/lib/portfolio";

const A = { face: 100_000, coupon: 0.05, yield_: 0.04, years: 10, freq: 2 };
const B = { face: 50_000, coupon: 0.03, yield_: 0.045, years: 2, freq: 2 };

describe("portfolioMetrics", () => {
  it("a single holding reproduces that bond's value and duration", () => {
    const p = portfolioMetrics([A]);
    expect(p.totalValue).toBeCloseTo(pvContinuous(A.face, A.coupon, A.yield_, A.years, A.freq), 6);
    const { cashflows, times } = bondCashflows(A.face, A.coupon, A.years, A.freq);
    expect(p.weightedDuration).toBeCloseTo(modifiedDuration(cashflows, times, A.yield_), 9);
    expect(p.weightedConvexity).toBeCloseTo(convexity(cashflows, times, A.yield_), 9);
    expect(p.weights[0]).toBeCloseTo(1, 9);
  });

  it("total value is the sum of holding market values", () => {
    const p = portfolioMetrics([A, B]);
    const va = pvContinuous(A.face, A.coupon, A.yield_, A.years, A.freq);
    const vb = pvContinuous(B.face, B.coupon, B.yield_, B.years, B.freq);
    expect(p.totalValue).toBeCloseTo(va + vb, 6);
  });

  it("portfolio DV01 is additive across holdings", () => {
    const p = portfolioMetrics([A, B]);
    const ca = bondCashflows(A.face, A.coupon, A.years, A.freq);
    const cb = bondCashflows(B.face, B.coupon, B.years, B.freq);
    const expected = dv01(ca.cashflows, ca.times, A.yield_) + dv01(cb.cashflows, cb.times, B.yield_);
    expect(p.dv01).toBeCloseTo(expected, 6);
  });

  it("weighted duration sits between the two holdings' durations", () => {
    const p = portfolioMetrics([A, B]);
    const da = modifiedDuration(bondCashflows(A.face, A.coupon, A.years, A.freq).cashflows, bondCashflows(A.face, A.coupon, A.years, A.freq).times, A.yield_);
    const db = modifiedDuration(bondCashflows(B.face, B.coupon, B.years, B.freq).cashflows, bondCashflows(B.face, B.coupon, B.years, B.freq).times, B.yield_);
    const lo = Math.min(da, db);
    const hi = Math.max(da, db);
    expect(p.weightedDuration).toBeGreaterThan(lo);
    expect(p.weightedDuration).toBeLessThan(hi);
  });

  it("an empty portfolio is all zeros and does not divide by zero", () => {
    const p = portfolioMetrics([]);
    expect(p.totalValue).toBe(0);
    expect(p.weightedDuration).toBe(0);
    expect(Number.isNaN(p.weightedDuration)).toBe(false);
  });
});

describe("portfolioScenario", () => {
  it("zero shift reproduces total value exactly", () => {
    const rows = portfolioScenario([A, B], [-50, 0, 50]);
    const base = portfolioMetrics([A, B]).totalValue;
    const zero = rows.find((r) => r.shiftBps === 0)!;
    expect(zero.newPrice).toBe(base);
    expect(zero.dollarChange).toBe(0);
  });

  it("a yield rise lowers portfolio value", () => {
    const rows = portfolioScenario([A, B], [0, 100]);
    const base = rows.find((r) => r.shiftBps === 0)!.newPrice;
    expect(rows.find((r) => r.shiftBps === 100)!.newPrice).toBeLessThan(base);
  });

  it("the approximation tracks the reprice for a small shift", () => {
    const rows = portfolioScenario([A, B], [25]);
    const r = rows.find((x) => x.shiftBps === 25)!;
    expect(Math.abs(r.approxPctChange - r.pctChange)).toBeLessThan(5e-4);
  });
});
