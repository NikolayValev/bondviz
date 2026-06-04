import { describe, it, expect } from "vitest";
import {
  pvContinuous,
  bondCashflows,
  priceFromCashflows,
  macaulayDuration,
  modifiedDuration,
  convexity,
  scenarioShift,
  symmetricShifts,
  dv01,
  currentYield,
  couponIncome,
  solveYield,
  priceYieldCurve,
  discountedCashflows,
  horizonReturn,
  discountFactors,
  computeCurveKpis,
  rowToCurve,
  spreadSeries,
} from "@/lib/finance";

// These mirror tests/test_analytics.py and tests/test_scenarios.py so the TS
// and Python implementations stay in agreement.

describe("bondCashflows", () => {
  it("builds a discrete semiannual schedule with face on the last payment", () => {
    const { cashflows, times } = bondCashflows(1000, 0.05, 10, 2);
    expect(cashflows.length).toBe(20);
    expect(times.length).toBe(20);
    expect(times[0]).toBeCloseTo(0.5, 9);
    expect(times[times.length - 1]).toBeCloseTo(10, 9);
    expect(cashflows[0]).toBeCloseTo(25, 9);
    expect(cashflows[cashflows.length - 1]).toBeCloseTo(1025, 9);
  });
});

describe("pvContinuous", () => {
  it("equals the discrete cashflow sum under continuous compounding", () => {
    const { cashflows, times } = bondCashflows(1000, 0.05, 10, 2);
    const expected = cashflows.reduce((acc, cf, i) => acc + cf * Math.exp(-0.04 * times[i]), 0);
    expect(pvContinuous(1000, 0.05, 0.04, 10)).toBeCloseTo(expected, 9);
  });
});

describe("duration & convexity", () => {
  it("zero-coupon Macaulay duration equals maturity", () => {
    const { cashflows, times } = bondCashflows(1000, 0, 7, 2);
    expect(macaulayDuration(cashflows, times, 0.04)).toBeCloseTo(7, 9);
  });
  it("modified equals Macaulay under continuous compounding", () => {
    const { cashflows, times } = bondCashflows(1000, 0.05, 10, 2);
    expect(modifiedDuration(cashflows, times, 0.04)).toBeCloseTo(
      macaulayDuration(cashflows, times, 0.04),
      12,
    );
  });
  it("duration and convexity are positive for a coupon bond", () => {
    const { cashflows, times } = bondCashflows(1000, 0.05, 10, 2);
    expect(macaulayDuration(cashflows, times, 0.04)).toBeGreaterThan(0);
    expect(convexity(cashflows, times, 0.04)).toBeGreaterThan(0);
  });
  it("longer maturity has larger duration", () => {
    const short = bondCashflows(1000, 0.05, 5, 2);
    const long = bondCashflows(1000, 0.05, 20, 2);
    expect(macaulayDuration(long.cashflows, long.times, 0.04)).toBeGreaterThan(
      macaulayDuration(short.cashflows, short.times, 0.04),
    );
  });
  it("duration predicts a small yield bump", () => {
    const { cashflows, times } = bondCashflows(1000, 0.05, 10, 2);
    const base = priceFromCashflows(cashflows, times, 0.04);
    const bumped = priceFromCashflows(cashflows, times, 0.041);
    const predicted = base * (1 - modifiedDuration(cashflows, times, 0.04) * 0.001);
    expect(bumped).toBeLessThan(base);
    // First-order prediction: right sign and rough magnitude (relative tol, like the Python test).
    expect(Math.abs(predicted - bumped) / bumped).toBeLessThan(1e-3);
  });
});

describe("scenarioShift", () => {
  const BOND = { face: 1000, coupon: 0.05, yield_: 0.04, years: 10, freq: 2 };
  const row = (rows: ReturnType<typeof scenarioShift>, bps: number) =>
    rows.find((r) => r.shiftBps === bps)!;

  it("zero shift reproduces the base price exactly", () => {
    const rows = scenarioShift(BOND, [-50, 0, 50]);
    const base = pvContinuous(BOND.face, BOND.coupon, BOND.yield_, BOND.years, BOND.freq);
    expect(row(rows, 0).newPrice).toBe(base);
    expect(row(rows, 0).dollarChange).toBe(0);
    expect(row(rows, 0).pctChange).toBe(0);
  });
  it("price moves opposite to yield", () => {
    const rows = scenarioShift(BOND, [-100, 0, 100]);
    const base = row(rows, 0).newPrice;
    expect(row(rows, 100).newPrice).toBeLessThan(base);
    expect(row(rows, -100).newPrice).toBeGreaterThan(base);
  });
  it("new yield reflects the shift", () => {
    const rows = scenarioShift(BOND, [25]);
    expect(row(rows, 25).newYield).toBeCloseTo(0.0425, 9);
  });
  it("approximation tracks reprice for ±25 bps and error grows with size", () => {
    const rows = scenarioShift(BOND, [25, 200]);
    const errSmall = Math.abs(row(rows, 25).approxPctChange - row(rows, 25).pctChange);
    const errLarge = Math.abs(row(rows, 200).approxPctChange - row(rows, 200).pctChange);
    expect(errSmall).toBeLessThan(errLarge);
    expect(errSmall).toBeLessThan(0.0005);
  });
});

describe("symmetricShifts", () => {
  it("dedups, mirrors, and includes zero", () => {
    expect(symmetricShifts([100, 25, 50])).toEqual([-100, -50, -25, 0, 25, 50, 100]);
    expect(symmetricShifts([])).toEqual([0]);
    expect(symmetricShifts([50, 50])).toEqual([-50, 0, 50]);
  });
});

describe("dv01", () => {
  it("approximates the price drop for a +1bp yield rise", () => {
    const { cashflows, times } = bondCashflows(1000, 0.05, 10, 2);
    const exact = priceFromCashflows(cashflows, times, 0.04) - priceFromCashflows(cashflows, times, 0.0401);
    expect(dv01(cashflows, times, 0.04)).toBeGreaterThan(0);
    expect(dv01(cashflows, times, 0.04)).toBeCloseTo(exact, 3);
  });
});

describe("currentYield & couponIncome", () => {
  it("current yield is annual coupon over price", () => {
    expect(currentYield(1000, 0.05, 1250)).toBeCloseTo(50 / 1250, 12);
  });
  it("coupon income is the annual coupon in dollars", () => {
    expect(couponIncome(1000, 0.05)).toBeCloseTo(50, 12);
  });
});

describe("solveYield", () => {
  it("round-trips: price at y0 solves back to y0", () => {
    const price = pvContinuous(1000, 0.05, 0.037, 10, 2);
    expect(solveYield(1000, 0.05, 10, 2, price)).toBeCloseTo(0.037, 6);
  });
  it("a price below par implies a yield above the coupon", () => {
    const price = pvContinuous(1000, 0.05, 0.06, 8, 2);
    expect(solveYield(1000, 0.05, 8, 2, price)).toBeGreaterThan(0.05);
  });
});

describe("priceYieldCurve", () => {
  it("is downward sloping and matches pvContinuous at each yield", () => {
    const pts = priceYieldCurve(1000, 0.05, 10, 2, [0.02, 0.04, 0.06]);
    expect(pts[0].price).toBeGreaterThan(pts[2].price);
    expect(pts[1].price).toBeCloseTo(pvContinuous(1000, 0.05, 0.04, 10, 2), 9);
  });
});

describe("discountedCashflows", () => {
  it("per-period PVs sum to the bond price", () => {
    const { cashflows, times } = bondCashflows(1000, 0.05, 10, 2);
    const dcf = discountedCashflows(cashflows, times, 0.04);
    const sum = dcf.reduce((a, d) => a + d.pv, 0);
    expect(sum).toBeCloseTo(priceFromCashflows(cashflows, times, 0.04), 9);
    expect(dcf[dcf.length - 1].cf).toBeCloseTo(1025, 9);
  });
});

describe("horizonReturn", () => {
  const BOND = { face: 1000, coupon: 0.05, yield_: 0.04, years: 10, freq: 2 };

  it("at maturity with no shift, total value is face plus all coupons", () => {
    const r = horizonReturn(BOND, 10, 0);
    expect(r.couponIncome).toBeCloseTo(1000 + 50 * 10, 6); // 20 semiannual coupons of 25 + face
    expect(r.bondValue).toBeCloseTo(0, 9);
    expect(r.totalReturn).toBeGreaterThan(0);
  });
  it("for a horizon before maturity, a yield rise lowers the bond's horizon value", () => {
    const base = horizonReturn(BOND, 3, 0);
    const up = horizonReturn(BOND, 3, 100);
    expect(up.bondValue).toBeLessThan(base.bondValue);
    expect(base.couponIncome).toBeCloseTo(25 * 6, 9); // 6 semiannual coupons by year 3
  });
});

describe("discountFactors", () => {
  it("returns e^-rt per tenor", () => {
    const dfs = discountFactors(0.04, [0, 1, 2]);
    expect(dfs[0]).toEqual({ t: 0, df: 1 });
    expect(dfs[2].df).toBeCloseTo(Math.exp(-0.08), 6);
  });
});

describe("computeCurveKpis", () => {
  it("computes 10Y, 2s10s, 3m10y", () => {
    const k = computeCurveKpis({ BC_3MONTH: 5.0, BC_2YEAR: 4.0, BC_10YEAR: 4.5 });
    expect(k.tenYear).toBe(4.5);
    expect(k.twos10s).toBeCloseTo(0.5, 9);
    expect(k.threeM10Y).toBeCloseTo(-0.5, 9);
  });
  it("returns null for missing/NaN inputs", () => {
    const k = computeCurveKpis({ BC_10YEAR: 4.5 });
    expect(k.tenYear).toBe(4.5);
    expect(k.twos10s).toBeNull();
    expect(k.threeM10Y).toBeNull();
  });
});

describe("rowToCurve", () => {
  it("builds sorted curve points from BC_* columns", () => {
    const pts = rowToCurve({ date: "2025-01-02", BC_10YEAR: 4.5, BC_2YEAR: 4.0, BC_1MONTH: 5.2 });
    expect(pts.map((p) => p.label)).toEqual(["1M", "2Y", "10Y"]);
    expect(pts[0].years).toBeCloseTo(1 / 12, 9);
  });
});

describe("spreadSeries", () => {
  it("builds 2s10s and 3m10y time series in pp", () => {
    const { twos10s, threeM10Y } = spreadSeries([
      { date: "2025-01-02", BC_3MONTH: 5.0, BC_2YEAR: 4.0, BC_10YEAR: 4.5 },
    ]);
    expect(twos10s[0][1]).toBeCloseTo(0.5, 9);
    expect(threeM10Y[0][1]).toBeCloseTo(-0.5, 9);
  });
});
