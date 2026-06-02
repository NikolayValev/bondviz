import { describe, it, expect } from "vitest";
import {
  pvContinuous,
  discountFactors,
  computeCurveKpis,
  rowToCurve,
  spreadSeries,
} from "@/lib/finance";

describe("pvContinuous", () => {
  it("matches the closed form for a normal yield", () => {
    const pv = pvContinuous(1000, 0.05, 0.04, 10);
    const expected = (50 * (1 - Math.exp(-0.4))) / 0.04 + 1000 * Math.exp(-0.4);
    expect(pv).toBeCloseTo(expected, 6);
  });
  it("handles the zero-yield limit", () => {
    expect(pvContinuous(1000, 0.05, 0, 10)).toBeCloseTo(50 * 10 + 1000, 6);
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
