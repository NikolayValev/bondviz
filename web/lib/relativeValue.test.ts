import { describe, it, expect } from "vitest";
import { relativeValue, spreadRelativeValue } from "@/lib/relativeValue";
import type { SpreadPoint } from "@/lib/signal";

describe("relativeValue", () => {
  it("matches a hand-computed fixture", () => {
    // [1,2,3,4,5]: mean 3, population variance 2, std √2
    const rv = relativeValue([1, 2, 3, 4, 5])!;
    expect(rv.mean).toBe(3);
    expect(rv.std).toBeCloseTo(Math.SQRT2, 12);
    expect(rv.z).toBeCloseTo(2 / Math.SQRT2, 12);
    expect(rv.percentile).toBe(100);
    expect([rv.min, rv.max, rv.n]).toEqual([1, 5, 5]);
  });

  it("scores an explicit current value against the window", () => {
    const rv = relativeValue([1, 2, 3, 4, 5], 3)!;
    expect(rv.z).toBe(0);
    expect(rv.percentile).toBe(60); // 1, 2, 3 are at or below 3
    expect(relativeValue([1, 2, 3, 4, 5], 0)!.percentile).toBe(0);
  });

  it("gives z = 0 for a flat window and null for too little data", () => {
    expect(relativeValue([2, 2, 2])!.z).toBe(0);
    // 0.3 × 40 doesn't sum exactly; rounding must not turn a flat series into a huge z.
    const flat = relativeValue(Array(40).fill(0.3))!;
    expect(flat.z).toBe(0);
    expect(flat.std).toBe(0);
    expect(relativeValue([1])).toBeNull();
    expect(relativeValue([])).toBeNull();
  });

  it("ignores non-finite values", () => {
    expect(relativeValue([1, NaN, 3, Infinity])!.n).toBe(2);
  });
});

describe("spreadRelativeValue", () => {
  const pts: SpreadPoint[] = [
    { date: "2014-06-01", s10y3m: -1, s2s10s: 0.5 },
    { date: "2016-06-01", s10y3m: 1, s2s10s: null },
    { date: "2020-06-01", s10y3m: 2, s2s10s: 0.2 },
    { date: "2024-06-01", s10y3m: 3, s2s10s: 0.1 },
    { date: "2026-06-01", s10y3m: 4, s2s10s: 0.4 },
  ];

  it("uses the whole history when years is null", () => {
    const rv = spreadRelativeValue(pts, "s10y3m", null)!;
    expect(rv.n).toBe(5);
    expect(rv.current).toBe(4);
    expect(rv.percentile).toBe(100);
  });

  it("restricts to a trailing window from the latest date", () => {
    expect(spreadRelativeValue(pts, "s10y3m", 10)!.n).toBe(4); // 2016-06-01 onward
    expect(spreadRelativeValue(pts, "s10y3m", 2)!.n).toBe(2); // 2024-06-01 onward
  });

  it("skips null observations for that spread", () => {
    const rv = spreadRelativeValue(pts, "s2s10s", null)!;
    expect(rv.n).toBe(4);
    expect(rv.current).toBe(0.4);
    expect(rv.percentile).toBe(75);
  });

  it("returns null when the spread has no data", () => {
    expect(spreadRelativeValue([], "s10y3m", null)).toBeNull();
  });
});
