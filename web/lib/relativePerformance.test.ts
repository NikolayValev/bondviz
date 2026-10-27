import { describe, it, expect } from "vitest";
import { comparePerformance, rebaseTo100 } from "@/lib/relativePerformance";
import type { StockBar } from "@/lib/types";

const bar = (date: string, close: number): StockBar => ({
  date,
  open: close,
  high: close,
  low: close,
  close,
  volume: 1,
  vwap: null,
});

describe("rebaseTo100", () => {
  it("indexes closes to the first close, keyed by UTC-midnight epoch-ms", () => {
    const pts = rebaseTo100([bar("2026-10-01", 50), bar("2026-10-02", 55), bar("2026-10-05", 45)]);
    expect(pts.map((p) => p[0])).toEqual([
      Date.UTC(2026, 9, 1),
      Date.UTC(2026, 9, 2),
      Date.UTC(2026, 9, 5),
    ]);
    expect(pts[0][1]).toBe(100);
    expect(pts[1][1]).toBeCloseTo(110, 10);
    expect(pts[2][1]).toBeCloseTo(90, 10);
  });

  it("skips unusable closes, including a bad first one", () => {
    const pts = rebaseTo100([bar("2026-10-01", 0), bar("2026-10-02", NaN), bar("2026-10-05", 20), bar("2026-10-06", 30)]);
    expect(pts.map((p) => p[1])).toEqual([100, 150]);
  });

  it("returns nothing without data", () => {
    expect(rebaseTo100([])).toEqual([]);
  });
});

describe("comparePerformance", () => {
  it("ranks window returns, best first, and flags late starters", () => {
    const rows = comparePerformance([
      { ticker: "AAA", bars: [bar("2026-10-01", 100), bar("2026-10-20", 104)] },
      { ticker: "BBB", bars: [bar("2026-10-01", 10), bar("2026-10-20", 12)] },
      { ticker: "NEW", bars: [bar("2026-10-10", 20), bar("2026-10-20", 19)] },
      { ticker: "NONE", bars: [] },
    ]);
    expect(rows.map((r) => r.ticker)).toEqual(["BBB", "AAA", "NEW"]);
    expect(rows[0].change).toBeCloseTo(0.2, 10);
    expect(rows[0].lastClose).toBe(12);
    expect(rows.map((r) => r.startsLate)).toEqual([false, false, true]);
  });

  it("handles no data at all", () => {
    expect(comparePerformance([])).toEqual([]);
  });
});
