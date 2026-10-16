import { describe, it, expect } from "vitest";
import { applyShock, benchmarkImpacts, curvePosition, shockBps, SHOCKS, type ShockKind } from "@/lib/curveShocks";
import { CurvePoint } from "@/lib/types";

const at = (kind: ShockKind, years: number, sizeBps = 50) => shockBps({ kind, sizeBps }, years);

describe("curvePosition", () => {
  it("is −1 at the front end, 0 at the 5Y pivot, +1 at 30Y, and clamps beyond", () => {
    expect(curvePosition(1 / 12)).toBeCloseTo(-1, 12);
    expect(curvePosition(5)).toBeCloseTo(0, 12);
    expect(curvePosition(30)).toBeCloseTo(1, 12);
    expect(curvePosition(0.01)).toBeCloseTo(-1, 12);
    expect(curvePosition(50)).toBeCloseTo(1, 12);
  });
  it("increases with maturity", () => {
    const ts = [1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30];
    const us = ts.map(curvePosition);
    for (let i = 1; i < us.length; i++) expect(us[i]).toBeGreaterThan(us[i - 1]);
  });
});

describe("shockBps", () => {
  it("parallel shocks move every tenor by the size", () => {
    for (const t of [0.25, 5, 30]) {
      expect(at("parallel-up", t)).toBe(50);
      expect(at("parallel-down", t)).toBe(-50);
    }
  });

  it.each([
    // kind, front (1M), pivot (5Y), long (30Y)
    ["bull-steepener", -50, -25, 0],
    ["bear-steepener", 0, 25, 50],
    ["bull-flattener", 0, -25, -50],
    ["bear-flattener", 50, 25, 0],
    ["twist", -50, 0, 50],
    ["butterfly", 50, -50, 50],
  ] as [ShockKind, number, number, number][])("%s: front %d, 5Y %d, 30Y %d bps", (kind, front, pivot, long) => {
    expect(at(kind, 1 / 12)).toBeCloseTo(front, 9);
    expect(at(kind, 5)).toBeCloseTo(pivot, 9);
    expect(at(kind, 30)).toBeCloseTo(long, 9);
  });

  it("steepeners steepen and flatteners flatten (10Y − 2Y moves the right way)", () => {
    const slope = (k: ShockKind) => at(k, 10) - at(k, 2);
    expect(slope("bull-steepener")).toBeGreaterThan(0);
    expect(slope("bear-steepener")).toBeGreaterThan(0);
    expect(slope("bull-flattener")).toBeLessThan(0);
    expect(slope("bear-flattener")).toBeLessThan(0);
  });

  it("bull shocks lower and bear shocks raise every tenor", () => {
    for (const t of [0.25, 2, 10, 30]) {
      expect(at("bull-steepener", t)).toBeLessThanOrEqual(0);
      expect(at("bull-flattener", t)).toBeLessThanOrEqual(0);
      expect(at("bear-steepener", t)).toBeGreaterThanOrEqual(0);
      expect(at("bear-flattener", t)).toBeGreaterThanOrEqual(0);
    }
  });

  it("scales linearly with size", () => {
    for (const { kind } of SHOCKS) expect(shockBps({ kind, sizeBps: 100 }, 2)).toBeCloseTo(2 * at(kind, 2), 9);
  });
});

describe("applyShock", () => {
  const curve: CurvePoint[] = [
    { label: "3M", years: 0.25, yield: 4.2 },
    { label: "10Y", years: 10, yield: 4.5 },
  ];
  it("moves yields (percent) by the per-tenor shift (bps) and keeps labels", () => {
    const out = applyShock(curve, { kind: "parallel-up", sizeBps: 25 });
    expect(out.map((p) => p.label)).toEqual(["3M", "10Y"]);
    expect(out[0].yield).toBeCloseTo(4.45, 12);
    expect(out[1].yield).toBeCloseTo(4.75, 12);
  });
  it("does not mutate the input", () => {
    applyShock(curve, { kind: "twist", sizeBps: 50 });
    expect(curve[0].yield).toBe(4.2);
  });
});

describe("benchmarkImpacts", () => {
  const curve: CurvePoint[] = [
    { label: "2Y", years: 2, yield: 4.0 },
    { label: "10Y", years: 10, yield: 4.5 },
    { label: "30Y", years: 30, yield: 4.8 },
  ];

  it("reprices each benchmark by its own tenor's shift; longer bonds move more", () => {
    const out = benchmarkImpacts(curve, { kind: "parallel-up", sizeBps: 100 }, ["2Y", "10Y", "30Y"]);
    expect(out.map((b) => b.shiftBps)).toEqual([100, 100, 100]);
    expect(out[0].yieldAfter).toBeCloseTo(5.0, 12);
    for (const b of out) expect(b.pctChange).toBeLessThan(0);
    expect(Math.abs(out[2].pctChange)).toBeGreaterThan(Math.abs(out[1].pctChange));
    expect(Math.abs(out[1].pctChange)).toBeGreaterThan(Math.abs(out[0].pctChange));
  });

  it("the duration-convexity estimate tracks the exact reprice for small shocks", () => {
    const [b] = benchmarkImpacts(curve, { kind: "parallel-down", sizeBps: 10 }, ["10Y"]);
    expect(b.pctChange).toBeGreaterThan(0);
    expect(b.approxPctChange).toBeCloseTo(b.pctChange, 5);
  });

  it("a twist rallies the front end and sells off the long end", () => {
    const out = benchmarkImpacts(curve, { kind: "twist", sizeBps: 50 }, ["2Y", "30Y"]);
    expect(out[0].pctChange).toBeGreaterThan(0); // front end rallies
    expect(out[1].pctChange).toBeLessThan(0); // long end sells off
  });

  it("skips tenors missing from the curve", () => {
    expect(benchmarkImpacts(curve, { kind: "parallel-up", sizeBps: 25 }, ["5Y", "10Y"]).map((b) => b.label)).toEqual(["10Y"]);
  });
});
