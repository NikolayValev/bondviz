import { describe, it, expect } from "vitest";
import { bootstrapZeros } from "@/lib/curve";

describe("bootstrapZeros", () => {
  it("returns empty arrays for empty input", () => {
    const r = bootstrapZeros([]);
    expect(r.grid).toEqual([]);
    expect(r.zero).toEqual([]);
    expect(r.forward).toEqual([]);
    expect(r.df).toEqual([]);
  });

  it("recovers the annual-compounded zero for a flat par curve", () => {
    // A flat 5% semiannual par curve ⇒ flat 2.5%/period spot ⇒
    // annual-compounded zero = (1.025)^2 - 1 = 0.050625 at every node.
    const par = [
      { years: 1, yieldPct: 5 },
      { years: 2, yieldPct: 5 },
      { years: 5, yieldPct: 5 },
      { years: 10, yieldPct: 5 },
    ];
    const r = bootstrapZeros(par);
    expect(r.grid[0]).toBeCloseTo(0.5, 9);
    for (const z of r.zero) expect(z).toBeCloseTo(0.050625, 6);
    for (const f of r.forward) expect(f).toBeCloseTo(0.050625, 6);
  });

  it("places zeros above par at the long end for an upward-sloping curve", () => {
    const par = [
      { years: 1, yieldPct: 1 },
      { years: 2, yieldPct: 2 },
      { years: 5, yieldPct: 3.5 },
      { years: 10, yieldPct: 4.5 },
      { years: 30, yieldPct: 5 },
    ];
    const r = bootstrapZeros(par);
    const last = r.zero[r.zero.length - 1];
    expect(last).toBeGreaterThan(0.05); // spot > par(30Y)=5% when par is rising
    expect(r.df.every((d) => d > 0 && d <= 1)).toBe(true);
  });
});
