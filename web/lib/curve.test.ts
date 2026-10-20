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

  describe("interpolation", () => {
    // A shape like late-2023: humped long end peaking at 20Y.
    const humped = [
      [1 / 12, 4.3], [0.25, 4.26], [0.5, 4.2], [1, 4.48], [2, 4.79], [3, 4.85],
      [5, 5.0], [7, 5.1], [10, 5.21], [20, 5.6], [30, 5.56],
    ].map(([years, yieldPct]) => ({ years, yieldPct }));

    /** Largest move between neighboring 6-month forwards past 5Y, in basis points. */
    const longEndStepBps = (r: ReturnType<typeof bootstrapZeros>) =>
      Math.max(
        ...r.forward
          .slice(1)
          .map((f, i) => (r.grid[i + 1] > 5 ? Math.abs(f - r.forward[i]) * 1e4 : 0)),
      );

    it("prices a par bond at par at every quoted on-grid tenor, either way", () => {
      for (const interpolation of ["linear", "pchip"] as const) {
        const r = bootstrapZeros(humped, { interpolation });
        for (const { years, yieldPct } of humped.filter((p) => p.years >= 0.5 && Number.isInteger(p.years * 2))) {
          const n = Math.round(years * 2);
          const c = yieldPct / 100 / 2;
          const price = c * r.df.slice(0, n).reduce((a, b) => a + b, 0) + r.df[n - 1];
          expect(price).toBeCloseTo(1, 10);
        }
      }
    });

    it("removes the forward spike that linear par interpolation puts at the 20Y node", () => {
      const linear = bootstrapZeros(humped);
      const pchip = bootstrapZeros(humped, { interpolation: "pchip" });
      expect(longEndStepBps(linear)).toBeGreaterThan(100);
      expect(longEndStepBps(pchip)).toBeLessThan(30);
    });

    it("leaves a flat curve flat", () => {
      const flat = [1, 2, 5, 10].map((years) => ({ years, yieldPct: 5 }));
      for (const f of bootstrapZeros(flat, { interpolation: "pchip" }).forward) expect(f).toBeCloseTo(0.050625, 6);
    });
  });
});
