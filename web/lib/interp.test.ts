import { describe, it, expect } from "vitest";
import { interp, makePchip } from "@/lib/interp";

const xs = [0.25, 1, 2, 5, 10, 20, 30];
const ys = [4.26, 4.48, 4.79, 5.0, 5.21, 5.6, 5.56]; // a real par curve (2026-10-13)

describe("interp (linear)", () => {
  it("hits the nodes, interpolates linearly and extrapolates flat", () => {
    expect(interp(2, xs, ys)).toBe(4.79);
    expect(interp(1.5, xs, ys)).toBeCloseTo((4.48 + 4.79) / 2, 12);
    expect(interp(0, xs, ys)).toBe(4.26);
    expect(interp(40, xs, ys)).toBe(5.56);
  });
});

describe("makePchip", () => {
  const f = makePchip(xs, ys);
  const slope = (x: number, e = 1e-6) => (f(x + e) - f(x - e)) / (2 * e);

  it("passes through every node and extrapolates flat", () => {
    xs.forEach((x, i) => expect(f(x)).toBeCloseTo(ys[i], 12));
    expect(f(0)).toBe(4.26);
    expect(f(40)).toBe(5.56);
  });

  it("never overshoots: each segment stays between its endpoints", () => {
    for (let i = 0; i < xs.length - 1; i++) {
      const lo = Math.min(ys[i], ys[i + 1]);
      const hi = Math.max(ys[i], ys[i + 1]);
      for (let t = 0; t <= 1; t += 0.05) {
        const v = f(xs[i] + t * (xs[i + 1] - xs[i]));
        expect(v).toBeGreaterThanOrEqual(lo - 1e-12);
        expect(v).toBeLessThanOrEqual(hi + 1e-12);
      }
    }
  });

  it("has a continuous slope at interior nodes (linear jumps there)", () => {
    for (const x of [1, 2, 5, 10, 20]) {
      expect(Math.abs(slope(x + 1e-4) - slope(x - 1e-4))).toBeLessThan(1e-3);
    }
    const lin = (x: number) => interp(x, xs, ys);
    const linJump = Math.abs((lin(20 + 1e-4) - lin(20)) - (lin(20) - lin(20 - 1e-4))) / 1e-4;
    expect(linJump).toBeGreaterThan(0.04); // the kink PCHIP removes
  });

  it("flattens at a local extremum (20Y peak) instead of overshooting it", () => {
    expect(Math.abs(slope(20))).toBeLessThan(1e-6);
  });

  it("falls back to linear with fewer than three points", () => {
    const g = makePchip([1, 3], [2, 4]);
    expect(g(2)).toBe(3);
  });
});
