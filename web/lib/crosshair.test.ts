import { describe, it, expect } from "vitest";
import { nearestIndex, readoutDecimals, valueAt } from "@/lib/crosshair";

describe("nearestIndex", () => {
  const xs = [0.25, 1, 2, 5, 10, 30];
  it("snaps to the closest value", () => {
    expect(nearestIndex(xs, 0)).toBe(0);
    expect(nearestIndex(xs, 1.4)).toBe(1);
    expect(nearestIndex(xs, 1.6)).toBe(2);
    expect(nearestIndex(xs, 21)).toBe(5);
    expect(nearestIndex(xs, 99)).toBe(5);
  });
  it("breaks ties to the left and handles tiny inputs", () => {
    expect(nearestIndex(xs, 1.5)).toBe(1);
    expect(nearestIndex([7], 3)).toBe(0);
    expect(nearestIndex([], 3)).toBe(-1);
  });
});

describe("valueAt", () => {
  const pts: [number, number][] = [[1, 4], [2, 5], [5, 4.4]];
  it("returns exact points", () => {
    expect(valueAt(pts, 1)).toBe(4);
    expect(valueAt(pts, 2)).toBe(5);
    expect(valueAt(pts, 5)).toBe(4.4);
  });
  it("interpolates along the drawn segment", () => {
    expect(valueAt(pts, 1.5)).toBeCloseTo(4.5, 12);
    expect(valueAt(pts, 3.5)).toBeCloseTo(4.7, 12);
  });
  it("is null outside the series", () => {
    expect(valueAt(pts, 0.5)).toBeNull();
    expect(valueAt(pts, 6)).toBeNull();
    expect(valueAt([], 1)).toBeNull();
  });
});

describe("readoutDecimals", () => {
  it("scales precision to the data's span", () => {
    expect(readoutDecimals(0.3)).toBe(3); // PCA loadings
    expect(readoutDecimals(2)).toBe(2); // yields in %
    expect(readoutDecimals(60)).toBe(2); // a stock's closes
    expect(readoutDecimals(400)).toBe(1); // spreads in bps
    expect(readoutDecimals(5000)).toBe(0);
  });
});
