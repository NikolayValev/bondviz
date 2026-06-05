import { describe, it, expect } from "vitest";
import { jacobiEigen, pca } from "@/lib/pca";
import { YieldRow } from "@/lib/types";

describe("jacobiEigen", () => {
  it("diagonalizes a diagonal matrix (sorted descending)", () => {
    const { values } = jacobiEigen([
      [2, 0],
      [0, 3],
    ]);
    expect(values[0]).toBeCloseTo(3, 9);
    expect(values[1]).toBeCloseTo(2, 9);
  });

  it("finds eigenpairs of [[2,1],[1,2]]", () => {
    const { values, vectors } = jacobiEigen([
      [2, 1],
      [1, 2],
    ]);
    expect(values[0]).toBeCloseTo(3, 9);
    expect(values[1]).toBeCloseTo(1, 9);
    // eigenvector for λ=3 is ±(1,1)/√2 (same-sign components)
    expect(Math.abs(vectors[0][0])).toBeCloseTo(Math.SQRT1_2, 6);
    expect(vectors[0][0] * vectors[0][1]).toBeGreaterThan(0);
    // eigenvector for λ=1 is ±(1,-1)/√2 (opposite-sign components)
    expect(vectors[1][0] * vectors[1][1]).toBeLessThan(0);
  });
});

describe("pca", () => {
  it("returns null when there is not enough data", () => {
    expect(pca([{ date: "2025-01-02", BC_10YEAR: 4.5 }], 3)).toBeNull();
  });

  it("captures a single dominant direction in PC1", () => {
    // All tenors move together (a level factor) ⇒ PC1 explains ~all variance.
    const rows: YieldRow[] = [];
    for (let i = 0; i < 30; i++) {
      const base = 3 + Math.sin(i / 3); // common driver
      rows.push({
        date: `2025-02-${String((i % 28) + 1).padStart(2, "0")}`,
        BC_2YEAR: base + 0.01 * Math.cos(i),
        BC_5YEAR: base + 0.2,
        BC_10YEAR: base + 0.4,
        BC_30YEAR: base + 0.6,
      });
    }
    const result = pca(rows, 3);
    expect(result).not.toBeNull();
    if (!result) return;
    const total = result.explained.reduce((a, b) => a + b, 0);
    expect(total).toBeLessThanOrEqual(1.0000001);
    expect(result.explained[0]).toBeGreaterThan(0.9);
    expect(result.scores).toHaveLength(rows.length);
    expect(result.loadings[0]).toHaveLength(result.tenors.length);
  });
});
