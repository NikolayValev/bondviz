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

  it("produces orthonormal eigenvectors in descending order for a 4x4 symmetric matrix", () => {
    const A = [
      [4, 1, 2, 0.5],
      [1, 3, 0.7, 1.2],
      [2, 0.7, 5, 0.3],
      [0.5, 1.2, 0.3, 2],
    ];
    const { values, vectors } = jacobiEigen(A);
    const n = 4;
    // descending order
    for (let i = 1; i < n; i++) expect(values[i - 1]).toBeGreaterThanOrEqual(values[i] - 1e-9);
    // orthonormality: vectors[i]·vectors[j] = δij
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let dot = 0;
        for (let m = 0; m < n; m++) dot += vectors[i][m] * vectors[j][m];
        expect(dot).toBeCloseTo(i === j ? 1 : 0, 6);
      }
    }
    // eigenvalue equation: A·v_i = λ_i·v_i
    for (let i = 0; i < n; i++) {
      for (let r = 0; r < n; r++) {
        let av = 0;
        for (let m = 0; m < n; m++) av += A[r][m] * vectors[i][m];
        expect(av).toBeCloseTo(values[i] * vectors[i][r], 6);
      }
    }
    // trace is preserved (sum of eigenvalues = sum of diagonal)
    const trace = A[0][0] + A[1][1] + A[2][2] + A[3][3];
    expect(values.reduce((a, b) => a + b, 0)).toBeCloseTo(trace, 6);
  });

  it("reconstructs A = V Λ Vᵀ for a 5x5 symmetric matrix", () => {
    // Build a random-but-deterministic symmetric 5x5.
    const n = 5;
    const A: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    let seed = 1;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return (seed / 0x7fffffff) * 2 - 1; };
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) { const x = rnd(); A[i][j] = x; A[j][i] = x; }
    const { values, vectors } = jacobiEigen(A);
    for (let r = 0; r < n; r++) {
      for (let cc = 0; cc < n; cc++) {
        let recon = 0;
        for (let i = 0; i < n; i++) recon += values[i] * vectors[i][r] * vectors[i][cc];
        expect(recon).toBeCloseTo(A[r][cc], 6);
      }
    }
  });

  it("handles a near-singular (rank-deficient) symmetric matrix", () => {
    // outer product u·uᵀ has one nonzero eigenvalue = |u|² and the rest ~0.
    const u = [1, 2, 3, 4];
    const n = 4;
    const A = u.map((ui) => u.map((uj) => ui * uj));
    const { values } = jacobiEigen(A);
    const norm2 = u.reduce((a, b) => a + b * b, 0);
    expect(values[0]).toBeCloseTo(norm2, 6);
    for (let i = 1; i < n; i++) expect(Math.abs(values[i])).toBeLessThan(1e-6);
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
