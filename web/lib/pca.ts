import { TenorLabel, TENOR_LABELS, YieldRow } from "@/lib/types";
import { BC_TO_TENOR, TENOR_YEARS } from "@/lib/finance";

export interface PcaResult {
  tenors: TenorLabel[];
  explained: number[];                          // variance ratio per component (top k)
  loadings: number[][];                         // [component][tenorIndex]
  scores: { date: string; values: number[] }[]; // [row].values[component]
}

/** Z-score each column with population std (ddof=0). Zero-variance columns
 *  collapse to zeros. Mirrors (yield_df - mean)/std(ddof=0) in pca_view.py. */
export function standardize(matrix: number[][]): number[][] {
  const nRows = matrix.length;
  if (nRows === 0) return [];
  const nCols = matrix[0].length;
  const out = matrix.map((r) => r.slice());
  for (let c = 0; c < nCols; c++) {
    let mean = 0;
    for (let r = 0; r < nRows; r++) mean += matrix[r][c];
    mean /= nRows;
    let varSum = 0;
    for (let r = 0; r < nRows; r++) varSum += (matrix[r][c] - mean) ** 2;
    const std = Math.sqrt(varSum / nRows); // ddof = 0
    for (let r = 0; r < nRows; r++) out[r][c] = std === 0 ? 0 : (matrix[r][c] - mean) / std;
  }
  return out;
}

/** Sample covariance (1/(N-1)) of the columns. With standardized input this is
 *  the correlation matrix. The 1/(N-1) factor cancels in variance ratios. */
export function covarianceMatrix(data: number[][]): number[][] {
  const nRows = data.length;
  const nCols = nRows > 0 ? data[0].length : 0;
  const cov: number[][] = Array.from({ length: nCols }, () => new Array(nCols).fill(0));
  const denom = Math.max(1, nRows - 1);
  for (let i = 0; i < nCols; i++) {
    for (let j = i; j < nCols; j++) {
      let s = 0;
      for (let r = 0; r < nRows; r++) s += data[r][i] * data[r][j];
      const v = s / denom;
      cov[i][j] = v;
      cov[j][i] = v;
    }
  }
  return cov;
}

/** Jacobi eigensolver for a symmetric matrix. Returns eigenvalues and
 *  eigenvectors (vectors[i] is the i-th eigenvector), sorted by descending
 *  eigenvalue. Each eigenvector's largest-magnitude component is made positive
 *  so the sign is deterministic. */
export function jacobiEigen(input: number[][]): { values: number[]; vectors: number[][] } {
  const n = input.length;
  const a = input.map((r) => r.slice());
  const v: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );

  for (let sweep = 0; sweep < 500; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-18) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const phi = 0.5 * Math.atan2(2 * a[p][q], a[p][p] - a[q][q]);
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        // A := Jᵀ A J — update off-(p,q) rows/cols symmetrically, then the 2×2 block exactly
        for (let k = 0; k < n; k++) {
          if (k === p || k === q) continue;
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
          a[p][k] = a[k][p];
          a[q][k] = a[k][q];
        }
        const app = a[p][p], aqq = a[q][q], apq = a[p][q];
        a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        a[p][q] = 0;
        a[q][p] = 0;
        // V := V J
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const pairs = Array.from({ length: n }, (_, j) => ({
    value: a[j][j],
    vector: v.map((row) => row[j]),
  }));
  pairs.sort((x, y) => y.value - x.value);

  for (const p of pairs) {
    let maxIdx = 0;
    for (let i = 1; i < p.vector.length; i++) {
      if (Math.abs(p.vector[i]) > Math.abs(p.vector[maxIdx])) maxIdx = i;
    }
    if (p.vector[maxIdx] < 0) p.vector = p.vector.map((x) => -x);
  }

  return { values: pairs.map((p) => p.value), vectors: pairs.map((p) => p.vector) };
}

/** Run PCA on standardized Treasury yields. Returns the top-k components or
 *  null if there are too few usable tenors/rows. */
export function pca(rows: YieldRow[], k: number): PcaResult | null {
  // Tenors numeric in EVERY row, ordered by maturity.
  const tenors: TenorLabel[] = [];
  const bcByTenor = new Map<TenorLabel, string>();
  for (const [bc, label] of Object.entries(BC_TO_TENOR)) bcByTenor.set(label, bc);
  for (const label of TENOR_LABELS) {
    const bc = bcByTenor.get(label);
    if (!bc) continue;
    if (rows.length > 0 && rows.every((r) => typeof r[bc] === "number" && Number.isFinite(r[bc] as number))) {
      tenors.push(label);
    }
  }

  if (tenors.length < 2 || rows.length < 5) return null;

  const matrix = rows.map((r) => tenors.map((t) => r[bcByTenor.get(t)!] as number));
  const std = standardize(matrix);
  const cov = covarianceMatrix(std);
  const { values, vectors } = jacobiEigen(cov);

  const totalVar = values.reduce((a, b) => a + Math.max(0, b), 0) || 1;
  const kk = Math.min(k, values.length);

  const explained = values.slice(0, kk).map((val) => Math.max(0, val) / totalVar);
  const loadings = vectors.slice(0, kk);
  const scores = std.map((row, ri) => ({
    date: rows[ri].date,
    values: loadings.map((vec) => row.reduce((acc, x, j) => acc + x * vec[j], 0)),
  }));

  return { tenors, explained, loadings, scores };
}

export { TENOR_YEARS };
