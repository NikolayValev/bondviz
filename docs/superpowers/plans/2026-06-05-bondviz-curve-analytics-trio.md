# BondViz Curve Analytics Trio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the par→zero/forward bootstrap, the time×tenor yield heatmap, and yield-curve PCA from the Streamlit app to the Next.js web app (`web/`), reusing the existing Treasury data pipeline.

**Architecture:** Pure math goes in `web/lib/` (TDD'd with Vitest); rendering uses React-owned SVG matching the existing `LineChart` pattern. Bootstrap + Heatmap become new sections on the existing Yield Curve page (reusing the rows that page already loads); PCA gets its own `/pca` page with a lookback selector. No new runtime dependencies — PCA's eigensolver (Jacobi) and the heatmap color ramp are hand-rolled.

**Tech Stack:** Next.js (App Router) + React + TypeScript + Tailwind, D3 (`d3-scale`/`d3-shape`/`d3-array`) for scales only, Vitest + Testing Library for tests. Commands run from `web/`. The `@/*` alias maps to `web/`.

> **Next.js version note:** `web/AGENTS.md` warns this Next.js differs from training-data knowledge. Do NOT invent App Router APIs. The new PCA page mirrors the existing, working `web/app/pricing/page.tsx` (server wrapper) + `web/app/pricing/PricingClient.tsx` (`"use client"`) pattern exactly — copy that structure rather than recalling it.

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `web/lib/curve.ts` | Create | `bootstrapZeros` (par→DF→zero) + `forwardRates` via `BootstrapResult` |
| `web/lib/curve.test.ts` | Create | Bootstrap math tests |
| `web/components/charts/Heatmap.tsx` | Create | SVG time×tenor heatmap + exported `colorRamp` |
| `web/components/charts/Heatmap.test.tsx` | Create | Color-ramp + render tests |
| `web/lib/pca.ts` | Create | `standardize`, `covarianceMatrix`, `jacobiEigen`, `pca` |
| `web/lib/pca.test.ts` | Create | Jacobi + PCA tests |
| `web/app/yield-curve/YieldCurveClient.tsx` | Modify | Add Bootstrap + Heatmap sections |
| `web/app/pca/page.tsx` | Create | Server wrapper + metadata |
| `web/app/pca/PcaClient.tsx` | Create | Lookback selector, fetch, render |
| `web/components/Nav.tsx` | Modify | Add "PCA" nav item |

---

## Task 1: Bootstrap math library (TDD)

**Files:**
- Create: `web/lib/curve.ts`
- Test: `web/lib/curve.test.ts`

- [ ] **Step 1: Write the failing test**

Create `web/lib/curve.test.ts`:
```ts
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run lib/curve.test.ts`
Expected: FAIL — cannot resolve `@/lib/curve`.

- [ ] **Step 3: Implement `curve.ts`**

Create `web/lib/curve.ts`:
```ts
// Par → zero/forward bootstrap. Mirrors src/bondviz/visualizer.py
// (bootstrap_zeros_from_par): par yields are SEMIANNUAL par coupons; the
// resulting zero rates are ANNUAL-COMPOUNDED. Do not conflate the two
// conventions. The grid is regular (0.5Y steps), so discount factors are
// solved in order and every prior factor already exists (no interpolation
// of missing nodes is needed).

export interface ParPoint {
  years: number;
  yieldPct: number; // par yield in percent
}

export interface BootstrapResult {
  grid: number[];    // semiannual grid times (0.5, 1.0, …, maxT)
  df: number[];      // discount factor at each grid time
  zero: number[];    // annual-compounded zero rate (decimal) at each grid time
  forward: number[]; // 6-month implied forward (annual-compounded, decimal)
}

/** Linear interpolation of y at x given sorted (xs, ys), flat-extrapolated. */
function interp(x: number, xs: number[], ys: number[]): number {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) {
      const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + (ys[i] - ys[i - 1]) * t;
    }
  }
  return ys[ys.length - 1];
}

export function bootstrapZeros(par: ParPoint[]): BootstrapResult {
  const empty: BootstrapResult = { grid: [], df: [], zero: [], forward: [] };
  const clean = par
    .filter((p) => Number.isFinite(p.years) && Number.isFinite(p.yieldPct))
    .sort((a, b) => a.years - b.years);
  if (clean.length === 0) return empty;

  const knownT = clean.map((p) => p.years);
  const knownR = clean.map((p) => p.yieldPct / 100); // to decimals
  const maxT = knownT[knownT.length - 1];

  const freq = 2;
  const grid: number[] = [];
  for (let t = 0.5; t <= maxT + 1e-9; t += 0.5) grid.push(Number(t.toFixed(10)));
  if (grid.length === 0) return empty;

  const rGrid = grid.map((t) => interp(t, knownT, knownR));

  const df: number[] = [];
  for (let i = 0; i < grid.length; i++) {
    const n = i + 1; // number of semiannual periods to this node
    const c = rGrid[i] / freq;
    if (n === 1) {
      df.push(1 / (1 + c));
    } else {
      let s = 0;
      for (let k = 0; k < n - 1; k++) s += df[k];
      df.push(Math.max(1e-9, (1 - c * s) / (1 + c)));
    }
  }

  const zero = grid.map((T, i) => Math.pow(df[i], -1 / T) - 1);
  const forward = grid.map((_, i) => {
    const dPrev = i === 0 ? 1 : df[i - 1];
    return Math.pow(dPrev / df[i], 1 / 0.5) - 1; // annualize the 0.5Y period
  });

  return { grid, df, zero, forward };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run lib/curve.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add web/lib/curve.ts web/lib/curve.test.ts
git commit -m "feat(web): par->zero/forward bootstrap lib with tests"
```

---

## Task 2: Wire the Bootstrap section into the Yield Curve page

**Files:**
- Modify: `web/app/yield-curve/YieldCurveClient.tsx`

- [ ] **Step 1: Add the import**

In `web/app/yield-curve/YieldCurveClient.tsx`, add after the existing `@/lib/finance` import line:
```ts
import { bootstrapZeros } from "@/lib/curve";
```

- [ ] **Step 2: Compute the bootstrap series in the `view` memo**

In the `view = useMemo(...)` block, immediately after the line `const latestCurve = rowToCurve(latest);`, insert:
```ts
    const boot = bootstrapZeros(latestCurve.map((p) => ({ years: p.years, yieldPct: p.yield })));
    const bootstrapSeries: Series[] = [
      { id: "par", label: "Par", color: "#00d68f", points: latestCurve.map((p) => [p.years, p.yield]) },
      { id: "zero", label: "Zero", color: "#5b8def", points: boot.grid.map((t, i) => [t, boot.zero[i] * 100]) },
      { id: "fwd", label: "Forward (6M)", color: "#f5a623", points: boot.grid.map((t, i) => [t, boot.forward[i] * 100]) },
    ];
```

- [ ] **Step 3: Return the new series from the memo**

Change the memo's return statement from:
```ts
    return { latest, latestCurve, curveSeries, spreadSeriesData };
```
to:
```ts
    return { latest, latestCurve, curveSeries, spreadSeriesData, bootstrapSeries };
```

- [ ] **Step 4: Add the Bootstrap card to the JSX**

In the returned JSX, insert this `<Card>` immediately after the "Curve shifts vs the past" card (i.e., after its closing `</Card>` and before the "Key spreads over time" card):
```tsx
      <Card>
        <h2 className="mb-2 text-lg">Zero & forward curve (bootstrapped)</h2>
        <LineChart
          ariaLabel="Bootstrapped zero and forward curves vs par"
          series={view.bootstrapSeries}
          xLabel="Maturity (years)"
          yLabel="Rate (%)"
          yUnit="%"
        />
        <p className="mt-2 text-sm text-[var(--muted)]">
          Par yields are treated as semiannual par coupons and bootstrapped into discount factors;
          zero rates are annual-compounded, and forwards are the implied 6-month rates.
        </p>
      </Card>
```

- [ ] **Step 5: Verify the build**

Run: `npm run build`
Expected: builds with no type errors; route list still shows `/yield-curve`.

- [ ] **Step 6: Commit**

```bash
git add web/app/yield-curve/YieldCurveClient.tsx
git commit -m "feat(web): add bootstrapped zero/forward curve to yield curve page"
```

---

## Task 3: Heatmap component + color ramp (TDD)

**Files:**
- Create: `web/components/charts/Heatmap.tsx`
- Test: `web/components/charts/Heatmap.test.tsx`

- [ ] **Step 1: Write the failing test**

Create `web/components/charts/Heatmap.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { Heatmap, colorRamp } from "@/components/charts/Heatmap";

describe("colorRamp", () => {
  it("maps 0 and 1 to the ramp endpoints", () => {
    expect(colorRamp(0)).toBe("rgb(13, 27, 42)");
    expect(colorRamp(1)).toBe("rgb(0, 214, 143)");
  });
  it("clamps out-of-range input", () => {
    expect(colorRamp(-5)).toBe("rgb(13, 27, 42)");
    expect(colorRamp(5)).toBe("rgb(0, 214, 143)");
  });
});

describe("Heatmap", () => {
  it("renders one cell per (date, tenor) when given explicit dimensions", () => {
    const { container } = render(
      <Heatmap
        ariaLabel="test heatmap"
        width={300}
        height={120}
        dates={["2025-01-02", "2025-01-03"]}
        tenors={["3M", "2Y", "10Y"]}
        values={[
          [5.0, 4.0, 4.5],
          [5.1, 4.1, 4.6],
        ]}
      />,
    );
    expect(container.querySelectorAll("rect.heatmap-cell")).toHaveLength(6);
    expect(container.querySelector("svg")?.getAttribute("aria-label")).toBe("test heatmap");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run components/charts/Heatmap.test.tsx`
Expected: FAIL — cannot resolve `@/components/charts/Heatmap`.

- [ ] **Step 3: Implement `Heatmap.tsx`**

Create `web/components/charts/Heatmap.tsx`:
```tsx
"use client";
import { useMemo } from "react";
import { useResizeObserver } from "@/components/charts/useResizeObserver";

// Three-stop ramp: deep navy (low) → teal → accent green (high). Hand-rolled
// to avoid pulling in d3-scale-chromatic.
const STOPS: [number, number, number][] = [
  [13, 27, 42],
  [27, 94, 110],
  [0, 214, 143],
];

export function colorRamp(t: number): string {
  const tc = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  const seg = tc * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(seg));
  const f = seg - i;
  const [r0, g0, b0] = STOPS[i];
  const [r1, g1, b1] = STOPS[i + 1];
  const r = Math.round(r0 + (r1 - r0) * f);
  const g = Math.round(g0 + (g1 - g0) * f);
  const b = Math.round(b0 + (b1 - b0) * f);
  return `rgb(${r}, ${g}, ${b})`;
}

export interface HeatmapProps {
  dates: string[];               // rows, old → new
  tenors: string[];              // columns
  values: (number | null)[][];   // [dateIndex][tenorIndex]
  ariaLabel: string;
  width?: number;
  height?: number;
}

const M = { top: 8, right: 12, bottom: 28, left: 12 };

export function Heatmap(props: HeatmapProps) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = props.width ?? measured;
  const height = props.height ?? 260;

  const content = useMemo(() => {
    if (width === 0 || props.dates.length === 0 || props.tenors.length === 0) return null;
    const iw = Math.max(1, width - M.left - M.right);
    const ih = Math.max(1, height - M.top - M.bottom);
    const cellW = iw / props.tenors.length;
    const cellH = ih / props.dates.length;

    let min = Infinity;
    let max = -Infinity;
    for (const row of props.values) {
      for (const v of row) {
        if (v !== null && Number.isFinite(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
    }
    const span = max - min || 1;
    return { iw, ih, cellW, cellH, min, max, span };
  }, [props.dates, props.tenors, props.values, width, height]);

  return (
    <div ref={ref} className="w-full">
      {content && (
        <svg width={width} height={height} role="img" aria-label={props.ariaLabel}>
          <g transform={`translate(${M.left},${M.top})`}>
            {props.values.map((row, di) =>
              row.map((v, ti) => (
                <rect
                  key={`${di}-${ti}`}
                  className="heatmap-cell"
                  x={ti * content.cellW}
                  y={di * content.cellH}
                  width={content.cellW + 0.5}
                  height={content.cellH + 0.5}
                  fill={v === null || !Number.isFinite(v) ? "var(--grid)" : colorRamp((v - content.min) / content.span)}
                />
              )),
            )}
            {props.tenors.map((label, ti) => (
              <text
                key={label}
                x={ti * content.cellW + content.cellW / 2}
                y={content.ih + 18}
                textAnchor="middle"
                fontSize={11}
                fill="var(--muted)"
                className="tabnum"
              >
                {label}
              </text>
            ))}
          </g>
        </svg>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run components/charts/Heatmap.test.tsx`
Expected: PASS (4 assertions across 3 tests).

- [ ] **Step 5: Commit**

```bash
git add web/components/charts/Heatmap.tsx web/components/charts/Heatmap.test.tsx
git commit -m "feat(web): SVG time x tenor heatmap with hand-rolled color ramp"
```

---

## Task 4: Wire the Heatmap section into the Yield Curve page

**Files:**
- Modify: `web/app/yield-curve/YieldCurveClient.tsx`

- [ ] **Step 1: Add the import**

Add after the `@/lib/curve` import:
```ts
import { Heatmap } from "@/components/charts/Heatmap";
```

- [ ] **Step 2: Build the heatmap matrix in the `view` memo**

In the `view` memo, immediately after the `bootstrapSeries` block from Task 2, insert:
```ts
    const tenorsPresent = latestCurve.map((p) => p.label);
    const heatmap = {
      dates: rows.map((r) => r.date),
      tenors: tenorsPresent,
      values: rows.map((r) => {
        const m = new Map(rowToCurve(r).map((p) => [p.label, p.yield] as const));
        return tenorsPresent.map((lab) => m.get(lab) ?? null);
      }),
    };
```

- [ ] **Step 3: Return the heatmap from the memo**

Change the memo's return statement to also include `heatmap`:
```ts
    return { latest, latestCurve, curveSeries, spreadSeriesData, bootstrapSeries, heatmap };
```

- [ ] **Step 4: Add the Heatmap card to the JSX**

Insert this `<Card>` at the end of the returned JSX, immediately before the final closing `</div>`:
```tsx
      <Card>
        <h2 className="mb-2 text-lg">Yield heatmap (time × tenor)</h2>
        <Heatmap
          ariaLabel="Treasury yields by tenor over time"
          dates={view.heatmap.dates}
          tenors={view.heatmap.tenors}
          values={view.heatmap.values}
        />
        <p className="mt-2 text-sm text-[var(--muted)]">
          Each row is a date (old → new), each column a tenor; brighter cells are higher yields.
          Watch horizontal bands for tenor-specific moves and diagonal gradients for rolling
          steepeners or flatteners.
        </p>
      </Card>
```

- [ ] **Step 5: Verify the build**

Run: `npm run build`
Expected: builds with no type errors.

- [ ] **Step 6: Commit**

```bash
git add web/app/yield-curve/YieldCurveClient.tsx
git commit -m "feat(web): add yield heatmap to yield curve page"
```

---

## Task 5: PCA math library (TDD)

**Files:**
- Create: `web/lib/pca.ts`
- Test: `web/lib/pca.test.ts`

- [ ] **Step 1: Write the failing test**

Create `web/lib/pca.test.ts`:
```ts
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run lib/pca.test.ts`
Expected: FAIL — cannot resolve `@/lib/pca`.

- [ ] **Step 3: Implement `pca.ts`**

Create `web/lib/pca.ts`:
```ts
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

  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-18) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const phi = 0.5 * Math.atan2(2 * a[p][q], a[p][p] - a[q][q]);
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        // A := Jᵀ A J
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run lib/pca.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add web/lib/pca.ts web/lib/pca.test.ts
git commit -m "feat(web): yield-curve PCA lib (standardize, Jacobi eigensolver) with tests"
```

---

## Task 6: PCA page (server wrapper + client)

**Files:**
- Create: `web/app/pca/page.tsx`
- Create: `web/app/pca/PcaClient.tsx`

- [ ] **Step 1: Create the server page wrapper**

Create `web/app/pca/page.tsx` (mirrors `web/app/pricing/page.tsx`):
```tsx
import { PcaClient } from "./PcaClient";

export const metadata = { title: "PCA · BondViz" };

export default function PcaPage() {
  return <PcaClient />;
}
```

- [ ] **Step 2: Create the client component**

Create `web/app/pca/PcaClient.tsx`:
```tsx
"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { Segmented } from "@/components/ui/Segmented";
import { LineChart, Series } from "@/components/charts/LineChart";
import { pca } from "@/lib/pca";
import { TENOR_YEARS } from "@/lib/finance";
import { YieldRow } from "@/lib/types";

const COMPONENT_COLORS = ["#00d68f", "#5b8def", "#f5a623"];

function iso(d: Date) {
  return d.toISOString().slice(0, 10);
}

export function PcaClient() {
  const [years, setYears] = useState(2);
  const [rows, setRows] = useState<YieldRow[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    setRows(null);
    setError(false);
    const end = new Date();
    const start = new Date();
    start.setFullYear(start.getFullYear() - years);
    start.setDate(start.getDate() - 14);
    fetch(`/api/treasury/range?start=${iso(start)}&end=${iso(end)}`)
      .then((r) => r.json())
      .then((d) => setRows(d.rows ?? []))
      .catch(() => setError(true));
  }, [years]);

  const result = useMemo(() => (rows ? pca(rows, 3) : null), [rows]);

  const charts = useMemo(() => {
    if (!result) return null;
    const loadingSeries: Series[] = result.loadings.map((vec, c) => ({
      id: `pc${c + 1}`,
      label: `PC${c + 1} · ${(result.explained[c] * 100).toFixed(1)}%`,
      color: COMPONENT_COLORS[c % COMPONENT_COLORS.length],
      points: result.tenors.map((t, j) => [TENOR_YEARS[t], vec[j]] as [number, number]),
    }));
    const scoreSeries: Series[] = result.loadings.map((_, c) => ({
      id: `pc${c + 1}-score`,
      label: `PC${c + 1}`,
      color: COMPONENT_COLORS[c % COMPONENT_COLORS.length],
      points: result.scores.map((s) => [new Date(s.date).getTime(), s.values[c]] as [number, number]),
    }));
    return { loadingSeries, scoreSeries };
  }, [result]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl">Yield Curve PCA</h1>
        <Segmented
          ariaLabel="Lookback window"
          options={[
            { label: "1Y", value: 1 },
            { label: "2Y", value: 2 },
            { label: "5Y", value: 5 },
          ]}
          value={years}
          onChange={(v) => setYears(v as number)}
        />
      </div>
      <p className="text-[var(--muted)]">
        Principal-component analysis of standardized daily Treasury yields. The first three
        components typically map to the level, slope, and curvature of the curve.
      </p>

      {error && <p className="text-[var(--muted)]">Treasury data is unavailable right now.</p>}
      {!error && !rows && <p className="text-[var(--muted)]">Loading Treasury data…</p>}
      {!error && rows && !result && (
        <p className="text-[var(--muted)]">Not enough complete data in this window to run PCA.</p>
      )}

      {result && charts && (
        <>
          <Card>
            <h2 className="mb-3 text-lg">Explained variance</h2>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              {result.explained.map((e, c) => (
                <Metric key={c} label={`PC${c + 1}`} value={`${(e * 100).toFixed(1)}%`} tone="accent" />
              ))}
            </div>
            <p className="mt-3 text-sm text-[var(--muted)]">
              Top {result.explained.length} components explain{" "}
              {(result.explained.reduce((a, b) => a + b, 0) * 100).toFixed(1)}% of the standardized
              yield variance.
            </p>
          </Card>

          <Card>
            <h2 className="mb-2 text-lg">Factor loadings</h2>
            <LineChart
              ariaLabel="PCA factor loadings by maturity"
              series={charts.loadingSeries}
              xLabel="Maturity (years)"
              yLabel="Loading"
              zeroBaseline
            />
            <p className="mt-2 text-sm text-[var(--muted)]">
              A roughly flat PC1 is a level shift; a monotonic PC2 is slope; a U-shaped PC3 is
              curvature.
            </p>
          </Card>

          <Card>
            <h2 className="mb-2 text-lg">Factor scores over time</h2>
            <LineChart
              ariaLabel="PCA factor scores over time"
              series={charts.scoreSeries}
              xType="time"
              xLabel="Date"
              yLabel="Score"
              zeroBaseline
            />
          </Card>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Verify the build**

Run: `npm run build`
Expected: builds with no type errors; route list now includes `/pca`.

- [ ] **Step 4: Commit**

```bash
git add web/app/pca
git commit -m "feat(web): PCA page with lookback selector, loadings & scores charts"
```

---

## Task 7: Add PCA to the navigation

**Files:**
- Modify: `web/components/Nav.tsx`

- [ ] **Step 1: Add the nav link**

In `web/components/Nav.tsx`, change the `LINKS` array from:
```tsx
const LINKS = [
  { href: "/yield-curve", label: "Yield Curve" },
  { href: "/pricing", label: "Bond Pricing" },
  { href: "/portfolio", label: "Portfolio" },
];
```
to:
```tsx
const LINKS = [
  { href: "/yield-curve", label: "Yield Curve" },
  { href: "/pricing", label: "Bond Pricing" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/pca", label: "PCA" },
];
```

- [ ] **Step 2: Verify the build**

Run: `npm run build`
Expected: builds with no errors.

- [ ] **Step 3: Commit**

```bash
git add web/components/Nav.tsx
git commit -m "feat(web): add PCA to navigation"
```

---

## Task 8: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Full test suite**

Run (from `web/`): `npm test`
Expected: all suites pass, including the new `curve`, `pca`, and `Heatmap` tests (previous count was 44; new tests add to it).

- [ ] **Step 2: Lint + production build**

Run: `npm run lint` (expected: no errors) and `npm run build` (expected: succeeds; routes include `/yield-curve`, `/pricing`, `/portfolio`, `/pca`).

- [ ] **Step 3: Manual smoke (live data)**

Run: `npm run dev`, open http://localhost:3000 and confirm:
- `/yield-curve` now shows the **bootstrapped zero/forward** card (three lines: par, zero, forward) and the **yield heatmap** card (grid of colored cells with tenor labels).
- `/pca` loads, the **1Y/2Y/5Y** selector re-fetches and updates the charts, explained-variance metrics sum to ≤100%, loadings render vs maturity, and factor scores render over time.
- "PCA" appears in the nav and is highlighted when active.
- All charts are dark-themed and resize with the window.

- [ ] **Step 4: Final commit (only if tweaks were needed)**

```bash
git add -A
git commit -m "chore(web): curve analytics trio verification fixes"
```

---

## Self-Review Notes

- **Spec coverage:** Bootstrap math + page section (Tasks 1–2); Heatmap component + page section (Tasks 3–4); PCA lib with hand-rolled Jacobi + standardize/covariance (Task 5); PCA page with 1Y/2Y/5Y selector, explained variance, loadings, and scores (Task 6); nav item (Task 7); verification (Task 8). No new runtime dependencies. Stocks/FRED remain out of scope per the spec.
- **Type consistency:** `BootstrapResult` (`grid`/`df`/`zero`/`forward`) from Task 1 is consumed unchanged in Task 2. `Heatmap`/`colorRamp` props (Task 3) match the call site (Task 4). `PcaResult` (`tenors`/`explained`/`loadings`/`scores`) and `pca(rows, k)`/`jacobiEigen` from Task 5 are used with identical signatures in Task 6. `Series` is the existing `LineChart` type. `Segmented`, `Metric`, `Card`, `TENOR_YEARS`, `BC_TO_TENOR`, `rowToCurve`, `TENOR_LABELS` are all existing exports used as defined.
- **Parity:** bootstrap mirrors `bootstrap_zeros_from_par` (semiannual par → annual-compounded zeros); PCA mirrors `pca_view.py` (standardize with ddof=0, then eigendecompose the resulting correlation matrix).
```
