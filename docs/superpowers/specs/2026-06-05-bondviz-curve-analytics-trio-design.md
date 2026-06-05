# BondViz Curve Analytics Trio — Design Spec

**Date:** 2026-06-05
**Status:** Approved (design); pending implementation plan.

## Goal

Port three curve-analytics features from the original Streamlit app (`src/bondviz/`) to the
Next.js web app (`web/`): the par→zero/forward **bootstrap**, the yield **heatmap**, and
yield-curve **PCA**. All three reuse the existing keyless U.S. Treasury par-yield pipeline
(`/api/treasury/range`, `web/lib/treasury.ts`). This is **Spec A** of a two-part effort to close
the gap between the two front-ends; **Stocks** (Polygon) is deferred to a separate Spec B because
it introduces a new data source and secret/env handling.

## Principles & conventions

- **Pure math lives in `web/lib/`** and is unit-tested with Vitest (TDD).
- **Rendering uses React-owned SVG**, matching the existing `web/components/charts/LineChart.tsx`
  pattern (the component owns the SVG DOM; D3 is used only for scales/shapes/array helpers).
- **No new runtime dependencies.** PCA's eigen-decomposition is hand-rolled (Jacobi); the heatmap
  color ramp is hand-rolled. This matches the repo's existing ethos (hand-rolled D3 charts,
  native C++ pricing).
- The `@/*` import alias maps to `web/`. All commands run from `web/`.
- **Numerical parity:** the TypeScript bootstrap and PCA must stay numerically faithful to the
  Python implementations in `src/bondviz/visualizer.py` (`bootstrap_zeros_from_par`) and
  `src/bondviz/pca_view.py`.

## Page layout & navigation

- **Bootstrap** and **Heatmap** are added as new sections on the **existing Yield Curve page**
  (`web/app/yield-curve/YieldCurveClient.tsx`) — they belong with the curve content and reuse the
  data that page already loads.
- **PCA** becomes its **own page** at `/pca` with a new nav item.
- Nav order becomes: Yield Curve · Bond Pricing · Portfolio · PCA
  (`web/components/Nav.tsx`).

---

## Feature 1 — Bootstrap (par → zero / forward)

**Section on the Yield Curve page.** Operates on the latest curve row the page already loads
(no new fetch).

**Lib:** `web/lib/curve.ts` (+ `curve.test.ts`)

- `bootstrapZeros(parYields: { years: number; yield: number }[])` — mirrors the Python
  `bootstrap_zeros_from_par`:
  1. Convert par yields from percent to decimals; drop missing.
  2. Build a **semiannual** grid (0.5Y steps) up to the longest tenor.
  3. Interpolate par yields onto the grid (linear; flat extrapolation at the ends).
  4. Solve discount factors `D(T)` iteratively: for `n = round(T·2)` semiannual coupons with
     coupon `c = r_par/2`, `D(T) = (1 − c·Σ D(prior)) / (1 + c)`; with linear interpolation of
     any missing intermediate `D` and a floor of `1e-9`.
  5. Convert DFs to **annual-compounded** zero rates: `z(T) = D(T)^(-1/T) − 1`.
- `forwardRates(dfs)` — 6-month implied forwards derived from the discount factors:
  `f = (D(T_i)/D(T_{i+1}))^(1/Δ) − 1` over the semiannual grid (annualized), returned per grid
  point so it can be plotted against maturity.

**Render:** a `LineChart` overlaying three series vs maturity (years):
**Par** (the input curve), **Zero** (annual-compounded), **Forward** (implied 6-month forwards).
Caption notes the semiannual par-coupon convention and the annual-compounding of zeros (the two
compounding conventions must not be conflated — same caveat as the Python code).

**Tests:**
- Flat par curve → zero curve ≈ par curve (within tolerance).
- Upward-sloping par curve → zeros above par at the long end (sanity on monotonic direction).
- Empty / all-missing input → returns empty arrays (no throw).

---

## Feature 2 — Heatmap (time × tenor)

**Section on the Yield Curve page.** Uses the page's existing ~1-year range rows.

**Component:** `web/components/charts/Heatmap.tsx` (+ `Heatmap.test.tsx`)

- Props: `rows` (per-date yields by tenor), the ordered tenor labels, and an `ariaLabel`.
- SVG grid: **x = tenor**, **y = date (old → new)**, each cell a `<rect>` filled by yield level.
- **Hand-rolled color ramp:** a small helper `colorRamp(t: number)` mapping a normalized value
  `t ∈ [0,1]` to a hex color by interpolating between ramp stops (low → high, ending near the
  app accent). No new dependency.
- A simple colorbar legend and a min/max caption (`Range X% to Y%`), matching the Streamlit
  interpretation text about horizontal bands (tenor-specific moves) and diagonal gradients
  (rolling steepeners/flatteners).

**Tests:**
- Render emits exactly `nDates × nTenors` `rect` cells for a small fixture.
- `colorRamp` maps `0` and `1` to the ramp endpoints; midpoint is between them.

---

## Feature 3 — PCA (yield-curve factors)

**New page** `/pca` (`web/app/pca/page.tsx` server wrapper + `PcaClient.tsx`).

**Lib:** `web/lib/pca.ts` (+ `pca.test.ts`)

- `standardize(columns)` — z-score each tenor column using population std (`ddof = 0`), matching
  `(yield_df - mean) / std(ddof=0)` in the Python. Drop rows with any non-finite value after
  standardization.
- `correlationMatrix(standardizedColumns)` — symmetric matrix from the standardized data
  (standardizing then taking covariance ⇒ correlation-matrix PCA, matching sklearn-on-standardized
  input).
- `jacobiEigen(symmetric)` — classic **Jacobi rotation** eigensolver for the symmetric ≤12×12
  matrix; returns eigenvalues and orthonormal eigenvectors. Iterate until off-diagonal mass is
  below tolerance or a max-sweeps cap.
- `pca(rows, tenors, k)` — orchestrates the above: returns the top-`k`
  **explained-variance ratios** (eigenvalue / Σ eigenvalues, sorted desc), the **loadings**
  (eigenvectors, one weight per tenor), and the **factor scores** (standardized data projected
  onto the components) with their dates.

**Page (`PcaClient.tsx`):**
- **Lookback selector: 1Y / 2Y / 5Y** (default **2Y**). On change, fetch `/api/treasury/range`
  for the corresponding window (the range route already fetches across multiple calendar years).
- Renders:
  1. **Explained variance** — list/bars for PC1/PC2/PC3 with percentages and a cumulative total.
  2. **Loadings** `LineChart` — loading weight vs tenor for the top components (the classic
     level / slope / curvature shapes).
  3. **Factor scores over time** `LineChart` — the top components' scores across dates
     (x = time, zero baseline).
- Loading / error / insufficient-data states (consistent with the Yield Curve page's
  loading/error handling).

**Tests:**
- `jacobiEigen` on a known small symmetric matrix → eigenvalues/eigenvectors match expected
  (up to sign/order).
- `pca` on synthetic data dominated by one direction → PC1 captures the bulk of variance and its
  loading aligns with the planted direction.
- Explained-variance ratios are non-negative and sum to ≈ 1.

---

## Files

| File | Action | Responsibility |
| --- | --- | --- |
| `web/lib/curve.ts` (+ `curve.test.ts`) | Create | `bootstrapZeros`, `forwardRates` |
| `web/lib/pca.ts` (+ `pca.test.ts`) | Create | `standardize`, `correlationMatrix`, `jacobiEigen`, `pca` |
| `web/components/charts/Heatmap.tsx` (+ `Heatmap.test.tsx`) | Create | SVG time×tenor heatmap + color ramp |
| `web/app/yield-curve/YieldCurveClient.tsx` | Modify | Add Bootstrap + Heatmap sections |
| `web/app/pca/page.tsx` | Create | Server wrapper + metadata |
| `web/app/pca/PcaClient.tsx` | Create | Lookback selector, PCA fetch + render |
| `web/components/Nav.tsx` | Modify | Add "PCA" nav item |

## Testing approach

- Pure logic (`curve.ts`, `pca.ts`) and the color-ramp/heatmap-cell helpers are TDD'd with
  Vitest. Chart/heatmap components get a render test (SVG element counts). Pages are verified by
  `npm run build` + manual `npm run dev` smoke. No new runtime dependencies are added.

## Out of scope

- **Stocks** (Polygon) — deferred to Spec B (separate data source, API key, env handling).
- FRED fallback, the Streamlit-specific cross-page session state, and any changes to the Python
  app. The Treasury `/api/treasury/range` route is reused unchanged.
