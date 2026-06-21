# Carry & Roll-Down Analyzer — Design

**Date:** 2026-06-21
**Status:** Approved (pending spec review)
**Surface:** `web/` (Next.js 16 terminal)

## Goal

Add a new `/carry` page that answers the rates-desk question: *if I buy each point
on the Treasury curve and hold it for a fixed horizon, what return do I earn from
**carry** and **roll-down**, and how far can yields rise before I break even?*

This is a new analytics feature, complementing the existing Yield Curve, Pricing,
PCA, Stocks, and Portfolio pages. It follows the app's one-feature-per-page pattern.

## Concepts

For each maturity tenor `T` on today's par curve, and a holding horizon `h` (in years):

- **Carry** — running yield earned over the horizon: `carry = y(T) · h`. Income you
  accrue just for holding. (Running-yield convention, chosen for a clean additive
  decomposition. The forward-based alternative was rejected because forwards already
  embed roll-down, which would make carry and roll-down overlap.)
- **Roll-down** — price gain as the bond ages down a *static* curve. After horizon `h`
  the bond is a `T−h` bond yielding the interpolated `y(T−h)`. Return ≈
  `Dur(T−h) · (y(T) − y(T−h))`, where `Dur` is modified duration. Positive on an
  upward-sloping curve, negative when inverted.
- **Total** — `carry + rolldown`, shown in bps of return and as a percent.
- **Breakeven** — `total / Dur(T−h)`: how many bps yields can sell off over the
  horizon before the position returns zero. The headline trader number.

### Conventions (made explicit in code + a UI footnote)

- Par yields are used directly as the maturity-point yield (`y(T)`).
- **Static curve** assumption: roll-down only, no rate forecast.
- Modified duration is computed for the *rolled* bond (maturity `T−h`), priced as a
  par bond (coupon = its own par yield) under continuous compounding, reusing
  `modifiedDuration` from `lib/finance.ts`.
- `y(T−h)` comes from linear interpolation on today's curve (same flat-extrapolated
  interp logic already used in `lib/curve.ts`).
- All yields handled in decimals internally; bps/percent only at the display edge.

## Architecture

### `web/lib/carry.ts` (new, pure + unit-tested)

```ts
export interface CarryPoint {
  label: string;       // tenor label, e.g. "5Y"
  years: number;       // T
  yieldPct: number;    // y(T) in percent
  carryBps: number;    // carry return over horizon, in bps
  rollBps: number;     // roll-down return over horizon, in bps
  totalBps: number;    // carryBps + rollBps
  breakevenBps: number;// yield sell-off (bps) that zeroes the horizon return
  totalPct: number;    // total return over horizon, percent
}

export function carryRollDown(
  curve: { years: number; yieldPct: number; label: string }[],
  horizonYears: number,
): CarryPoint[];
```

- Filters to finite points, sorts by `years`.
- For each point with `T > h` (a bond shorter than the horizon can't roll — skip),
  computes the fields above.
- Duration uses a par bond: `bondCashflows(face=100, coupon=y(T−h), years=T−h)`
  then `modifiedDuration(...)` at yield `y(T−h)`.

### `web/app/carry/page.tsx` + `web/app/carry/CarryClient.tsx`

- Client component fetches the latest curve via the existing
  `/api/treasury/range` route (same call the Yield Curve page makes), takes the
  latest row, `rowToCurve`.
- Horizon segmented control (1M / 3M / 6M / 1Y) reusing `components/ui/Segmented`.
- Recomputes `carryRollDown` on horizon change via `useMemo`.
- Fail-soft: loading + unavailable states mirroring `YieldCurveClient`.

### `web/components/Nav.tsx`

- Add a "Carry & Roll" entry in the Fixed Income group.

### Home page card (optional, low cost)

- Add a card linking to `/carry` alongside Yield Curve and Pricing.

## UI

1. **Carry vs roll-down (stacked bars per tenor)** — carry and roll-down stacked so
   total height = total return; negative roll-down renders below the axis.
2. **Breakeven cushion (bars per tenor)** — bps of yield sell-off each tenor can
   absorb; color tone by sign.
3. **Table** — tenor, yield, carry (bps), roll (bps), total (bps), breakeven (bps),
   total return (%).
4. **Auto-summary line** — e.g. "Best breakeven cushion: 5Y at +38 bps over 3M",
   in the spirit of `describeCurve`.

Charting: reuse the existing D3 chart components where they fit; a small stacked-bar
renderer may be added if no current component covers stacked bars (decision deferred
to the implementation plan after auditing `components/charts`).

## Error handling

- Empty/short curve → render an "unavailable" message, never throw.
- Tenors with `T ≤ h` are skipped (documented in the footnote).
- `Dur ≈ 0` guard to avoid divide-by-zero in breakeven (clamp / skip).

## Testing (`web/lib/carry.test.ts`, vitest)

- Upward-sloping curve → roll-down positive.
- Inverted curve → roll-down negative.
- Flat curve → roll-down ≈ 0; total ≈ carry.
- Carry scales linearly with horizon.
- Breakeven sign matches total sign; `breakeven · Dur ≈ total`.
- Points with `T ≤ h` are excluded.
- Empty input → empty output (no throw).

## Out of scope (YAGNI)

- Levered/repo-financed carry, real-yield/TIPS carry, custom user bonds,
  forward-based carry, multi-currency. Static-curve running-yield only.
```

## Note for implementation

`web/AGENTS.md`: this is Next.js 16 with breaking changes — read the relevant guide
in `node_modules/next/dist/docs/` before writing page/route code.
