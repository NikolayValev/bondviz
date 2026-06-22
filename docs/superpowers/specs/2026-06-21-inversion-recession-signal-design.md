# Inversion & Recession Signal Dashboard — Design

**Date:** 2026-06-21
**Status:** Approved (pending spec review)
**Surface:** `web/` (Next.js 16 terminal)

## Goal

Add a `/signal` page that frames yield-curve inversion as a recession indicator:
current status and streak, a deep-history spread chart with inversion shading and
NBER recession bands, and a table of every inversion episode since 1990 with whether
a recession followed.

Headline signal: **10y–3m** spread (the Fed's most predictive measure). The popular
**2s10s** is shown alongside as a secondary line/readout.

## Architecture

### Data layer

The Treasury feed is fetched one XML file per calendar year (`fetchTreasuryYear`).
Returning 35 years of full yield rows would be a large payload, so add a slim,
spread-only endpoint.

**New route `web/app/api/treasury/spreads/route.ts`:**
- `GET ?start=YYYY-MM-DD&end=YYYY-MM-DD`
- Loops years `y0..y1`, calls `fetchTreasuryYear(y)` (each `.catch(() => [])` so a
  failed year is skipped, not fatal), concatenates, filters to `[start, end]`.
- Maps rows → spread points via `toSpreadPoints` (below) and returns
  `{ points: SpreadPoint[] }`.
- On total failure returns `{ points: [] }` with status 503.
- Inherits hourly caching from `fetchTreasuryYear`'s `next: { revalidate: 3600 }`.
- Mirrors the existing `app/api/treasury/range/route.ts` structure and its
  400-on-missing-params behavior.

### Analysis module `web/lib/signal.ts` (pure, unit-tested)

```ts
export interface SpreadPoint {
  date: string;   // ISO yyyy-mm-dd
  s10y3m: number | null; // 10Y − 3M, percentage points
  s2s10s: number | null; // 10Y − 2Y, percentage points
}

export interface SignalStatus {
  date: string | null;
  s10y3m: number | null;
  s2s10s: number | null;
  inverted: boolean;       // s10y3m <= 0
  streakDays: number;      // consecutive most-recent points with s10y3m <= 0
}

export interface InversionEpisode {
  start: string;           // first date of the inverted run
  end: string;             // last date of the inverted run
  days: number;            // number of points in the run
  maxDepthBps: number;     // most negative s10y3m in the run, in bps (<= 0)
  recessionFollowed: boolean; // an NBER recession began within 24 months of start
}

export interface NberRecession { start: string; end: string; } // ISO month-precision dates

export const NBER_RECESSIONS: NberRecession[]; // 4 entries, 1990–2020 (see below)

export function toSpreadPoints(rows: Record<string, unknown>[]): SpreadPoint[];
export function currentStatus(points: SpreadPoint[]): SignalStatus;
export function inversionEpisodes(points: SpreadPoint[]): InversionEpisode[];
```

**Conventions / decisions:**
- Inversion test: `s10y3m <= 0` (a flat 0 spread counts as inverted; tie goes to the
  signal firing). Points with `s10y3m === null` break a run (treated as not-inverted,
  not carried over).
- `streakDays` counts back from the most recent point while `s10y3m <= 0`; stops at
  the first non-inverted or null point.
- An **episode** is a maximal consecutive run of `s10y3m <= 0` points. A single
  inverted day is a 1-day episode.
- `maxDepthBps` = `min(s10y3m over the run) * 100` (most negative), so it is ≤ 0.
- `recessionFollowed` = some `NBER_RECESSIONS[i].start` falls in
  `[episode.start, episode.start + 24 months]`.
- `toSpreadPoints` reads `BC_10YEAR`, `BC_3MONTH`, `BC_2YEAR` (same columns as
  `spreadSeries` in `lib/finance.ts`); a spread is `null` if either leg is missing.

**NBER_RECESSIONS (US, peak→trough, month precision):**
- 1990-07-01 → 1991-03-01
- 2001-03-01 → 2001-11-01
- 2007-12-01 → 2009-06-01
- 2020-02-01 → 2020-04-01

### Page `web/app/signal/page.tsx` + `SignalClient.tsx`

- `page.tsx`: thin server component with `metadata`, renders `<SignalClient/>`
  (mirrors `app/pricing/page.tsx`).
- `SignalClient.tsx` ("use client"): on mount, fetches
  `/api/treasury/spreads?start=1990-01-01&end=<today>`; fail-soft loading/error
  states identical in spirit to `YieldCurveClient`. Derives `currentStatus` and
  `inversionEpisodes` via `useMemo`.

## UI

1. **Status hero** — large current 10y–3m (bps), a normal/inverted badge (tone by
   sign), current streak ("Inverted 412 days" / "Normal"), and the 2s10s readout.
2. **Spread history chart** — 10y–3m and 2s10s since 1990 over time (bps), zero
   baseline dashed, **red fill where 10y–3m < 0** (inversion), **gray vertical bands
   for NBER recessions**. Implemented as a dedicated `SpreadHistoryChart` component
   (the band + negative-fill shading is beyond `LineChart`'s current props; building
   a focused chart is cleaner than overloading `LineChart`). D3 `scaleTime`/
   `scaleLinear`, consistent with existing charts' styling tokens.
3. **Episode table** — every 10y–3m inversion since 1990: start, end, duration
   (days), max depth (bps), recession-within-24m (✓/✗). Most recent first.
4. **Auto-summary line** — e.g. "10y–3m inverted before all 4 recessions since 1990;
   currently inverted 412 days, deepest −108 bps." Built from status + episodes.

## Error handling

- Missing `start`/`end` → 400 `{ points: [] }`.
- Any year fetch failure → that year skipped.
- Empty points → page shows an "unavailable" message; `currentStatus` returns a
  null/`inverted:false`/`streakDays:0` shape; `inversionEpisodes` returns `[]`.

## Testing (`web/lib/signal.test.ts`, vitest)

- `toSpreadPoints`: computes both spreads; `null` when a leg is missing.
- `currentStatus`: streak counts consecutive trailing inverted points; stops at a
  positive or null; `inverted` reflects latest; empty → zeroed status.
- `inversionEpisodes`:
  - single inverted day → one 1-day episode;
  - multi-day run → correct start/end/days;
  - run still inverted at the last point → episode ends at last date;
  - two separated runs → two episodes;
  - `maxDepthBps` is the most negative value × 100;
  - `recessionFollowed` true when an NBER start is within 24 months, false otherwise;
  - all-positive series → `[]`.

## Navigation

- Add "Signal" (or "Recession Signal") to `components/Nav.tsx` in the Fixed Income
  group, and a Home card linking to `/signal`.

## Out of scope (YAGNI)

- Real-time alerting, Fed NY recession-probability model, other spread pairs,
  configurable lookahead, downloadable data.

## Note for implementation

`web/AGENTS.md`: Next.js 16 has breaking changes — read the route/page guide in
`node_modules/next/dist/docs/` before writing the API route and page.
