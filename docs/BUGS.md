# BondViz — Bugs & Tech Debt

> Tiered by severity. Each item is self-contained: location, what's wrong, and a fix sketch, so an
> agent can pick one up cold. IDs are stable — reference them in commits/PRs. Reviewed 2026-06-28.
>
> Baseline at review time: web 105 tests pass, build + eslint clean; Python 18 tests pass. **None
> of these break the build or current tests** — they are latent correctness, polish, and debt.

## Severity legend

- **P1 — correctness**, can produce wrong output in some environment/input.
- **P2 — polish / UX / minor correctness**, visible but not wrong-by-much.
- **P3 — debt / consistency**, internal quality, no user-facing symptom.

---

## P1 — Correctness

### BUG-1 · Timezone-dependent Treasury date parsing
- **Where:** `web/lib/treasury.ts:26-29` (`new Date(rawDate).toISOString().slice(0,10)`); same
  shape in `polygon.ts` is safe because its input is epoch-ms.
- **Problem:** `NEW_DATE` arrives as a wall-clock string (e.g. `2025-01-03T00:00:00`). `new Date()`
  interprets it in the **server's local timezone**, then `toISOString()` converts to UTC. On any
  server **east of UTC** (positive offset), midnight-local maps to the **previous** UTC day → every
  date shifts back one day. Production is currently safe only because **Vercel runs UTC**; local
  dev/tests on other zones can silently produce off-by-one dates and flaky fixtures.
- **Fix:** parse the date components explicitly (take the `YYYY-MM-DD` prefix, or construct with
  `Date.UTC(...)`), don't round-trip through local time. Add a test that sets `TZ=Asia/Tokyo`.

### BUG-2 · `next lint` script is broken on Next 16
- **Where:** `web/README.md` and any agent muscle-memory; `package.json` `scripts.lint` = `eslint`.
- **Problem:** `next lint` was **removed in Next 16** — running it errors with
  `Invalid project directory provided, no such directory: .../web/lint`. Docs/agents that reach for
  `next lint` will think the project is broken.
- **Fix:** ensure all docs say `npm run lint` / `npx eslint .`. (ARCHITECTURE.md already does.)
  Optionally add a clearer `lint` script comment. Low effort, prevents wasted agent cycles.

---

## P2 — Polish / UX / minor correctness

### BUG-3 · `LineChart` uses a linear scale for time axes
- **Where:** `web/components/charts/LineChart.tsx:45` — always `scaleLinear()`, even when
  `xType="time"`. `SpreadHistoryChart.tsx` correctly uses `scaleTime`.
- **Problem:** with `xType="time"` the x values are epoch-ms; `scaleLinear().nice()` snaps the
  domain to round **millisecond** numbers and `.ticks(6)` puts ticks at arbitrary ms, so month
  labels land at odd positions and can repeat. Affects spreads-over-time (yield-curve), PCA scores,
  and stocks charts. Cosmetic but visibly "off."
- **Fix:** branch on `xType`: use `scaleTime()` (from `d3-scale`) for time, `scaleLinear()`
  otherwise; format ticks from the time scale.

### BUG-4 · Heatmap has no date (row) axis or color legend
- **Where:** `web/components/charts/Heatmap.tsx` (renders tenor labels only) + blurb in
  `YieldCurveClient.tsx` that says "each row is a date (old → new)".
- **Problem:** the explanatory text references a date axis that isn't drawn, and there's no
  colorbar to read magnitudes. The heatmap is decorative rather than legible.
- **Fix:** add sparse y-axis date ticks (first/last + a few in between) and a small min→max
  colorbar legend using the same `colorRamp`.

### BUG-5 · `PortfolioClient` resolves per-holding metrics by `indexOf`
- **Where:** `web/app/portfolio/PortfolioClient.tsx:109` —
  `metrics.holdings[holdings.indexOf(h)]`.
- **Problem:** `indexOf` is O(n) inside a `.map` (O(n²)) and couples the metrics array order to
  object identity. Harmless today (small books) but fragile if holdings are ever keyed/reordered.
- **Fix:** iterate with the map index (`holdings.map((h, i) => … metrics.holdings[i])`) since
  `portfolioMetrics` preserves input order.

### BUG-6 · Stocks fetch effect depends on the whole `cache` object
- **Where:** `web/app/stocks/StocksClient.tsx:74` — effect deps include `cache`.
- **Problem:** every cache write re-runs the effect for all tickers; it's guarded by a
  `cache[cacheKey]` early-return so it won't refetch, but it's needless churn and easy to turn into
  a refetch loop during edits.
- **Fix:** drop `cache` from deps and read latest via the `setCache` updater, or split the
  "has this key" check out of the effect.

---

## P3 — Debt / consistency

### BUG-7 · Hardcoded chart colors defeat tokenization
- **Where:** 20 hex literals across 8 files (full inventory in `docs/DESIGN-SYSTEM.md`).
- **Problem:** series colors don't come from CSS vars, so a theme/design-system change won't reach
  charts. This is the top blocker for the in-progress design-system work.
- **Fix:** `web/lib/chartColors.ts` palette sourced from new `--series-*` vars; replace literals.
  (Same work as **FEAT-2**.)

### BUG-8 · `treasury/range` fetches years sequentially; `spreads` is parallel
- **Where:** `web/app/api/treasury/range/route.ts:15-18` (for-loop `await`) vs
  `spreads/route.ts:19` (`Promise.all`).
- **Problem:** inconsistent and slower for multi-year windows (PCA 5yr, yield-curve ~1yr). The
  parallel pattern already exists next door.
- **Fix:** mirror `spreads`: `await Promise.all(years.map(y => fetchTreasuryYear(y).catch(()=>[])))`.

### BUG-9 · Root `README.md` / `CLAUDE.md` describe only the Streamlit app
- **Where:** repo-root `README.md`, `CLAUDE.md`.
- **Problem:** they predate the Vercel rebuild and present Streamlit as the product; an agent
  orienting from them builds the wrong mental model. `web/` is barely mentioned.
- **Fix:** add a prominent "the web app in `web/` is the active product; this section is the legacy
  Streamlit reference" banner at the top of both, pointing to `docs/ARCHITECTURE.md`. (Low effort,
  high leverage for the agent swarm.)

### BUG-10 · `web/README.md` page list is stale
- **Where:** `web/README.md:3-5` — "Home, Yield Curve explorer, and Bond Pricing".
- **Problem:** the app now also has Carry, Signal, Portfolio, PCA, Stocks. Doc drift.
- **Fix:** update to the current 8-route list (see ARCHITECTURE.md table).

### BUG-11 · `Kpi` and `Metric` primitives overlap
- **Where:** `web/components/ui/Kpi.tsx` vs `Metric.tsx`.
- **Problem:** two primitives do nearly the same job; `Kpi` is the older/simpler one and is now
  largely unused in favor of `Metric`. Redundant surface for the design-system refactor.
- **Fix:** fold `Kpi` usages into `Metric` and delete `Kpi`, or keep one deliberately. Do it as
  part of the restyle.

### BUG-12 · No route-level `loading.tsx` / `error.tsx`
- **Where:** `web/app/**` — pages handle their own loading/error inside the client component.
- **Problem:** initial navigation shows nothing until the client mounts; an uncaught render error
  has no boundary. Not wrong, but below App-Router norms.
- **Fix:** add `loading.tsx` skeletons and an `error.tsx` boundary per route (or a shared one).

### BUG-13 · `SpreadHistoryChart` inversion fill doesn't interpolate zero-crossings
- **Where:** `web/components/charts/SpreadHistoryChart.tsx:45-48` (carried over as a known MINOR in
  `.superpowers/sdd/progress.md` Task 3).
- **Problem:** the red "inverted" area clips each point to zero rather than the exact crossing, so
  fills are very slightly clipped at sign changes. Acceptable for a dashboard; noted for honesty.
- **Fix (optional):** split segments at interpolated zero crossings before building the area.
