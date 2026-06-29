# BondViz — Backlog (features & improvements)

> Agent-ready briefs. Each is self-contained (goal · why · files · approach · effort · deps ·
> acceptance) so a fresh agent can start with this entry + the named files and little else. IDs are
> stable. Effort: **S** ≈ <½ day, **M** ≈ 1–2 days, **L** ≈ 3+ days. Reviewed 2026-06-28.
>
> Build on the patterns in `docs/ARCHITECTURE.md` ("Recipe: add a new tool/page"). Keep
> `npm test` + `npx eslint .` + `npm run build` green. Fixed-income features should reuse
> `web/lib/finance.ts` primitives.

## Suggested order

1. **Enablers** (unblock the design-system work + safety net): FEAT-1, FEAT-2, FEAT-3.
2. **High-leverage, low-risk features**: FEAT-4, FEAT-5, FEAT-6, FEAT-7.
3. **Depth / showcase**: FEAT-8 … FEAT-14.

---

## A. Enablers (do these first)

### FEAT-1 · CI workflow (test + lint + build gate)  — **S**
- **Goal:** every push/PR runs Python + web checks so the swarm can't silently break `main`.
- **Why:** finished branches merge straight to `main` (no PR review). There's no automated gate.
- **Files:** new `.github/workflows/ci.yml`.
- **Approach:** two jobs — (a) Python: setup 3.x, `pip install -e .`, `pytest`; (b) web: Node 20,
  `cd web && npm ci && npm test && npx eslint . && npm run build`. Cache npm + pip.
- **Acceptance:** workflow green on a trivial PR; a deliberately failing test fails the job.

### FEAT-2 · Tokenize chart series colors  — **M**  *(also BUG-7; unblocks design system)*
- **Goal:** no raw hex in chart/series code; one palette module driven by CSS vars.
- **Files:** new `web/lib/chartColors.ts`; `web/app/globals.css` (add `--series-1..6` + heatmap
  ramp stops); edit `YieldCurveClient`, `PcaClient`, `CarryClient`, `StocksClient`,
  `SpreadHistoryChart`, `Heatmap`.
- **Approach:** export `SERIES: string[]`, `COMPARE: string[]`, `ACCENT`, and `HEATMAP_STOPS` from
  `chartColors.ts`, each `var(--…)`. Replace the 20 literals (inventory in
  `docs/DESIGN-SYSTEM.md`). Keep current colors (map new vars to existing hex) so this ships with
  **zero visual change**.
- **Acceptance:** `grep -rE '#[0-9a-fA-F]{6}' web/app web/components` returns only test files;
  build + tests green; app looks identical.

### FEAT-3 · Map CSS vars into Tailwind v4 `@theme`  — **S**  *(unblocks design system)*
- **Goal:** allow semantic utilities (`bg-panel`, `text-muted`, `border-accent`) instead of
  `bg-[var(--panel)]` everywhere.
- **Files:** `web/app/globals.css` (add an `@theme { --color-*: var(--*); }` block).
- **Approach:** register the palette as Tailwind theme colors; optionally codemod the most common
  `*-[var(--*)]` usages to the new utilities (can be incremental — both work side by side).
- **Acceptance:** a sample component using `bg-panel text-muted` renders correctly; build green.

---

## B. High-leverage features

### FEAT-4 · Shareable / bookmarkable state via URL query params  — **M**
- **Goal:** encode page inputs (pricing bond, portfolio holdings, lookbacks, horizons) in the URL
  so a configured view can be linked/refreshed — a strong recruiter-facing touch.
- **Files:** new `web/lib/urlState.ts` (encode/decode + test); the client pages (`PricingClient`,
  `PortfolioClient`, `CarryClient`, `StocksClient`, `PcaClient`).
- **Approach:** sync `useState` ⇄ `useSearchParams`/`router.replace` (App Router). Keep it a thin,
  tested serializer; debounce URL writes. Hydrate initial state from the query.
- **Acceptance:** changing inputs updates the URL; pasting that URL reproduces the view; tests
  cover round-trip encode/decode.

### FEAT-5 · CSV / PNG export for charts & tables  — **M**
- **Goal:** "Export" on each Card to download the underlying table as CSV and the chart as PNG.
- **Files:** new `web/lib/exportCsv.ts` (+test) and `web/lib/exportSvg.ts` (serialize the chart
  `<svg>` → canvas → PNG); a small `ExportButton` in `components/ui/`; wire into chart Cards.
- **Approach:** CSV is pure (test it). PNG: serialize SVG, draw to canvas, `toBlob`. No new deps.
- **Acceptance:** CSV downloads with correct headers/rows; PNG matches the on-screen chart; CSV
  serializer unit-tested.

### FEAT-6 · Methodology / "About the math" page  — **S**
- **Goal:** a `/about` route documenting conventions (continuous compounding, bootstrap, carry,
  PCA, NBER signal) with the formulas already in code comments.
- **Why:** turns the analytics into a legible portfolio narrative; cheap, high signal for recruiters.
- **Files:** `web/app/about/page.tsx` (+ optional client); add to `Nav` `LINKS`.
- **Approach:** static content; pull formula prose from the docstrings in `finance.ts`/`curve.ts`/
  `carry.ts`/`signal.ts`. No data fetching.
- **Acceptance:** route builds static; linked from nav; renders the formulas/conventions.

### FEAT-7 · Command palette (⌘K) navigation  — **S/M**
- **Goal:** keyboard-driven nav across the 8 tools; signals product polish.
- **Files:** new `web/components/CommandPalette.tsx`; mount in `layout.tsx`.
- **Approach:** hand-rolled (no dep): a modal listening for ⌘/Ctrl-K, filters the `LINKS` list,
  `router.push` on select. Reuse design tokens; trap focus; Escape closes; respect reduced-motion.
- **Acceptance:** ⌘K opens, arrow/enter navigates, a11y (focus trap + aria) holds; eslint green.

---

## C. Depth / showcase features

### FEAT-8 · Curve scenario / shock builder  — **M**
- **Goal:** on `/yield-curve`, apply named shocks (parallel ±, bull/bear steepener/flattener,
  twist) to the live curve and overlay the shocked curve + reprice a sample bond.
- **Files:** new `web/lib/curveShocks.ts` (+test); `YieldCurveClient`.
- **Approach:** pure functions mapping a base curve + shock spec → new curve (per-tenor bps). Reuse
  `LineChart` overlay. Optionally feed the shocked curve into `finance.scenarioShift`.
- **Acceptance:** each shock transforms the curve as specified (unit-tested); overlay renders.

### FEAT-9 · `/styleguide` component catalog  — **S**  *(supports design-system work)*
- **Goal:** one page rendering every `ui/*` primitive and chart in all states — the design
  iteration surface for your new system.
- **Files:** `web/app/styleguide/page.tsx`.
- **Approach:** import each primitive/chart with representative props; group by category. Gate from
  nav (or leave unlinked). No tests required beyond build.
- **Acceptance:** page shows Card/Metric/Kpi/Segmented + each chart; updates live when tokens change.

### FEAT-10 · Decision: adopt shadcn/ui vs keep hand-rolled kit  — **S (spike)**
- **Goal:** resolve the design-system foundation before mass-restyling.
- **Approach:** short spike doc (`docs/superpowers/specs/…`) weighing shadcn (Radix a11y, velocity,
  `vercel:shadcn` skill) vs the current zero-dependency kit (a selling point for the portfolio).
  Recommend one. Charts stay hand-rolled either way.
- **Acceptance:** a decision doc with a recommendation; no code unless approved.

### FEAT-11 · Theming infrastructure (light mode + toggle)  — **M**  *(only if the new system needs it)*
- **Files:** `globals.css` (a second token set under `[data-theme="light"]`/`prefers-color-scheme`),
  a `ThemeToggle` in `Nav`, persistence in `localStorage`.
- **Approach:** depends on FEAT-2/FEAT-3 (tokens must be complete first). Ensure charts read tokens,
  not hex.
- **Acceptance:** toggle flips palette app-wide incl. charts; choice persists; no FOUC.

### FEAT-12 · Multi-ticker overlay on Stocks  — **S/M**
- **Goal:** plot all entered tickers on one normalized (rebased-to-100) chart, not just the active one.
- **Files:** `StocksClient`, `LineChart` (already multi-series).
- **Approach:** fetch each ticker (cache already keyed per ticker), rebase to first close, push one
  `Series` per ticker. Keep the single-ticker detail table.
- **Acceptance:** N tickers → N normalized lines + legend; missing key still degrades gracefully.

### FEAT-13 · Relative-value: spread z-scores vs history  — **M**
- **Goal:** on `/signal` (or a new `/relative-value`), show current 2s10s / 3m10y vs their trailing
  distribution (z-score, percentile) using the deep history already cached.
- **Files:** new `web/lib/relativeValue.ts` (+test); `SignalClient` or a new client.
- **Approach:** pure stats over `SpreadPoint[]` from `spreadsCache`. Reuse `Metric` for the readouts.
- **Acceptance:** z-score/percentile match a hand-computed fixture; renders on the page.

### FEAT-14 · Key-rate (partial) durations  — **M**
- **Goal:** add per-tenor key-rate durations to `/pricing` and `/portfolio` so risk is shown by
  curve bucket, not just total duration.
- **Files:** `web/lib/finance.ts` (add `keyRateDurations`, +tests); `PricingClient` /
  `PortfolioClient`; reuse `CategoryBarChart`.
- **Approach:** bump each tenor's zero rate by 1bp, reprice, measure ΔP/P per bucket; sums ≈ total
  effective duration. Continuous-compounding consistent with existing math.
- **Acceptance:** KRDs sum to ~total duration (tested); bar chart per tenor renders.

---

## Parked / depends on external decisions

- **Per-visitor input persistence** — see memory note `deferred-user-input-storage`; depends on an
  auth provider. The no-auth interim is FEAT-4 (URL state) + `localStorage`.
- **Auth** — a full Clerk build exists on the unmerged `feat/auth-clerk` branch
  (memory `parked-clerk-auth-branch`); the user is evaluating providers. Don't re-implement auth
  without confirming the provider.
- **Streamlit/Python parity** — decide whether to keep porting (keep `src/bondviz` ⇄ `web/lib`
  numerically identical) or archive the Streamlit app. Affects whether new math lands in one place
  or two. Worth an explicit decision doc.
