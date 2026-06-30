# BondViz — Backlog (features & improvements)

> Agent-ready briefs. Each is self-contained (goal · why · files · approach · effort · deps ·
> acceptance) so a fresh agent can start with this entry + the named files and little else. IDs are
> stable. Effort: **S** ≈ <½ day, **M** ≈ 1–2 days, **L** ≈ 3+ days. Reviewed 2026-06-28.
>
> Build on the patterns in `docs/ARCHITECTURE.md` ("Recipe: add a new tool/page"). Keep
> `npm test` + `npx eslint .` + `npm run build` green. Fixed-income features should reuse
> `web/lib/finance.ts` primitives.
>
> **Dispatch unit = one atomic sub-task.** Items larger than ~S are split into lettered sub-tasks
> (`FEAT-2a`, `FEAT-2b`, …). Give a small agent **one sub-task**, the named files, and the parent
> entry for context — each sub-task touches 1–2 files, has a single acceptance check, ships
> independently, and must leave the build/tests green. Do the lib/test sub-task before its UI
> sub-tasks. Land each as its own small commit/branch.

## Suggested order

1. **Enablers** (unblock the design-system work + safety net): FEAT-1, FEAT-2, FEAT-3.
2. **High-leverage, low-risk features**: FEAT-4, FEAT-5, FEAT-6, FEAT-7.
3. **Depth / showcase**: FEAT-8 … FEAT-14.

---

## A. Enablers (do these first)

### FEAT-1 · CI workflow (test + lint + build gate) — **S**

- **Goal:** every push/PR runs Python + web checks so the swarm can't silently break `main`.
- **Why:** finished branches merge straight to `main` (no PR review). There's no automated gate.
- **Files:** new `.github/workflows/ci.yml`.
- **Approach:** two jobs — (a) Python: setup 3.x, `pip install -e .`, `pytest`; (b) web: Node 20,
  `cd web && npm ci && npm test && npx eslint . && npm run build`. Cache npm + pip.
- **Acceptance:** workflow green on a trivial PR; a deliberately failing test fails the job.
- **Atomic:** single-file task; no split needed.

### FEAT-2 · Tokenize chart series colors — **M** — ✅ DONE (`feat/design-tokens`) *(also BUG-7; unblocks design system)*

- **Goal:** no raw hex in chart/series code; one palette module driven by CSS vars.
- **Files:** new `web/lib/chartColors.ts`; `web/app/globals.css`; edit `YieldCurveClient`,
  `PcaClient`, `CarryClient`, `StocksClient`, `SpreadHistoryChart`, `Heatmap`.
- **Approach:** export `SERIES`, `COMPARE`, `ACCENT`, `HEATMAP_STOPS` from `chartColors.ts`, each
  `var(--…)`; replace the 20 literals (inventory in `docs/DESIGN-SYSTEM.md`). Keep current colors
  (map new vars to existing hex) so it ships with **zero visual change**.
- **Acceptance (parent):** `grep -rE '#[0-9a-fA-F]{6}' web/app web/components` returns only test
  files; build + tests green; app looks identical.
- **Atomic sub-tasks** (do 2a first; 2b–2f are independent after it):
  - **FEAT-2a** — add `--series-1..6` + ramp stops to `globals.css`; create `chartColors.ts`
    exporting the palette (values = current hex) + a test asserting the exports. *Accept:* module
    imports, build green, nothing visual changes.
  - **FEAT-2b** — `YieldCurveClient.tsx`: COMPARE palette + series colors → `chartColors`. *Accept:*
    no hex in file; chart identical.
  - **FEAT-2c** — `PcaClient.tsx`: `COMPONENT_COLORS` → `SERIES`.
  - **FEAT-2d** — `CarryClient.tsx` + `StocksClient.tsx` literals → tokens.
  - **FEAT-2e** — `SpreadHistoryChart.tsx` `#5b8def` → token.
  - **FEAT-2f** — `Heatmap.tsx` `STOPS` → `HEATMAP_STOPS`.

### FEAT-3 · Map CSS vars into Tailwind v4 `@theme` — **S** — ✅ DONE (`feat/design-tokens`) *(unblocks design system)*

- **Goal:** allow semantic utilities (`bg-panel`, `text-muted`, `border-accent`) instead of
  `bg-[var(--panel)]` everywhere.
- **Files:** `web/app/globals.css` (add an `@theme { --color-*: var(--*); }` block).
- **Approach:** register the palette as Tailwind theme colors; codemod of existing
  `*-[var(--*)]` usages can be incremental (both forms work side by side).
- **Acceptance:** a sample component using `bg-panel text-muted` renders correctly; build green.
- **Atomic:** single-file task; the optional codemod can be its own later sweep.

---

## B. High-leverage features

### FEAT-4 · Shareable / bookmarkable state via URL query params — **M**

- **Goal:** encode page inputs (pricing bond, portfolio holdings, lookbacks, horizons) in the URL
  so a configured view can be linked/refreshed — a strong recruiter-facing touch.
- **Files:** new `web/lib/urlState.ts` (+test); the client pages.
- **Approach:** sync `useState` ⇄ `useSearchParams`/`router.replace`; debounce URL writes; hydrate
  initial state from the query.
- **Acceptance (parent):** changing inputs updates the URL; pasting that URL reproduces the view.
- **Atomic sub-tasks** (4a first):
  - **FEAT-4a** — `urlState.ts` encode/decode helpers + round-trip test. No UI.
  - **FEAT-4b** — wire `PricingClient`.
  - **FEAT-4c** — wire `PortfolioClient` (holdings array).
  - **FEAT-4d** — wire Carry / PCA / Stocks lookbacks & horizons.

### FEAT-5 · CSV / PNG export for charts & tables — **M**

- **Goal:** an "Export" affordance on each Card to download the table as CSV and the chart as PNG.
- **Files:** new `web/lib/exportCsv.ts` (+test), `web/lib/exportSvg.ts`; an `ExportButton` in
  `components/ui/`; wire into chart Cards.
- **Approach:** CSV is pure (test it). PNG: serialize the chart `<svg>` → canvas → `toBlob`. No new
  deps.
- **Acceptance (parent):** CSV downloads with correct headers/rows; PNG matches the on-screen chart.
- **Atomic sub-tasks** (5a first):
  - **FEAT-5a** — `exportCsv.ts` serializer (rows → CSV string) + test.
  - **FEAT-5b** — `ExportButton` primitive + CSV download wired on one Card (e.g. Pricing scenario).
  - **FEAT-5c** — `exportSvg.ts` SVG→PNG + wire across chart Cards.

### FEAT-6 · Methodology / "About the math" page — **S**

- **Goal:** an `/about` route documenting conventions (continuous compounding, bootstrap, carry,
  PCA, NBER signal) with the formulas already in code comments.
- **Why:** turns the analytics into a legible portfolio narrative; cheap, high signal.
- **Files:** `web/app/about/page.tsx`; add to `Nav` `LINKS`.
- **Approach:** static content; pull formula prose from the docstrings in `finance.ts`/`curve.ts`/
  `carry.ts`/`signal.ts`. No data fetching.
- **Acceptance:** route builds static; linked from nav; renders the formulas/conventions.
- **Atomic:** single page; no split needed.

### FEAT-7 · Command palette (⌘K) navigation — **S/M**

- **Goal:** keyboard-driven nav across the 8 tools; signals product polish.
- **Files:** new `web/components/CommandPalette.tsx`; mount in `layout.tsx`.
- **Approach:** hand-rolled (no dep): a modal listening for ⌘/Ctrl-K, filters the `LINKS` list,
  `router.push` on select. Reuse tokens; trap focus; Escape closes; respect reduced-motion.
- **Acceptance:** ⌘K opens, arrow/enter navigates, a11y (focus trap + aria) holds; eslint green.
- **Atomic:** one component + one mount; keep as a single task.

---

## C. Depth / showcase features

### FEAT-8 · Curve scenario / shock builder — **M**

- **Goal:** on `/yield-curve`, apply named shocks (parallel ±, bull/bear steepener/flattener,
  twist) to the live curve and overlay the shocked curve + reprice a sample bond.
- **Files:** new `web/lib/curveShocks.ts` (+test); `YieldCurveClient`.
- **Approach:** pure functions mapping a base curve + shock spec → new curve (per-tenor bps); reuse
  `LineChart` overlay; optionally feed into `finance.scenarioShift`.
- **Acceptance (parent):** each shock transforms the curve as specified; overlay renders.
- **Atomic sub-tasks** (8a first):
  - **FEAT-8a** — `curveShocks.ts`: named shocks → per-tenor bps → new curve, + tests per shock.
  - **FEAT-8b** — shock selector + overlay shocked curve in `YieldCurveClient`.
  - **FEAT-8c** — reprice a sample bond on the shocked curve via `scenarioShift`.

### FEAT-9 · `/styleguide` component catalog — **S** — ✅ DONE (`feat/styleguide`) *(supports design-system work)*

- **Goal:** one page rendering every `ui/*` primitive and chart in all states — the design
  iteration surface for the new system.
- **Files:** `web/app/styleguide/page.tsx`.
- **Approach:** import each primitive/chart with representative props; group by category. No tests
  beyond build.
- **Acceptance:** page shows Card/Metric/Kpi/Segmented + each chart; updates live when tokens change.
- **Atomic:** single page; no split needed.

### FEAT-10 · Decision: adopt shadcn/ui vs keep hand-rolled kit — **S (spike)**

- **Goal:** resolve the design-system foundation before mass-restyling.
- **Approach:** short spike doc weighing shadcn (Radix a11y, velocity, `vercel:shadcn` skill) vs the
  current zero-dependency kit (a selling point for the portfolio). Recommend one. Charts stay
  hand-rolled either way.
- **Acceptance:** a decision doc with a recommendation; no code unless approved.
- **Atomic:** single decision doc.

### FEAT-11 · Theming infrastructure (light mode + toggle) — **M** *(only if the new system needs it)*

- **Goal:** a second palette + a toggle, app-wide including charts.
- **Files:** `globals.css`, a `ThemeToggle` in `Nav`, `localStorage` persistence.
- **Approach:** depends on FEAT-2 + FEAT-3 (tokens must be complete and charts must read tokens).
- **Acceptance (parent):** toggle flips the palette app-wide incl. charts; choice persists; no FOUC.
- **Atomic sub-tasks** (blocked on FEAT-2/3):
  - **FEAT-11a** — second token set under `[data-theme="light"]` in `globals.css`.
  - **FEAT-11b** — `ThemeToggle` in `Nav` + `localStorage` + no-FOUC inline script in `layout.tsx`.

### FEAT-12 · Multi-ticker overlay on Stocks — **S/M**

- **Goal:** plot all entered tickers on one normalized (rebased-to-100) chart, not just the active one.
- **Files:** `StocksClient`, `LineChart` (already multi-series).
- **Approach:** fetch each ticker (cache already keyed per ticker), rebase to first close, push one
  `Series` per ticker; keep the single-ticker detail table.
- **Acceptance:** N tickers → N normalized lines + legend; missing key still degrades gracefully.
- **Atomic:** one client change; keep as a single task (depends on FEAT-2 for series colors).

### FEAT-13 · Relative-value: spread z-scores vs history — **M**

- **Goal:** show current 2s10s / 3m10y vs their trailing distribution (z-score, percentile) using
  the deep history already cached.
- **Files:** new `web/lib/relativeValue.ts` (+test); `SignalClient` or a new client.
- **Approach:** pure stats over `SpreadPoint[]` from `spreadsCache`; reuse `Metric` for readouts.
- **Acceptance (parent):** z-score/percentile match a hand-computed fixture; renders on the page.
- **Atomic sub-tasks** (13a first):
  - **FEAT-13a** — `relativeValue.ts` z-score/percentile + test against a fixture.
  - **FEAT-13b** — render the readouts on `/signal` (or a new `/relative-value`).

### FEAT-14 · Key-rate (partial) durations — **M**

- **Goal:** per-tenor key-rate durations on `/pricing` and `/portfolio` so risk shows by curve
  bucket, not just total duration.
- **Files:** `web/lib/finance.ts` (+tests); `PricingClient` / `PortfolioClient`; reuse
  `CategoryBarChart`.
- **Approach:** bump each tenor's zero rate by 1bp, reprice, measure ΔP/P per bucket; sums ≈ total
  effective duration. Continuous-compounding consistent with existing math.
- **Acceptance (parent):** KRDs sum to ~total duration; bar chart per tenor renders.
- **Atomic sub-tasks** (14a first):
  - **FEAT-14a** — `keyRateDurations` in `finance.ts` + test (sums ≈ total duration).
  - **FEAT-14b** — KRD bar chart in `PricingClient`.
  - **FEAT-14c** — KRD in `PortfolioClient`.

---

## Parked / depends on external decisions

- **Per-visitor input persistence** — depends on an auth provider. The no-auth interim is FEAT-4
  (URL state) + `localStorage`.
- **Auth** — a full Clerk build exists on the unmerged `feat/auth-clerk` branch; the user is
  evaluating providers. Don't re-implement auth without confirming the provider.
- **Streamlit/Python parity** — decide whether to keep porting (`src/bondviz` ⇄ `web/lib`
  numerically identical) or archive the Streamlit app. Affects whether new math lands in one place
  or two. Worth an explicit decision doc.
