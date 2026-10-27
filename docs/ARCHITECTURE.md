# BondViz — Architecture & Agent Orientation

> **Read this first.** One file, full mental model. If you are an agent picking up a task,
> this doc + the one file you're editing should be enough context. Last reviewed: 2026-06-28.

## TL;DR

BondViz is **two parallel codebases** in one repo:

| Surface | Path | Stack | Status |
| --- | --- | --- | --- |
| **Web app (PRIMARY)** | `web/` | Next.js 16 (App Router) · React 19 · TypeScript · Tailwind v4 · hand-rolled D3 | **Active.** Deployed to Vercel at `bondviz.nikolayvalev.com`. All new work happens here. |
| Streamlit app (legacy) | `src/bondviz/`, `app/`, `scripts/` | Python · Streamlit · matplotlib · pybind11 | Frozen reference. Not deployed publicly. The TS lib is a port of this. |

**The web app is the product.** The Python app is kept as a numerical reference; the TS
`web/lib/*.ts` modules are direct ports of `src/bondviz/*.py`. Don't assume the root `CLAUDE.md`
/ `README.md` (which describe the Streamlit app) reflect where the work is — they predate the
Vercel rebuild.

Health baseline (2026-10-16): web `npm test` = **242 pass** (39 files), `npm run build` = **green**,
`npx eslint .` = **clean**; Python `pytest` = **18 pass**.

---

## The web app in one diagram

```
Browser ──► Next.js App Router (web/)
              │
   ┌──────────┴───────────────────────────────────────────────┐
   │ Pages (app/<route>/page.tsx)  →  thin server wrapper       │
   │   sets <metadata title>, renders one Client component      │
   │                                                            │
   │ Clients (app/<route>/<Name>Client.tsx)  "use client"       │
   │   owns state, fetches /api/*, calls lib/*, renders charts  │
   └──────────┬─────────────────────────────┬───────────────────┘
              │ fetch                        │ pure calls
              ▼                              ▼
   API routes (app/api/**/route.ts)    lib/*.ts  (pure, tested)
     fetch external feed, parse,         finance, curve, carry,
     cache, return typed JSON            signal, pca, portfolio
              │                              ▲
              ▼                              │ render from values
   External data                       components/charts/*  (D3 math,
     Treasury XML (keyless)              React owns the SVG DOM)
     Polygon.io (POLYGON_API_KEY)       components/ui/*  (Card, Metric, …)
```

**One rule that explains the layout:** *math and data are pure & tested (`lib/`), the DOM is
React (`components/`, `app/`), and external I/O is isolated in `app/api/`.* Charts never call
`d3.select`; D3 only computes scales/paths and React renders the SVG.

---

## Directory map (`web/`)

```
web/
├── app/
│   ├── layout.tsx            # root shell: fonts (IBM Plex), <Nav>, <main>, footer; root metadata
│   ├── page.tsx              # Home: hero + live KPI snapshot + grouped tool cards (ISR, 1h revalidate)
│   ├── about/page.tsx        # Methodology: conventions + every formula (static)
│   ├── not-found / error / global-error .tsx   # branded 404 + error boundaries (no root loading.tsx: it made
│   │                                            #   static pages stream a skeleton first → CLS 0.13)
│   ├── opengraph-image.tsx   # 1200×630 social card (static, Satori; colors from lib/brand.ts)
│   ├── robots.ts, sitemap.ts # generated from lib/site.ts + lib/tools.ts
│   ├── globals.css           # ★ ALL design tokens live here (CSS vars in :root) + Tailwind
│   ├── PrefetchSignal.tsx    # warms the 35-yr spread cache on idle from Home
│   ├── <route>/page.tsx      # server wrapper (metadata only)
│   ├── <route>/<Name>Client.tsx   # the actual page (client component)
│   ├── command-panel/data.ts # DataRegistry: the only metrics chat-generated widgets can read
│   │   └── …/ChatPanel, ChatSession, DashboardPanel, chatStorage, layoutStorage, types, panelChrome
│   └── api/
│       ├── command-panel/route.ts      # POST chat → streamed widget proposals (GitHub Models); token check, per-IP rate limit, size cap
│       ├── command-panel/data/route.ts # POST read-only metric loads backing widget useMetric()
│       ├── treasury/latest/route.ts    # GET → latest par-yield row
│       ├── treasury/range/route.ts     # GET ?start&end → daily rows in window (parallel per-year fetch)
│       ├── treasury/spreads/route.ts   # GET ?start&end → slim spread points (parallel fetch)
│       └── stocks/aggregates/route.ts  # GET ?ticker&from&to → daily bars (needs API key)
├── lib/                      # ★ pure logic, each with a .test.ts sibling
│   ├── types.ts              # TENOR_LABELS, YieldRow, CurvePoint, Kpis, StockBar
│   ├── finance.ts            # bond math: PV, duration, convexity, DV01, scenarios, KPIs, tenor maps
│   ├── curve.ts              # par→zero/forward bootstrap (semiannual)
│   ├── interp.ts             # linear + monotone cubic (PCHIP) interp, flat extrapolation
│   ├── carry.ts              # carry & roll-down per tenor
│   ├── signal.ts             # inversion episodes vs NBER recessions
│   ├── pca.ts                # standardize → covariance → Jacobi eigensolver → PCA
│   ├── portfolio.ts          # multi-bond aggregation built on finance.ts
│   ├── treasury.ts           # fetch + parse Treasury XML (fast-xml-parser)
│   ├── polygon.ts            # fetch + parse Polygon aggregates
│   ├── relativePerformance.ts # rebase closes to 100, window returns + late-start flags (Stocks)
│   ├── spreadsCache.ts       # in-memory promise cache for the deep spread series
│   ├── panelLayout.ts        # command-panel dashboard grid math (span/height clamp, persist)
│   ├── chatStore.ts          # command-panel conversation index (create/rename/delete/evict)
│   ├── tools.ts              # ★ the tool registry: Nav, Home, sitemap and page metadata read it
│   ├── site.ts               # canonical URL, author/source links, pageMetadata()/rootMetadata
│   ├── brand.ts              # literal token hex for the OG renderer (test guards drift)
│   ├── rateLimit.ts          # fixed-window per-key limiter (command-panel chat route)
│   ├── curveShocks.ts        # named curve shocks (log-maturity shapes, 5Y pivot) + benchmark reprices
│   ├── relativeValue.ts      # z-score / percentile of today's spread vs a trailing window
│   ├── urlState.ts           # typed query-string codecs (num, oneOf, text, numSet, records)
│   ├── paletteSearch.ts      # ⌘K palette ranking
│   ├── exportCsv.ts          # CSV text (RFC 4180, formula-safe), series/category tables, file names
│   └── format.ts             # iso(), money(), money0(), signedPct()
├── components/
│   ├── Nav.tsx               # sticky top nav + menu below lg (links come from lib/tools.ts)
│   ├── CommandPalette.tsx    # ⌘K / Ctrl-K dialog (combobox + listbox), mounted in layout.tsx
│   ├── useUrlState.ts        # page state ⇄ query string (useSyncExternalStore + debounced replaceState)
│   ├── KeyRateRisk.tsx       # key-rate duration / DV01 bars (pricing + portfolio)
│   ├── LazyMount.tsx         # mount heavy below-the-fold charts when they near the viewport
│   ├── ui/                   # Card, Metric, Segmented, Skeleton, ScrollRegion, CopyLinkButton,
│   │                         # ExportButtons (CSV / PNG downloads)
│   └── charts/               # LineChart, CategoryBarChart, Heatmap, SpreadHistoryChart,
│                             # ScenarioChart, PriceYieldChart, CashflowChart, useResizeObserver,
│                             # exportPng (SVG → standalone copy → canvas → PNG)
├── test/                     # fixtures (treasury-sample.xml) + smoke test
├── globals/config            # next.config.ts (serverExternalPackages: command-panel), vercel.json,
│                             # tsconfig.json, eslint.config.mjs,
│                             # vitest.config.ts, postcss.config.mjs
└── CLAUDE.md / AGENTS.md     # agent rules (AGENTS.md warns: Next 16 ≠ training-data Next)
```

---

## The pages (what exists today)

| Route | Client | lib used | Data | Charts |
| --- | --- | --- | --- | --- |
| `/` | `page.tsx` (Home) | `finance.computeCurveKpis` | `/api/treasury/latest` (inline) | KPI metrics only |
| `/yield-curve` | `YieldCurveClient` (+ `ShockScenario`) | `finance`, `curve`, `curveShocks` | `/api/treasury/range` (~1yr) | LineChart ×5, CategoryBar, Heatmap |
| `/pricing` | `PricingClient` | `finance` (incl. `keyRateDurations`) | none (pure, client-side) | PriceYield, Cashflow, KeyRateRisk, Scenario |
| `/carry` | `CarryClient` | `carry`, `finance` | `/api/treasury/range` (30d) | CategoryBarChart ×2 |
| `/signal` | `SignalClient` (+ `RelativeValuePanel`) | `signal`, `spreadsCache`, `relativeValue` | `/api/treasury/spreads` (1990→now) | SpreadHistoryChart, range bars |
| `/portfolio` | `PortfolioClient` | `portfolio`, `finance` | none (book lives in the URL) | KeyRateRisk, ScenarioChart |
| `/pca` | `PcaClient` | `pca`, `finance` | `/api/treasury/range` (1–5yr) | LineChart ×2 |
| `/stocks` | `StocksClient` | `relativePerformance` | `/api/stocks/aggregates` (one call per ticker, debounced) | LineChart |
| `/command-panel` | `CommandPanelClient` (browser-only mount) | registry in `command-panel/data.ts` → `treasury`, `finance`, `curve`, `carry`, `signal` | `/api/command-panel` (chat, needs `GITHUB_MODELS_TOKEN`) + `/api/command-panel/data` | LLM-proposed widgets (`@nikolayvalev/command-panel`; see `docs/DECISIONS-command-panel.md`) |
| `/about` | `page.tsx` (static, server) | — (prose from lib docstrings) | none | none (formula blocks) |

---

## ★ Recipe: add a new tool/page (the repeating pattern)

Every page in this app was built the same way. Follow it and you'll match the codebase.

1. **Logic first (TDD).** Add a pure function to a new or existing `web/lib/<name>.ts`. Write
   `web/lib/<name>.test.ts` alongside it (Vitest). No React, no fetch — just inputs→outputs.
   If it's bond math, build on `finance.ts` primitives (`bondCashflows`, `priceFromCashflows`,
   `modifiedDuration`, `convexity`, `dv01`) so it stays numerically consistent.
2. **Data (only if external).** If the page needs Treasury/Polygon data, add or reuse a route
   under `app/api/.../route.ts`: fetch → parse (reuse `treasury.ts`/`polygon.ts`) → `revalidate`
   cache → return typed JSON. Fail soft: `catch` → `{ ...: [] }` with a 5xx status, never throw
   to the client. Add a `route.test.ts` that stubs `fetch`.
3. **Page wrapper.** `app/<route>/page.tsx`:
   ```tsx
   import { FooClient } from "./FooClient";
   import { toolMetadata } from "@/lib/tools";
   export const metadata = toolMetadata("/foo"); // title, description, canonical, link preview
   export default function FooPage() { return <FooClient />; }
   ```
4. **Client component.** `app/<route>/FooClient.tsx` (`"use client"`): hold state with
   `useState`, fetch in `useEffect` (guard against races with a `cancelled` flag — see
   `PcaClient`/`StocksClient`), derive view data in `useMemo`, render `<Card>`/`<Metric>` +
   charts. Show explicit loading / error / empty states (copy the pattern from any client).
5. **Charts.** Reuse a chart in `components/charts/`. If you need a new one, copy the shape of
   `LineChart.tsx`: `useResizeObserver` for width, `useMemo` for scales/paths, React renders the
   `<svg>`, `role="img"` + `aria-label`, colors from CSS vars (see Design System doc).
6. **Register the tool.** Add one entry to `TOOLS` in `lib/tools.ts` (href, nav label, title,
   description, group). Nav, the Home cards, the sitemap and `toolMetadata` all read it, and
   `tools.test.ts` checks the route exists.
7. **Make it shareable.** If the page has inputs, keep them in the URL: declare a module-level
   `URL_DEFAULTS` + `URL_SCHEMA` (codecs from `lib/urlState.ts`) and use
   `const [state, set, shareUrl] = useUrlState(URL_SCHEMA, URL_DEFAULTS)`. Only non-default
   values are written; invalid params fall back to defaults. Add `<CopyLinkButton getUrl={shareUrl} />`.
8. **Make it exportable.** Give each shared chart an `exportTitle` (it adds CSV / PNG buttons to
   the legend row) and put `<ExportButtons title csv={() => ({ columns, rows })} />` by each data
   table. Pass raw numbers (percent as percent, not decimals) and put units in the headers.
9. **Verify.** `npm test` (Vitest), `npx eslint .`, `npm run build` — all must stay green.

There are full worked examples of this exact flow in `docs/superpowers/specs/*` (design) and
`docs/superpowers/plans/*` (step-by-step). The two most recent (`carry-rolldown`,
`inversion-recession-signal`) are the best templates.

---

## Conventions (do these without being told)

- **Continuous compounding** is the bond-math convention everywhere in `finance.ts`
  (`e^{-y·t}`). The curve bootstrap in `curve.ts` is the *one* exception: par yields are
  **semiannual** par coupons producing **annual-compounded** zero rates. Never conflate them. Its par interpolation defaults to
  linear (Python parity); `/yield-curve` passes `interpolation: "pchip"` so forwards don't spike
  at tenor kinks.
- **Pure & tested.** Anything in `lib/` must be a pure function with a `.test.ts`. UI logic that
  can be pulled into a pure helper, should be.
- **Fail soft.** Data routes and data-driven pages degrade to "—" / "unavailable" placeholders
  rather than throwing. The Stocks page degrades to a "set `POLYGON_API_KEY`" message when no key.
- **React owns the DOM.** D3 computes (`scaleLinear`, `line`, `area`, `extent`); React renders.
  No `d3.select`/`.append`.
- **Tokens, not hex** (aspirational — see known debt). Use `var(--accent)` etc. New code should
  not add raw hex; chart series colors are the current offender.
- **Numbers are tabular.** Add `className="tabnum"` to any numeric display (monospace + `tnum`).
- **Accessibility.** Charts carry `role="img"` + a descriptive `aria-label`; spreads/changes use
  sign + text, not color alone; inputs have labels; inline links are underlined; wide tables go
  in a `ScrollRegion`. Baseline 2026-10-16: zero axe-core (WCAG 2.1 AA) violations on every route
  at 1280 and 390px. Keep it that way.
- **CSS layers.** Never add an unlayered rule to `globals.css` that a utility might need to
  override: unlayered CSS beats every Tailwind utility regardless of specificity. Use
  `@layer base` / `@layer components` (see DESIGN-SYSTEM.md).
- **No layout shift.** Charts reserve their height before measuring width; loading skeletons are
  sized like the loaded page; the page `<h1>` renders in every state (it's usually the LCP
  element). Heavy below-the-fold charts go in `LazyMount`. There is deliberately no root
  `loading.tsx` (it made static pages stream a skeleton first).
- **Secrets** only in `app/api/*` server routes (read `process.env.POLYGON_API_KEY`). Never ship
  a key to the browser. `.env*` is gitignored at both repo root and `web/`.

## Data sources & contracts

- **U.S. Treasury daily par-yield XML** — keyless. `treasury.ts` fetches per calendar year,
  strips namespaces, coerces dates, emits `BC_*` columns (`BC_10YEAR`, …). Past years cached 7d,
  current year cached 1h. A multi-year span = one fetch per year, in parallel (`spreads`, `range`).
  Dates are taken from the feed's calendar string, never round-tripped through local time.
- **Polygon.io** — daily stock bars. Needs `POLYGON_API_KEY`. Read only in the server route.
- **Tenor canonicalization** lives in `finance.ts`: `TENOR_YEARS`, `BC_TO_TENOR`. Any code that
  maps yield columns to maturities must go through these (mirrors `visualizer.py`).
- **NBER recessions** are a hardcoded list in `signal.ts` (`NBER_RECESSIONS`); extend it when a
  new recession is dated.

## Testing & tooling

- **Vitest** (`npm test`), jsdom env, Testing Library for components. 39 test files, 242 tests.
  Every `lib/*.ts` has a sibling `*.test.ts`; charts have render/smoke tests; API routes stub
  `fetch` with `vi.stubGlobal`.
- **ESLint** via `eslint.config.mjs` (`next/core-web-vitals` + `next/typescript`). Run with
  `npx eslint .` or `npm run lint` — **not** `next lint` (removed in Next 16).
- **Build**: `npm run build` (Turbopack). TypeScript `strict` is on.
- **CI**: `.github/workflows/ci.yml` runs on every push and PR. The web job runs `npm test`,
  `npx eslint .`, `npx tsc --noEmit` and `npm run build`; the Python job runs `pytest`.

## Process conventions (how features get built here)

The repo uses a spec → plan → execute flow with artifacts under `docs/superpowers/`:
`specs/<date>-<feature>-design.md` (the what/why), `plans/<date>-<feature>.md` (numbered tasks),
and `.superpowers/sdd/` (per-task briefs/reports + a progress ledger). Branches are named
`feat/…`, `fix/…`, `perf/…`; finished branches are **merged to `main` and pushed (no PR)**.

## Deploy

Vercel project, **Root Directory = `web/`**, framework preset Next.js, no required env vars
(`POLYGON_API_KEY` optional for Stocks). Custom domain via Cloudflare CNAME → `cname.vercel-dns.com`
(DNS-only). See `web/README.md`.

---

## See also

- `docs/DESIGN-SYSTEM.md` — tokens, primitives, the hardcoded-color inventory, and a playbook for
  swapping in a new design system.
- `docs/BACKLOG.md` — prioritized, agent-ready feature & improvement briefs (FEAT-*).
- `docs/BUGS.md` — known bugs & tech debt, tiered (BUG-*).
