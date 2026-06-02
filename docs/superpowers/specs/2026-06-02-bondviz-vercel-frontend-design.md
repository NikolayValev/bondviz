# BondViz — Native Vercel Front-End (MVP)

**Date:** 2026-06-02
**Status:** Approved design, pending implementation plan

## Context

BondViz currently ships as a Streamlit app (recently redesigned with a dark "quant terminal"
look, now merged to `main`). The user wants a public deployment at **`bondviz.nikolayvalev.com`**
— their domain only, no `streamlit.app` in the URL — hosted on **Vercel**, primarily to showcase
**front-end engineering skill** to recruiters.

Streamlit cannot satisfy this: it is a long-running WebSocket server, and Vercel (serverless,
HTTP-only rewrites) cannot proxy it. The chosen path is therefore a **native front-end rebuild**
on Vercel that reuses the same public data, with **no Streamlit in the hosted version**. This is
the "future front-end round" anticipated in the quant-terminal redesign spec, now brought
forward. The existing Streamlit app stays as-is on `main`.

## Goal

Ship a polished, native **Next.js** front-end on Vercel, at the user's bare domain, covering a
focused subset of BondViz, with hand-rolled **D3** charts as the centerpiece skill signal.

## Scope

**In (MVP v1):**
- **Home** (`/`) — hero, live KPI snapshot (10Y level, 2s10s, 3m10y), cards to the two tools.
- **Yield Curve explorer** (`/yield-curve`) — latest curve; curve shifts vs 1M/3M/6M/1Y ago;
  key spreads (2s10s, 3m10y) over time; short interpretive blurbs (curve shape/slope) mirroring
  the Streamlit version.
- **Bond Pricing** (`/pricing`) — face / coupon / continuous-yield / years inputs; live PV under
  continuous compounding; small discount-factor curve.

**Out (explicitly later passes):** Yield-curve PCA, Stocks (Polygon), the time×tenor heatmap, the
zero/forward bootstrapping, and FRED as a data source. The Streamlit app is unchanged.

## Non-Goals

- No changes to the Python package or the Streamlit app.
- No API keys in v1 (Treasury feed is keyless).
- No backend service beyond Next.js route handlers (no separate server, no database).

## Tech Stack

- **Next.js (App Router) + TypeScript**, deployed on **Vercel** with Root Directory `web/`.
- **Tailwind CSS** for styling (dark quant-terminal palette).
- **D3** (`d3-scale`, `d3-shape`, `d3-axis`, `d3-array`) for chart math; **React owns the SVG DOM**.
- **Vitest** for unit tests.
- Node 20, React 19, Next 15 (current Vercel defaults).

## Architecture

Greenfield app in `web/` of the existing repo. The Python project at the repo root is untouched
and excluded from the Vercel build (Root Directory = `web/`).

### Routes / pages (App Router)

| Path | Responsibility |
| --- | --- |
| `/` | Home: hero, KPI snapshot, navigation cards |
| `/yield-curve` | Yield Curve explorer (latest curve, shifts, spreads, blurbs) |
| `/pricing` | Bond Pricing calculator (live PV + discount-factor curve) |

### API route handlers (server, Node runtime, keyless)

| Path | Responsibility |
| --- | --- |
| `GET /api/treasury/latest` | Latest par-yield row as typed JSON (Home + latest curve) |
| `GET /api/treasury/range?start=YYYY-MM-DD&end=YYYY-MM-DD` | Merged daily rows across the needed years (shifts + spread series) |

Both fetch the **U.S. Treasury daily par-yield XML** and parse it server-side (a TS port of
`treasury.py`: namespace stripping, `BC_*` columns, date coercion). Responses use
`fetch(..., { next: { revalidate: 3600 } })` (data is daily) so the Treasury feed is hit at most
hourly. Fetching server-side avoids browser CORS issues.

### Library modules (`web/lib/`, pure & tested)

- `finance.ts` — `pvContinuous(face, coupon, yield, years)`, `discountFactors(yield, tenors)`,
  `computeCurveKpis(row)` → `{ "10Y", "2s10s", "3m10y" }`, spread-series helpers, and the tenor
  maps (`BC_*` → label, label → years). Direct TS ports of `pricing.py` / `app_logic.py` /
  `visualizer.py` constants. **Unit-tested.**
- `treasury.ts` — fetch + parse Treasury XML into typed records. **Parsing unit-tested against a
  saved XML fixture.**
- `types.ts` — shared types (`YieldRow`, `CurvePoint`, `Kpis`, `TenorLabel`).

### Components (`web/components/`)

- Layout: top nav (Home · Yield Curve · Bond Pricing) + page shell.
- UI: `Card`, `Kpi`, `InterpretationNote`.
- Charts (hand-rolled D3 + React, themed dark, responsive via a `useResizeObserver` hook):
  - `YieldCurveChart` — yield vs maturity, with optional overlaid comparison curves (shifts).
  - `SpreadChart` — 2s10s / 3m10y time series with a zero baseline.
  - `DiscountCurveChart` — discount factor vs years (pricing page).
  - Shared chart primitives: `Axis`, `GridLines`, `Tooltip`, `useResizeObserver`.

### Styling

Tailwind config encodes the palette as CSS variables / theme tokens matching `theme.py`:
`--bg #0a0e14`, `--panel #131722`, `--accent #00d68f`, `--text #e6e6e6`, `--muted #8b95a7`.
Monospace, tabular-nums for all numerics. Accessibility: WCAG-AA contrast, semantic headings,
SVG charts carry `<title>`/`aria-label`, no color-only signaling (spreads keep +/- and labels).

## Data Flow

1. Page (server component) or client hook calls `/api/treasury/*`.
2. Route handler fetches + parses Treasury XML (cached hourly) → typed JSON.
3. `finance.ts` derives KPIs / spreads / curve points from rows.
4. D3 chart components render SVG from those values.
5. Bond Pricing is fully client-side (pure `finance.ts` math, live on input change).

## Error Handling

- Treasury fetch/parse failure → route handler returns `503` with an empty payload.
- UI degrades gracefully: KPIs render "—", charts show a friendly "data unavailable" placeholder
  (same fail-soft spirit as the Streamlit `home_view`).

## Testing

- **Vitest** unit tests for `finance.ts` (PV math incl. zero-yield edge case; `computeCurveKpis`
  incl. missing/NaN → null; spread math) and `treasury.ts` (parse a fixture XML → expected rows;
  missing-date handling).
- Build verification: `next build` succeeds; `next lint` clean.
- Manual: run `next dev`, confirm all three pages render with live Treasury data and charts.

## Deployment & Domain

- Vercel project, **Root Directory `web/`**, framework preset Next.js, no env vars.
- Custom domain `bondviz.nikolayvalev.com`: add in Vercel → Vercel returns a
  `cname.vercel-dns.com` target → add **CNAME `bondviz` → `cname.vercel-dns.com`** in Cloudflare
  set to **DNS-only (grey cloud)**; Vercel provisions the TLS cert. Result: clean URL on the
  user's domain, no Streamlit.
- The Vercel deploy can be driven via the assistant's Vercel integration if the user connects it;
  the single Cloudflare DNS record is added by the user.

## Risks & Mitigations

- **Treasury XML quirks** (namespace changes, missing date tags) — port the existing tolerant
  parsing from `treasury.py`; cover with a fixture test.
- **Multi-year range fetch latency** (shifts/spreads need several years) — fetch only the years
  spanned by the requested range; cache hourly; render incrementally / show a loading state.
- **D3 + React DOM ownership** — keep React as the single DOM owner (render from D3-computed
  values; no `d3.select` mutation) to avoid reconciliation bugs.
- **Cloudflare ↔ Vercel TLS** — use DNS-only (grey cloud) so Vercel manages the certificate.

## Success Criteria

- `bondviz.nikolayvalev.com` serves the Next.js app over HTTPS, no `streamlit.app` in the URL.
- Home shows live KPIs (or graceful "—"); Yield Curve shows latest curve + shifts + spreads with
  D3 charts and interpretive text; Bond Pricing updates PV live.
- All charts are hand-rolled D3 + React, dark-themed, responsive.
- `finance.ts` and `treasury.ts` covered by passing Vitest tests; `next build` clean.
- Streamlit app on `main` remains unchanged and functional.

## Future (later passes)

PCA factors, Stocks (Polygon via a Vercel serverless function with `POLYGON_API_KEY`), the
heatmap, zero/forward bootstrapping, and FRED as an additional data source.
