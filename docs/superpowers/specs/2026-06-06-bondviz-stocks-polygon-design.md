# BondViz Stocks (Polygon) Page — Design Spec

**Date:** 2026-06-06
**Status:** Approved (design); pending implementation plan.

## Goal

Add a `/stocks` page to the Next.js web app (`web/`) showing Polygon-sourced daily price
data for one or more tickers. This is **Spec B** of the two-part effort to close the gap with
the original Streamlit app; the curve-analytics trio (Spec A) is already merged. Unlike Spec A,
this feature introduces a new external data source (Polygon.io) and a server-side secret
(`POLYGON_API_KEY`), so the defining concerns are key handling and graceful degradation when
no key is configured.

## Principles & conventions

- **The API key never reaches the browser.** All Polygon calls happen in a server route
  handler that reads `process.env.POLYGON_API_KEY`. The client only ever calls our own
  `/api/stocks/...` route.
- **No new runtime dependency.** Call Polygon's REST endpoint with plain `fetch` (the same
  approach `web/lib/treasury.ts` uses for the Treasury feed) — not the Polygon SDK.
- **Pure logic in `web/lib/`**, unit-tested with Vitest; rendering reuses existing components
  (`LineChart`, `Card`, `Metric`, `Segmented`).
- The `@/*` import alias maps to `web/`. All commands run from `web/`.
- Follow the established page pattern: a server `page.tsx` wrapper (exports `metadata`, renders
  the client component) + a `"use client"` `StocksClient.tsx`.

## Data flow & key handling

**`web/lib/polygon.ts`** (mirrors the structure of `web/lib/treasury.ts`):

- `aggregatesUrl(ticker, from, to, apiKey)` — builds the Polygon URL:
  `https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/1/day/{from}/{to}?adjusted=true&sort=asc&limit=50000&apiKey={apiKey}`
  (`from`/`to` are ISO `yyyy-mm-dd`; ticker upper-cased).
- `parseAggregates(json): StockBar[]` — maps Polygon's `results` array
  (`{ t, o, h, l, c, v, vw }`, `t` in ms) into `StockBar[]`, sorted ascending by date.
  Returns `[]` for missing/empty/garbage `results`.
- `fetchAggregates(ticker, from, to, apiKey): Promise<StockBar[]>` — fetches the URL with
  `{ next: { revalidate: 3600 } }`; throws on a non-OK response; otherwise returns
  `parseAggregates(await res.json())`.

**`web/lib/types.ts`** — add:
```ts
export interface StockBar {
  date: string;   // ISO yyyy-mm-dd
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap: number | null;
}
```

**`web/app/api/stocks/aggregates/route.ts`** — `GET(req)`:
- Reads `ticker`, `from`, `to` from the query string.
- If `ticker` is missing → `400 { bars: [] }`.
- Resolves `process.env.POLYGON_API_KEY`. If absent/empty → `200 { configured: false, bars: [] }`
  (this is the signal the page uses to show the "not configured" state; it is NOT an error).
- Otherwise calls `fetchAggregates(...)`; on success → `200 { configured: true, bars }`.
- On an upstream/parse failure → `502 { configured: true, bars: [], error: "…" }`.

## Page (`web/app/stocks/page.tsx` + `web/app/stocks/StocksClient.tsx`)

- **Controls:**
  - A comma-separated **ticker input** (default `AAPL, MSFT, GOOGL`), parsed into an
    upper-cased, de-duplicated list of symbols.
  - A **lookback** `Segmented`: 1M / 3M / 6M / 1Y (default 3M). Maps to a `from` date =
    today minus the window; `to` = today.
  - A **ticker-tab** `Segmented` listing the parsed symbols; selecting one sets the active
    ticker. If the input changes and the active ticker is no longer present, the active ticker
    resets to the first symbol.
- **Fetching:** the active ticker is fetched **on demand** from
  `/api/stocks/aggregates?ticker=…&from=…&to=…`. Results are **cached client-side** keyed by
  `ticker|from|to`, so re-selecting a previously loaded tab (or returning to it) does not
  refetch — this also keeps within Polygon's free-tier rate limit. Stale in-flight responses
  for a no-longer-active key are ignored.
- **Render (active ticker):**
  - A close-price **`LineChart`** (x = time, y = close), one series.
  - **`Metric`** cards: last close, % change over the window
    (`(lastClose − firstClose)/firstClose`), and latest daily volume.
  - A **recent-bars table**: the last ~10 bars (most recent first) with columns
    date, open, high, low, close, volume.
- **States:**
  - **Not configured** (`configured: false`): a friendly `Card` explaining that
    `POLYGON_API_KEY` must be set (local `.env.local` / Vercel project env var); no error styling.
  - **Loading**, **error** (upstream failure / non-OK), and **empty** (no bars, e.g. an
    invalid ticker) — each a short muted message, consistent with the other pages.

## Nav

Add **"Stocks"** to `web/components/Nav.tsx`:
Yield Curve · Bond Pricing · Portfolio · PCA · Stocks.

## Testing

- **`web/lib/polygon.test.ts`:**
  - `parseAggregates` on a small JSON fixture → correct `StockBar[]` (dates from ms, sorted).
  - `parseAggregates` on `{}` / `{ results: [] }` → `[]`.
  - `aggregatesUrl` includes ticker (upper-cased), the `1/day` range, `from`/`to`, `adjusted`,
    `sort`, and `apiKey`.
  - `fetchAggregates` with mocked `fetch`: OK response → parsed bars; non-OK → throws.
- **`web/app/api/stocks/aggregates/route.test.ts`** (mocked `fetch`, env stubbed):
  - No `POLYGON_API_KEY` → `200 { configured: false }`.
  - Key set + mocked OK fetch → `200 { configured: true, bars: [...] }`.
  - Missing `ticker` → `400`.
  - Key set + mocked non-OK upstream → `502 { configured: true, error }`.
- The page is verified by `npm run build` + manual smoke. No live key is used in tests.

## Docs

- Add a short note to `web/README.md`: set `POLYGON_API_KEY` in `web/.env.local` for local dev
  and as a Vercel project environment variable for deployment; without it the Stocks page shows
  the not-configured state and the rest of the app is unaffected.

## Out of scope

- Intraday (hour/minute) timespans, custom date-range pickers, multi-ticker overlay charts,
  candlestick/volume-bar charts, snapshots/previous-close endpoints, and watchlist persistence.
  Daily bars with a preset lookback are sufficient for this page and friendly to the free tier.
