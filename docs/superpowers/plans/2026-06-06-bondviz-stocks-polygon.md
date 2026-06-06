# BondViz Stocks (Polygon) Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/stocks` page to the Next.js web app showing Polygon-sourced daily price data (close-price chart, metrics, recent-bars table) for one or more tickers, with the API key kept server-side and a graceful "not configured" state.

**Architecture:** A pure TS lib (`web/lib/polygon.ts`) builds the Polygon REST URL and parses aggregates into typed `StockBar[]` (Vitest-tested with mocked `fetch`). A server route (`web/app/api/stocks/aggregates/route.ts`) reads `process.env.POLYGON_API_KEY`, calls the lib, and returns `{ configured, bars, error? }` so the key never reaches the browser. The client page (`StocksClient.tsx`) fetches the active ticker on demand, caches per `ticker|from|to`, and renders with the existing `LineChart`/`Card`/`Metric`/`Segmented` components.

**Tech Stack:** Next.js (App Router) + React + TypeScript + Tailwind; D3 only via the existing `LineChart`; Vitest for tests. Plain `fetch` to Polygon — no new runtime dependency. Commands run from `web/`; the `@/*` alias maps to `web/`.

> **Next.js version note:** `web/AGENTS.md` warns this Next.js differs from training-data knowledge. The route handler mirrors the existing `web/app/api/treasury/range/route.ts` (`export async function GET(req: Request)` returning `NextResponse.json(...)`), and the page mirrors the existing `web/app/pricing/page.tsx` + client-component split. Copy those patterns rather than recalling APIs.

---

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `web/lib/types.ts` | Modify | Add the `StockBar` interface |
| `web/lib/polygon.ts` | Create | `aggregatesUrl`, `parseAggregates`, `fetchAggregates` |
| `web/lib/polygon.test.ts` | Create | Lib unit tests (parse, url, fetch) |
| `web/app/api/stocks/aggregates/route.ts` | Create | Server route: key handling + fetch |
| `web/app/api/stocks/aggregates/route.test.ts` | Create | Route tests (mocked fetch + stubbed env) |
| `web/app/stocks/page.tsx` | Create | Server wrapper + metadata |
| `web/app/stocks/StocksClient.tsx` | Create | Controls, on-demand fetch+cache, render |
| `web/components/Nav.tsx` | Modify | Add the "Stocks" nav link |
| `web/README.md` | Modify | Document `POLYGON_API_KEY` |

---

## Task 1: StockBar type + Polygon library (TDD)

**Files:**
- Modify: `web/lib/types.ts`
- Create: `web/lib/polygon.ts`
- Test: `web/lib/polygon.test.ts`

- [ ] **Step 1: Add the `StockBar` type**

Append to `web/lib/types.ts`:
```ts

export interface StockBar {
  date: string; // ISO yyyy-mm-dd
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap: number | null;
}
```

- [ ] **Step 2: Write the failing test**

Create `web/lib/polygon.test.ts`:
```ts
import { describe, it, expect, vi, afterEach } from "vitest";
import { aggregatesUrl, parseAggregates, fetchAggregates } from "@/lib/polygon";

const sample = {
  ticker: "AAPL",
  results: [
    // intentionally out of order to prove sorting
    { t: 1735948800000, o: 254, h: 258, l: 253, c: 257, v: 1200000, vw: 256 },
    { t: 1735862400000, o: 250, h: 255, l: 249, c: 254, v: 1000000, vw: 252 },
  ],
  status: "OK",
};

describe("aggregatesUrl", () => {
  it("includes ticker, daily range, dates, and query params", () => {
    const url = aggregatesUrl("aapl", "2025-01-01", "2025-01-31", "KEY123");
    expect(url).toContain("/aggs/ticker/AAPL/range/1/day/2025-01-01/2025-01-31");
    expect(url).toContain("adjusted=true");
    expect(url).toContain("sort=asc");
    expect(url).toContain("apiKey=KEY123");
  });
});

describe("parseAggregates", () => {
  it("maps results to sorted StockBar[]", () => {
    const bars = parseAggregates(sample);
    expect(bars).toHaveLength(2);
    expect(bars[0].date).toBe("2025-01-03"); // earlier timestamp first
    expect(bars[0].close).toBe(254);
    expect(bars[1].date).toBe("2025-01-04");
    expect(bars[0].vwap).toBe(252);
  });

  it("returns [] for missing/empty/garbage input", () => {
    expect(parseAggregates({})).toEqual([]);
    expect(parseAggregates({ results: [] })).toEqual([]);
    expect(parseAggregates(null)).toEqual([]);
  });
});

describe("fetchAggregates", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fetches and parses on a 200", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(sample), { status: 200 })));
    const bars = await fetchAggregates("AAPL", "2025-01-01", "2025-01-31", "KEY");
    expect(bars).toHaveLength(2);
  });

  it("throws on a non-OK response", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 500 })));
    await expect(fetchAggregates("AAPL", "2025-01-01", "2025-01-31", "KEY")).rejects.toThrow();
  });
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `npx vitest run lib/polygon.test.ts`
Expected: FAIL — cannot resolve `@/lib/polygon`.

- [ ] **Step 4: Implement `polygon.ts`**

Create `web/lib/polygon.ts`:
```ts
import { StockBar } from "@/lib/types";

// Polygon aggregates (daily bars). Mirrors src/bondviz/stocks.py fetch_aggregates,
// but uses plain fetch (no SDK). The API key is supplied by the server route and
// must never be exposed to the browser.
export function aggregatesUrl(ticker: string, from: string, to: string, apiKey: string): string {
  const t = encodeURIComponent(ticker.toUpperCase());
  return (
    `https://api.polygon.io/v2/aggs/ticker/${t}/range/1/day/${from}/${to}` +
    `?adjusted=true&sort=asc&limit=50000&apiKey=${encodeURIComponent(apiKey)}`
  );
}

export function parseAggregates(json: unknown): StockBar[] {
  const results = (json as { results?: unknown } | null)?.results;
  if (!Array.isArray(results)) return [];
  const bars: StockBar[] = [];
  for (const r of results as Record<string, unknown>[]) {
    const t = r.t;
    if (typeof t !== "number") continue;
    bars.push({
      date: new Date(t).toISOString().slice(0, 10),
      open: Number(r.o),
      high: Number(r.h),
      low: Number(r.l),
      close: Number(r.c),
      volume: Number(r.v),
      vwap: r.vw === undefined || r.vw === null ? null : Number(r.vw),
    });
  }
  bars.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  return bars;
}

export async function fetchAggregates(
  ticker: string,
  from: string,
  to: string,
  apiKey: string,
): Promise<StockBar[]> {
  const res = await fetch(aggregatesUrl(ticker, from, to, apiKey), { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error(`Polygon aggregates for ${ticker} returned ${res.status}`);
  return parseAggregates(await res.json());
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run lib/polygon.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add web/lib/types.ts web/lib/polygon.ts web/lib/polygon.test.ts
git commit -m "feat(web): polygon aggregates lib (url, parse, fetch) with tests"
```
(Commit messages in this repo end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Run git from the repo root `c:\Users\Nikolay\bondviz`, or stage the explicit paths.)

---

## Task 2: Stocks API route (TDD)

**Files:**
- Create: `web/app/api/stocks/aggregates/route.ts`
- Test: `web/app/api/stocks/aggregates/route.test.ts`

- [ ] **Step 1: Write the failing test**

Create `web/app/api/stocks/aggregates/route.test.ts`:
```ts
import { describe, it, expect, vi, afterEach } from "vitest";
import { GET } from "@/app/api/stocks/aggregates/route";

const sample = JSON.stringify({
  ticker: "AAPL",
  results: [
    { t: 1735862400000, o: 250, h: 255, l: 249, c: 254, v: 1000000, vw: 252 },
    { t: 1735948800000, o: 254, h: 258, l: 253, c: 257, v: 1200000, vw: 256 },
  ],
  status: "OK",
});

const url = "http://x/api/stocks/aggregates?ticker=AAPL&from=2025-01-01&to=2025-01-31";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("/api/stocks/aggregates", () => {
  it("reports not configured when the key is missing", async () => {
    vi.stubEnv("POLYGON_API_KEY", "");
    const res = await GET(new Request(url));
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.configured).toBe(false);
  });

  it("returns bars when configured", async () => {
    vi.stubEnv("POLYGON_API_KEY", "testkey");
    vi.stubGlobal("fetch", vi.fn(async () => new Response(sample, { status: 200 })));
    const res = await GET(new Request(url));
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.configured).toBe(true);
    expect(body.bars).toHaveLength(2);
    expect(body.bars[0].close).toBe(254);
  });

  it("400s without a ticker", async () => {
    vi.stubEnv("POLYGON_API_KEY", "testkey");
    const res = await GET(new Request("http://x/api/stocks/aggregates?from=2025-01-01&to=2025-01-31"));
    expect(res.status).toBe(400);
  });

  it("502s on an upstream error", async () => {
    vi.stubEnv("POLYGON_API_KEY", "testkey");
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 500 })));
    const res = await GET(new Request(url));
    const body = await res.json();
    expect(res.status).toBe(502);
    expect(body.configured).toBe(true);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run app/api/stocks/aggregates/route.test.ts`
Expected: FAIL — cannot resolve the route module.

- [ ] **Step 3: Implement the route**

Create `web/app/api/stocks/aggregates/route.ts`:
```ts
import { NextResponse } from "next/server";
import { fetchAggregates } from "@/lib/polygon";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const ticker = searchParams.get("ticker");
  const from = searchParams.get("from");
  const to = searchParams.get("to");
  if (!ticker || !from || !to) {
    return NextResponse.json({ configured: true, bars: [] }, { status: 400 });
  }

  const apiKey = process.env.POLYGON_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ configured: false, bars: [] });
  }

  try {
    const bars = await fetchAggregates(ticker, from, to, apiKey);
    return NextResponse.json({ configured: true, bars });
  } catch (e) {
    return NextResponse.json(
      { configured: true, bars: [], error: e instanceof Error ? e.message : "fetch failed" },
      { status: 502 },
    );
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run app/api/stocks/aggregates/route.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add web/app/api/stocks
git commit -m "feat(web): stocks aggregates API route with key handling + tests"
```

---

## Task 3: Stocks page (server wrapper + client)

**Files:**
- Create: `web/app/stocks/page.tsx`
- Create: `web/app/stocks/StocksClient.tsx`

- [ ] **Step 1: Create the server page wrapper**

Create `web/app/stocks/page.tsx` (mirrors `web/app/pricing/page.tsx`):
```tsx
import { StocksClient } from "./StocksClient";

export const metadata = { title: "Stocks · BondViz" };

export default function StocksPage() {
  return <StocksClient />;
}
```

- [ ] **Step 2: Create the client component**

Create `web/app/stocks/StocksClient.tsx`:
```tsx
"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { Segmented } from "@/components/ui/Segmented";
import { LineChart, Series } from "@/components/charts/LineChart";
import { StockBar } from "@/lib/types";

interface CacheEntry {
  configured: boolean;
  bars: StockBar[];
  error?: string;
}

const LOOKBACKS = [
  { label: "1M", value: 1 },
  { label: "3M", value: 3 },
  { label: "6M", value: 6 },
  { label: "1Y", value: 12 },
];

const money = (v: number) => v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const vol = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 });
const signedPct = (v: number) => `${v >= 0 ? "+" : "−"}${(Math.abs(v) * 100).toFixed(2)}%`;

function iso(d: Date) {
  return d.toISOString().slice(0, 10);
}

function parseTickers(s: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of s.split(",")) {
    const t = raw.trim().toUpperCase();
    if (t && !seen.has(t)) {
      seen.add(t);
      out.push(t);
    }
  }
  return out;
}

export function StocksClient() {
  const [tickersInput, setTickersInput] = useState("AAPL, MSFT, GOOGL");
  const [lookback, setLookback] = useState(3);
  const [active, setActive] = useState<string | null>(null);
  const [cache, setCache] = useState<Record<string, CacheEntry>>({});

  const tickers = useMemo(() => parseTickers(tickersInput), [tickersInput]);
  const activeTicker = active && tickers.includes(active) ? active : tickers[0] ?? null;

  // Recomputed only when lookback changes, so the fetch effect has stable deps.
  const { from, to } = useMemo(() => {
    const end = new Date();
    const start = new Date();
    start.setMonth(start.getMonth() - lookback);
    return { from: iso(start), to: iso(end) };
  }, [lookback]);

  const cacheKey = activeTicker ? `${activeTicker}|${from}|${to}` : null;

  useEffect(() => {
    if (!cacheKey || !activeTicker || cache[cacheKey]) return;
    let cancelled = false;
    fetch(`/api/stocks/aggregates?ticker=${encodeURIComponent(activeTicker)}&from=${from}&to=${to}`)
      .then((r) => r.json())
      .then((d) => {
        if (!cancelled)
          setCache((prev) => ({
            ...prev,
            [cacheKey]: { configured: d.configured ?? true, bars: d.bars ?? [], error: d.error },
          }));
      })
      .catch(() => {
        if (!cancelled)
          setCache((prev) => ({ ...prev, [cacheKey]: { configured: true, bars: [], error: "Failed to load" } }));
      });
    return () => {
      cancelled = true;
    };
  }, [cacheKey, activeTicker, from, to, cache]);

  const entry = cacheKey ? cache[cacheKey] : undefined;

  const view = useMemo(() => {
    if (!entry || !entry.configured || entry.error || entry.bars.length === 0) return null;
    const bars = entry.bars;
    const first = bars[0];
    const last = bars[bars.length - 1];
    const series: Series[] = [
      {
        id: "close",
        label: "Close",
        color: "#00d68f",
        points: bars.map((b) => [new Date(b.date).getTime(), b.close] as [number, number]),
      },
    ];
    const pctChange = first.close !== 0 ? (last.close - first.close) / first.close : 0;
    const recent = [...bars].slice(-10).reverse();
    return { last, pctChange, series, recent };
  }, [entry]);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl">Stocks</h1>
      <p className="text-[var(--muted)]">Daily price history from Polygon.io.</p>

      <Card>
        <div className="grid gap-4 sm:grid-cols-[1fr_auto]">
          <label className="block">
            <span className="eyebrow">Tickers (comma-separated)</span>
            <input
              type="text"
              value={tickersInput}
              onChange={(e) => setTickersInput(e.target.value)}
              className="tabnum mt-1.5 w-full rounded-[var(--radius)] border border-[var(--panel-border)] bg-[var(--bg)] px-3 py-2.5 text-[var(--text)] outline-none focus:border-[var(--accent)]"
            />
          </label>
          <label className="block">
            <span className="eyebrow">Lookback</span>
            <div className="mt-1.5">
              <Segmented
                ariaLabel="Lookback window"
                options={LOOKBACKS}
                value={lookback}
                onChange={(v) => setLookback(v as number)}
              />
            </div>
          </label>
        </div>
        {tickers.length > 1 && (
          <div className="mt-4">
            <Segmented
              ariaLabel="Active ticker"
              options={tickers.map((t) => ({ label: t, value: t }))}
              value={activeTicker ?? tickers[0]}
              onChange={(v) => setActive(v as string)}
            />
          </div>
        )}
      </Card>

      {tickers.length === 0 && <p className="text-[var(--muted)]">Enter at least one ticker.</p>}

      {activeTicker && (
        <>
          {!entry && <p className="text-[var(--muted)]">Loading {activeTicker}…</p>}

          {entry && !entry.configured && (
            <Card>
              <h2 className="mb-2 text-lg">Polygon API key not set</h2>
              <p className="text-sm text-[var(--muted)]">
                Set <code>POLYGON_API_KEY</code> in <code>web/.env.local</code> for local development, or as a
                Vercel project environment variable, to load stock data. The rest of the app works without it.
              </p>
            </Card>
          )}

          {entry && entry.configured && entry.error && (
            <p className="text-[var(--muted)]">
              Couldn’t load {activeTicker}: {entry.error}
            </p>
          )}

          {entry && entry.configured && !entry.error && entry.bars.length === 0 && (
            <p className="text-[var(--muted)]">No data for {activeTicker} in this window (check the symbol).</p>
          )}

          {view && (
            <>
              <Card>
                <h2 className="mb-3 text-lg">{activeTicker} · close</h2>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                  <Metric label="Last close" value={money(view.last.close)} tone="accent" />
                  <Metric
                    label="Change (window)"
                    value={signedPct(view.pctChange)}
                    tone={view.pctChange >= 0 ? "pos" : "neg"}
                  />
                  <Metric label="Latest volume" value={vol(view.last.volume)} />
                </div>
                <div className="mt-4">
                  <LineChart
                    ariaLabel={`${activeTicker} daily close price`}
                    series={view.series}
                    xType="time"
                    xLabel="Date"
                    yLabel="Close"
                  />
                </div>
              </Card>

              <Card>
                <h2 className="mb-2 text-lg">Recent bars</h2>
                <div className="overflow-x-auto">
                  <table className="tabnum w-full text-sm">
                    <thead>
                      <tr className="text-left text-[var(--muted)]">
                        <th className="py-1 pr-4">Date</th>
                        <th className="py-1 pr-4">Open</th>
                        <th className="py-1 pr-4">High</th>
                        <th className="py-1 pr-4">Low</th>
                        <th className="py-1 pr-4">Close</th>
                        <th className="py-1">Volume</th>
                      </tr>
                    </thead>
                    <tbody>
                      {view.recent.map((b) => (
                        <tr key={b.date} className="border-t border-[var(--panel-border)]">
                          <td className="py-1 pr-4">{b.date}</td>
                          <td className="py-1 pr-4">{money(b.open)}</td>
                          <td className="py-1 pr-4">{money(b.high)}</td>
                          <td className="py-1 pr-4">{money(b.low)}</td>
                          <td className="py-1 pr-4">{money(b.close)}</td>
                          <td className="py-1">{vol(b.volume)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </>
          )}
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Verify the build**

Run: `npm run build`
Expected: builds with no type/lint errors; the route list includes `/stocks` and `/api/stocks/aggregates`.

- [ ] **Step 4: Commit**

```bash
git add web/app/stocks
git commit -m "feat(web): stocks page — ticker tabs, lookback, close chart, metrics, table"
```

---

## Task 4: Add Stocks to the navigation

**Files:**
- Modify: `web/components/Nav.tsx`

- [ ] **Step 1: Add the nav link**

In `web/components/Nav.tsx`, change the `LINKS` array from:
```tsx
const LINKS = [
  { href: "/yield-curve", label: "Yield Curve" },
  { href: "/pricing", label: "Bond Pricing" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/pca", label: "PCA" },
];
```
to:
```tsx
const LINKS = [
  { href: "/yield-curve", label: "Yield Curve" },
  { href: "/pricing", label: "Bond Pricing" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/pca", label: "PCA" },
  { href: "/stocks", label: "Stocks" },
];
```

- [ ] **Step 2: Verify the build**

Run: `npm run build`
Expected: builds with no errors.

- [ ] **Step 3: Commit**

```bash
git add web/components/Nav.tsx
git commit -m "feat(web): add Stocks to navigation"
```

---

## Task 5: Document POLYGON_API_KEY

**Files:**
- Modify: `web/README.md`

- [ ] **Step 1: Add a Stocks section to the README**

In `web/README.md`, immediately before the "## Deploy to Vercel" heading, insert:
```markdown
## Stocks (Polygon)

The Stocks page reads daily price data from [Polygon.io](https://polygon.io). Provide an API key:

- **Local:** create `web/.env.local` with `POLYGON_API_KEY=your_key_here` (already git-ignored).
- **Vercel:** add `POLYGON_API_KEY` as a Project Environment Variable.

The key is read only in the server route (`/api/stocks/aggregates`) and is never sent to the
browser. Without a key, the Stocks page shows a "not configured" message and the rest of the
app is unaffected.

```

- [ ] **Step 2: Commit**

```bash
git add web/README.md
git commit -m "docs(web): document POLYGON_API_KEY for the stocks page"
```

---

## Task 6: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Full test suite**

Run (from `web/`): `npm test`
Expected: all suites pass, including the new `polygon` lib tests (5) and stocks `route` tests (4). Total goes from 61 to 70.

- [ ] **Step 2: Lint + production build**

Run: `npm run lint` (expected: no errors) and `npm run build` (expected: succeeds; routes include `/stocks` and `/api/stocks/aggregates`).

- [ ] **Step 3: Manual smoke**

Run: `npm run dev`, open http://localhost:3000/stocks and confirm:
- **Without** `POLYGON_API_KEY` set: the page shows the "Polygon API key not set" card; the rest of the nav/app still works.
- **With** a valid key in `web/.env.local`: entering `AAPL, MSFT, GOOGL` shows ticker tabs; selecting a tab loads its close-price chart, the three metrics, and the recent-bars table; switching the lookback (1M/3M/6M/1Y) refetches; switching back to a previously loaded tab does not refetch (cached).
- "Stocks" appears in the nav and is highlighted when active.

- [ ] **Step 4: Final commit (only if tweaks were needed)**

```bash
git add -A
git commit -m "chore(web): stocks page verification fixes"
```

---

## Self-Review Notes

- **Spec coverage:** server-side key handling + `configured` flag (Task 2); `polygon.ts` url/parse/fetch with no SDK dependency (Task 1); `StockBar` type (Task 1); page with comma-separated tickers, `Segmented` lookback (1M/3M/6M/1Y, default 3M) and ticker tabs, on-demand fetch cached per `ticker|from|to`, close `LineChart`, metrics (last close, % change, latest volume), recent-bars table, and not-configured / loading / error / empty states (Task 3); nav entry (Task 4); README docs (Task 5); tests for lib and route with mocked fetch + stubbed env, page build-verified (Tasks 1, 2, 6). Intraday/custom-range/overlay/candlestick are out of scope per the spec.
- **Type consistency:** `StockBar` (Task 1) is consumed unchanged by `parseAggregates`/`fetchAggregates` (Task 1), the route (Task 2), and the page (Task 3). The route's response shape `{ configured: boolean; bars: StockBar[]; error?: string }` is exactly what `StocksClient`'s `CacheEntry` reads. `aggregatesUrl`/`parseAggregates`/`fetchAggregates` signatures match between lib, tests, and route. `Segmented`, `Metric`, `Card`, `LineChart`/`Series` are existing exports used as defined.
- **Lint safety:** `StocksClient`'s effect sets state only inside async callbacks (never synchronously), and `from`/`to` are memoized on `lookback` so the effect deps are stable — avoiding the `react-hooks/set-state-in-effect` and infinite-fetch pitfalls.
```
