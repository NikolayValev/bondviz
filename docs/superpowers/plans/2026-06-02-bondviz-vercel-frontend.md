# BondViz Vercel Front-End (MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native Next.js + TypeScript + D3 front-end (Home, Yield Curve explorer, Bond Pricing) in `web/`, deployable to Vercel at `bondviz.nikolayvalev.com`, reusing the keyless U.S. Treasury par-yield feed.

**Architecture:** Greenfield Next.js (App Router) app in `web/` of the existing repo (Vercel Root Directory = `web/`; the Python app at the root is untouched). Server route handlers fetch and parse the Treasury XML feed (cached hourly) into typed JSON; pure TS libs (`finance.ts`, `treasury.ts`) hold the ported math and parsing and are unit-tested with Vitest; pages render hand-rolled D3 charts where React owns the SVG DOM.

**Tech Stack:** Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, D3 (`d3-scale`/`d3-shape`/`d3-array`), `fast-xml-parser`, Vitest + Testing Library + jsdom. Node 20.

---

## File Structure

| File | Responsibility | Status |
| --- | --- | --- |
| `web/` (Next scaffold) | Next.js app root; Vercel builds from here | Create |
| `web/vitest.config.ts`, `web/vitest.setup.ts` | Test runner config (jsdom, `@` alias, jest-dom) | Create |
| `web/app/globals.css` | Dark quant-terminal tokens + base styles | Modify |
| `web/tailwind.config.ts` | Palette tokens, monospace numerics | Modify |
| `web/app/layout.tsx` | Root layout + `<Nav>` + page shell | Modify |
| `web/components/Nav.tsx` | Top navigation (Home · Yield Curve · Bond Pricing) | Create |
| `web/components/ui/Card.tsx`, `Kpi.tsx` | Presentational card + KPI tiles | Create |
| `web/lib/types.ts` | Shared types (`YieldRow`, `CurvePoint`, `Kpis`, `TenorLabel`) | Create |
| `web/lib/finance.ts` | Ported math: PV, KPIs, spreads, tenor maps, curve helpers | Create |
| `web/lib/treasury.ts` | Fetch + parse Treasury XML → `YieldRow[]` | Create |
| `web/app/api/treasury/latest/route.ts` | Latest par-yield row | Create |
| `web/app/api/treasury/range/route.ts` | Daily rows across a date range | Create |
| `web/components/charts/useResizeObserver.ts` | Responsive-size hook | Create |
| `web/components/charts/LineChart.tsx` | Generic D3 multi-series line chart (React-owned SVG) | Create |
| `web/app/page.tsx` | Home: hero, KPI snapshot, tool cards | Modify |
| `web/app/yield-curve/page.tsx` + `YieldCurveClient.tsx` | Yield Curve explorer | Create |
| `web/app/pricing/page.tsx` + `PricingClient.tsx` | Bond Pricing calculator | Create |
| `web/test/fixtures/treasury-sample.xml` | XML fixture for parser test | Create |
| `web/README.md` | Run + deploy (Vercel + Cloudflare) docs | Create |

**Testing approach:** Pure logic (`finance.ts`, `treasury.ts`) and route handlers are unit-tested with Vitest (mocked `fetch`). One render test proves `LineChart` emits SVG paths. Pages are verified by `next build` + manual `next dev`. Run a single test with `npx vitest run path -t "name"`.

**Conventions for every task:** all commands run from `web/` unless noted. Use `npm`. The `@/*` import alias maps to `web/`.

---

## Task 1: Scaffold the Next.js app and test runner

**Files:** create `web/` (scaffold), `web/vitest.config.ts`, `web/vitest.setup.ts`; modify `web/package.json`.

- [ ] **Step 1: Scaffold Next.js into `web/`**

From the repo root (`c:\Users\Nikolay\bondviz`):
```bash
npx create-next-app@latest web --ts --tailwind --eslint --app --no-src-dir --import-alias "@/*" --use-npm --yes
```
If prompted about Turbopack, accept the default (Yes). This creates `web/app`, `web/package.json`, `web/tsconfig.json`, `web/next.config.ts`, `web/tailwind.config.ts` (or `postcss`), `web/app/globals.css`.

- [ ] **Step 2: Install runtime + dev dependencies**

```bash
cd web
npm install d3-scale d3-shape d3-array fast-xml-parser
npm install -D @types/d3-scale @types/d3-shape @types/d3-array vitest @vitejs/plugin-react jsdom @testing-library/react @testing-library/dom @testing-library/jest-dom
```

- [ ] **Step 3: Add Vitest config**

Create `web/vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
  },
  resolve: {
    alias: { "@": fileURLToPath(new URL("./", import.meta.url)) },
  },
});
```

Create `web/vitest.setup.ts`:
```ts
import "@testing-library/jest-dom";
```

- [ ] **Step 4: Add test scripts to `web/package.json`**

In `web/package.json`, add to the `"scripts"` object:
```json
"test": "vitest run",
"test:watch": "vitest"
```

- [ ] **Step 5: Add a smoke test**

Create `web/test/smoke.test.ts`:
```ts
import { describe, it, expect } from "vitest";

describe("smoke", () => {
  it("runs", () => {
    expect(1 + 1).toBe(2);
  });
});
```

- [ ] **Step 6: Verify tests and build**

Run: `npm test`
Expected: 1 passed.

Run: `npm run build`
Expected: Next build completes with no errors.

- [ ] **Step 7: Commit**

```bash
cd ..
git add web .gitignore
git commit -m "feat(web): scaffold Next.js app with Vitest"
```
(The scaffold writes `web/.gitignore` for `node_modules`/`.next`; confirm `web/node_modules` is NOT staged.)

---

## Task 2: Dark theme tokens, layout shell, and nav

**Files:** modify `web/app/globals.css`, `web/app/layout.tsx`; create `web/components/Nav.tsx`, `web/components/ui/Card.tsx`, `web/components/ui/Kpi.tsx`.

- [ ] **Step 1: Replace `web/app/globals.css` with the dark palette**

Replace the entire contents of `web/app/globals.css` with:
```css
@import "tailwindcss";

:root {
  --bg: #0a0e14;
  --panel: #131722;
  --panel-border: rgba(255, 255, 255, 0.08);
  --accent: #00d68f;
  --text: #e6e6e6;
  --muted: #8b95a7;
  --grid: rgba(255, 255, 255, 0.10);
}

html, body {
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, sans-serif;
}

.tabnum {
  font-variant-numeric: tabular-nums;
  font-family: "SFMono-Regular", "Consolas", "Roboto Mono", monospace;
}

h1, h2, h3 { letter-spacing: 0.01em; }

a { color: inherit; }
```

- [ ] **Step 2: Create the Card and Kpi components**

Create `web/components/ui/Card.tsx`:
```tsx
import { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-lg border border-[var(--panel-border)] border-l-[3px] border-l-[var(--accent)] bg-[var(--panel)] p-5 ${className}`}
    >
      {children}
    </div>
  );
}
```

Create `web/components/ui/Kpi.tsx`:
```tsx
export function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-sm text-[var(--muted)]">{label}</div>
      <div className="tabnum text-2xl text-[var(--accent)]">{value}</div>
    </div>
  );
}
```

- [ ] **Step 3: Create the Nav component**

Create `web/components/Nav.tsx`:
```tsx
import Link from "next/link";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/yield-curve", label: "Yield Curve" },
  { href: "/pricing", label: "Bond Pricing" },
];

export function Nav() {
  return (
    <header className="border-b border-[var(--panel-border)]">
      <nav className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-4">
        <Link href="/" className="font-semibold tracking-wide text-[var(--accent)]">
          BONDVIZ
        </Link>
        <div className="flex gap-5 text-sm text-[var(--muted)]">
          {LINKS.slice(1).map((l) => (
            <Link key={l.href} href={l.href} className="hover:text-[var(--text)]">
              {l.label}
            </Link>
          ))}
        </div>
      </nav>
    </header>
  );
}
```

- [ ] **Step 4: Wire the layout**

Replace `web/app/layout.tsx` with:
```tsx
import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";

export const metadata: Metadata = {
  title: "BondViz",
  description: "Fixed-income & markets research terminal",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
```

- [ ] **Step 5: Verify build**

Run: `npm run build`
Expected: builds with no errors. (The default `web/app/page.tsx` still exists; that's fine.)

- [ ] **Step 6: Commit**

```bash
cd ..
git add web
git commit -m "feat(web): dark theme tokens, layout shell, and nav"
cd web
```

---

## Task 3: Types and finance lib (TDD)

**Files:** create `web/lib/types.ts`, `web/lib/finance.ts`, `web/lib/finance.test.ts`.

- [ ] **Step 1: Create the shared types**

Create `web/lib/types.ts`:
```ts
export const TENOR_LABELS = [
  "1M", "2M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y",
] as const;
export type TenorLabel = (typeof TENOR_LABELS)[number];

export interface YieldRow {
  date: string; // ISO yyyy-mm-dd
  [bcColumn: string]: string | number | null;
}

export interface CurvePoint {
  label: TenorLabel;
  years: number;
  yield: number; // percent
}

export interface Kpis {
  tenYear: number | null; // percent
  twos10s: number | null; // percentage points (10Y - 2Y)
  threeM10Y: number | null; // percentage points (10Y - 3M)
}
```

- [ ] **Step 2: Write the failing test**

Create `web/lib/finance.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import {
  pvContinuous,
  discountFactors,
  computeCurveKpis,
  rowToCurve,
  spreadSeries,
} from "@/lib/finance";

describe("pvContinuous", () => {
  it("matches the closed form for a normal yield", () => {
    const pv = pvContinuous(1000, 0.05, 0.04, 10);
    // C*(1-e^-rT)/r + F*e^-rT
    const expected = (50 * (1 - Math.exp(-0.4))) / 0.04 + 1000 * Math.exp(-0.4);
    expect(pv).toBeCloseTo(expected, 6);
  });

  it("handles the zero-yield limit", () => {
    expect(pvContinuous(1000, 0.05, 0, 10)).toBeCloseTo(50 * 10 + 1000, 6);
  });
});

describe("discountFactors", () => {
  it("returns e^-rt per tenor", () => {
    const dfs = discountFactors(0.04, [0, 1, 2]);
    expect(dfs[0]).toEqual({ t: 0, df: 1 });
    expect(dfs[2].df).toBeCloseTo(Math.exp(-0.08), 6);
  });
});

describe("computeCurveKpis", () => {
  it("computes 10Y, 2s10s, 3m10y", () => {
    const k = computeCurveKpis({ BC_3MONTH: 5.0, BC_2YEAR: 4.0, BC_10YEAR: 4.5 });
    expect(k.tenYear).toBe(4.5);
    expect(k.twos10s).toBeCloseTo(0.5, 9);
    expect(k.threeM10Y).toBeCloseTo(-0.5, 9);
  });

  it("returns null for missing/NaN inputs", () => {
    const k = computeCurveKpis({ BC_10YEAR: 4.5 });
    expect(k.tenYear).toBe(4.5);
    expect(k.twos10s).toBeNull();
    expect(k.threeM10Y).toBeNull();
  });
});

describe("rowToCurve", () => {
  it("builds sorted curve points from BC_* columns", () => {
    const pts = rowToCurve({ date: "2025-01-02", BC_10YEAR: 4.5, BC_2YEAR: 4.0, BC_1MONTH: 5.2 });
    expect(pts.map((p) => p.label)).toEqual(["1M", "2Y", "10Y"]);
    expect(pts[0].years).toBeCloseTo(1 / 12, 9);
  });
});

describe("spreadSeries", () => {
  it("builds 2s10s and 3m10y time series in pp", () => {
    const { twos10s, threeM10Y } = spreadSeries([
      { date: "2025-01-02", BC_3MONTH: 5.0, BC_2YEAR: 4.0, BC_10YEAR: 4.5 },
    ]);
    expect(twos10s[0][1]).toBeCloseTo(0.5, 9);
    expect(threeM10Y[0][1]).toBeCloseTo(-0.5, 9);
  });
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `npx vitest run lib/finance.test.ts`
Expected: FAIL — cannot resolve `@/lib/finance`.

- [ ] **Step 4: Implement `finance.ts`**

Create `web/lib/finance.ts`:
```ts
import { CurvePoint, Kpis, TenorLabel, YieldRow } from "@/lib/types";

export const TENOR_YEARS: Record<TenorLabel, number> = {
  "1M": 1 / 12, "2M": 2 / 12, "3M": 3 / 12, "6M": 6 / 12,
  "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30,
};

export const BC_TO_TENOR: Record<string, TenorLabel> = {
  BC_1MONTH: "1M", BC_2MONTH: "2M", BC_3MONTH: "3M", BC_6MONTH: "6M",
  BC_1YEAR: "1Y", BC_2YEAR: "2Y", BC_3YEAR: "3Y", BC_5YEAR: "5Y",
  BC_7YEAR: "7Y", BC_10YEAR: "10Y", BC_20YEAR: "20Y", BC_30YEAR: "30Y",
};

export function pvContinuous(face: number, coupon: number, yieldRate: number, years: number): number {
  const C = face * coupon;
  if (yieldRate === 0) return C * years + face;
  const d = Math.exp(-yieldRate * years);
  return (C * (1 - d)) / yieldRate + face * d;
}

export function discountFactors(yieldRate: number, tenors: number[]): { t: number; df: number }[] {
  return tenors.map((t) => ({ t, df: Math.exp(-yieldRate * t) }));
}

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && !Number.isNaN(v) ? v : null;
}

export function computeCurveKpis(row: Record<string, unknown>): Kpis {
  const y10 = numOrNull(row.BC_10YEAR);
  const y2 = numOrNull(row.BC_2YEAR);
  const y3m = numOrNull(row.BC_3MONTH);
  return {
    tenYear: y10,
    twos10s: y10 !== null && y2 !== null ? y10 - y2 : null,
    threeM10Y: y10 !== null && y3m !== null ? y10 - y3m : null,
  };
}

export function rowToCurve(row: Record<string, unknown>): CurvePoint[] {
  const pts: CurvePoint[] = [];
  for (const [bc, label] of Object.entries(BC_TO_TENOR)) {
    const v = numOrNull(row[bc]);
    if (v !== null) pts.push({ label, years: TENOR_YEARS[label], yield: v });
  }
  return pts.sort((a, b) => a.years - b.years);
}

export function spreadSeries(rows: YieldRow[]): {
  twos10s: [number, number][];
  threeM10Y: [number, number][];
} {
  const twos10s: [number, number][] = [];
  const threeM10Y: [number, number][] = [];
  for (const r of rows) {
    const t = new Date(r.date).getTime();
    const y10 = numOrNull(r.BC_10YEAR);
    const y2 = numOrNull(r.BC_2YEAR);
    const y3m = numOrNull(r.BC_3MONTH);
    if (y10 !== null && y2 !== null) twos10s.push([t, y10 - y2]);
    if (y10 !== null && y3m !== null) threeM10Y.push([t, y10 - y3m]);
  }
  return { twos10s, threeM10Y };
}

export function describeCurve(curve: CurvePoint[]): string {
  const front = curve.find((p) => ["3M", "6M", "1Y", "2Y"].includes(p.label));
  const long = [...curve].reverse().find((p) => ["10Y", "20Y", "30Y"].includes(p.label));
  if (!front || !long) return "Incomplete tenor coverage in this snapshot.";
  const slope = long.yield - front.yield;
  if (slope < -0.1) return `Inverted: ${long.label} (${long.yield.toFixed(2)}%) below ${front.label} (${front.yield.toFixed(2)}%).`;
  if (slope < 0.1) return `Roughly flat between ${front.label} and ${long.label}.`;
  return `Upward sloping: ${long.label} (${long.yield.toFixed(2)}%) above ${front.label} (${front.yield.toFixed(2)}%).`;
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run lib/finance.test.ts`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd ..
git add web/lib
git commit -m "feat(web): ported finance lib with tests"
cd web
```

---

## Task 4: Treasury fetch + parse (TDD with fixture)

**Files:** create `web/test/fixtures/treasury-sample.xml`, `web/lib/treasury.ts`, `web/lib/treasury.test.ts`.

- [ ] **Step 1: Create the XML fixture**

Create `web/test/fixtures/treasury-sample.xml`:
```xml
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2025-01-02T00:00:00</d:NEW_DATE>
        <d:BC_3MONTH>5.00</d:BC_3MONTH>
        <d:BC_2YEAR>4.00</d:BC_2YEAR>
        <d:BC_10YEAR>4.50</d:BC_10YEAR>
      </m:properties>
    </content>
  </entry>
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2025-01-03T00:00:00</d:NEW_DATE>
        <d:BC_3MONTH>5.01</d:BC_3MONTH>
        <d:BC_2YEAR>4.05</d:BC_2YEAR>
        <d:BC_10YEAR>4.55</d:BC_10YEAR>
      </m:properties>
    </content>
  </entry>
</feed>
```

- [ ] **Step 2: Write the failing test**

Create `web/lib/treasury.test.ts`:
```ts
import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseTreasuryXml, fetchTreasuryYear } from "@/lib/treasury";

const xml = readFileSync(
  fileURLToPath(new URL("../test/fixtures/treasury-sample.xml", import.meta.url)),
  "utf-8",
);

describe("parseTreasuryXml", () => {
  it("extracts sorted rows with numeric BC_* fields", () => {
    const rows = parseTreasuryXml(xml);
    expect(rows).toHaveLength(2);
    expect(rows[0].date).toBe("2025-01-02");
    expect(rows[0].BC_10YEAR).toBe(4.5);
    expect(rows[1].date).toBe("2025-01-03");
  });

  it("returns [] for empty/garbage input", () => {
    expect(parseTreasuryXml("<feed></feed>")).toEqual([]);
  });
});

describe("fetchTreasuryYear", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fetches and parses the year feed", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(xml, { status: 200 })));
    const rows = await fetchTreasuryYear(2025);
    expect(rows).toHaveLength(2);
  });

  it("throws on non-OK responses", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 500 })));
    await expect(fetchTreasuryYear(2025)).rejects.toThrow();
  });
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `npx vitest run lib/treasury.test.ts`
Expected: FAIL — cannot resolve `@/lib/treasury`.

- [ ] **Step 4: Implement `treasury.ts`**

Create `web/lib/treasury.ts`:
```ts
import { XMLParser } from "fast-xml-parser";
import { YieldRow } from "@/lib/types";

const YIELD_COLS = [
  "BC_1MONTH", "BC_2MONTH", "BC_3MONTH", "BC_6MONTH", "BC_1YEAR", "BC_2YEAR",
  "BC_3YEAR", "BC_5YEAR", "BC_7YEAR", "BC_10YEAR", "BC_20YEAR", "BC_30YEAR",
];

const TREASURY_URL = (year: number) =>
  "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml" +
  `?data=daily_treasury_yield_curve&field_tdr_date_value=${year}`;

export function parseTreasuryXml(xml: string): YieldRow[] {
  const parser = new XMLParser({ ignoreAttributes: true, removeNSPrefix: true });
  const doc = parser.parse(xml);
  const feed = doc?.feed;
  if (!feed?.entry) return [];
  const entries = Array.isArray(feed.entry) ? feed.entry : [feed.entry];

  const rows: YieldRow[] = [];
  for (const entry of entries) {
    const props = entry?.content?.properties;
    if (!props) continue;
    const rawDate = props.NEW_DATE ?? props.DATE ?? props.RecordDate;
    if (!rawDate) continue;
    const d = new Date(rawDate);
    if (Number.isNaN(d.getTime())) continue;

    const row: YieldRow = { date: d.toISOString().slice(0, 10) };
    for (const col of YIELD_COLS) {
      const v = props[col];
      row[col] = v === undefined || v === null || v === "" ? null : Number(v);
    }
    rows.push(row);
  }
  rows.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  return rows;
}

export async function fetchTreasuryYear(year: number): Promise<YieldRow[]> {
  const res = await fetch(TREASURY_URL(year), { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error(`Treasury feed ${year} returned ${res.status}`);
  return parseTreasuryXml(await res.text());
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run lib/treasury.test.ts`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd ..
git add web/lib web/test
git commit -m "feat(web): treasury XML fetch + parse with fixture test"
cd web
```

---

## Task 5: API route handlers (TDD with mocked fetch)

**Files:** create `web/app/api/treasury/latest/route.ts`, `web/app/api/treasury/range/route.ts`, `web/app/api/treasury/route.test.ts`.

- [ ] **Step 1: Write the failing test**

Create `web/app/api/treasury/route.test.ts`:
```ts
import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { GET as latestGET } from "@/app/api/treasury/latest/route";
import { GET as rangeGET } from "@/app/api/treasury/range/route";

const xml = readFileSync(
  fileURLToPath(new URL("../../../test/fixtures/treasury-sample.xml", import.meta.url)),
  "utf-8",
);

afterEach(() => vi.restoreAllMocks());

describe("/api/treasury/latest", () => {
  it("returns the last row", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(xml, { status: 200 })));
    const res = await latestGET();
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.row.date).toBe("2025-01-03");
  });

  it("503s when the feed fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 500 })));
    const res = await latestGET();
    expect(res.status).toBe(503);
  });
});

describe("/api/treasury/range", () => {
  it("returns rows filtered to the requested window", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(xml, { status: 200 })));
    const req = new Request("http://x/api/treasury/range?start=2025-01-03&end=2025-01-03");
    const res = await rangeGET(req);
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.rows).toHaveLength(1);
    expect(body.rows[0].date).toBe("2025-01-03");
  });

  it("400s without start/end", async () => {
    const req = new Request("http://x/api/treasury/range");
    const res = await rangeGET(req);
    expect(res.status).toBe(400);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run app/api/treasury/route.test.ts`
Expected: FAIL — cannot resolve the route modules.

- [ ] **Step 3: Implement the `latest` route**

Create `web/app/api/treasury/latest/route.ts`:
```ts
import { NextResponse } from "next/server";
import { fetchTreasuryYear } from "@/lib/treasury";

export async function GET() {
  try {
    const year = new Date().getUTCFullYear();
    let rows = await fetchTreasuryYear(year);
    if (rows.length === 0) rows = await fetchTreasuryYear(year - 1);
    if (rows.length === 0) return NextResponse.json({ row: null }, { status: 503 });
    return NextResponse.json({ row: rows[rows.length - 1] });
  } catch {
    return NextResponse.json({ row: null }, { status: 503 });
  }
}
```

- [ ] **Step 4: Implement the `range` route**

Create `web/app/api/treasury/range/route.ts`:
```ts
import { NextResponse } from "next/server";
import { fetchTreasuryYear } from "@/lib/treasury";
import { YieldRow } from "@/lib/types";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const start = searchParams.get("start");
  const end = searchParams.get("end");
  if (!start || !end) return NextResponse.json({ rows: [] }, { status: 400 });

  try {
    const y0 = new Date(start).getUTCFullYear();
    const y1 = new Date(end).getUTCFullYear();
    const all: YieldRow[] = [];
    for (let y = y0; y <= y1; y++) {
      const rows = await fetchTreasuryYear(y).catch(() => [] as YieldRow[]);
      all.push(...rows);
    }
    const rows = all.filter((r) => r.date >= start && r.date <= end);
    return NextResponse.json({ rows });
  } catch {
    return NextResponse.json({ rows: [] }, { status: 503 });
  }
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run app/api/treasury/route.test.ts`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd ..
git add web/app/api
git commit -m "feat(web): treasury latest + range API routes with tests"
cd web
```

---

## Task 6: Responsive hook + generic D3 LineChart

**Files:** create `web/components/charts/useResizeObserver.ts`, `web/components/charts/LineChart.tsx`, `web/components/charts/LineChart.test.tsx`.

- [ ] **Step 1: Create the resize hook**

Create `web/components/charts/useResizeObserver.ts`:
```ts
"use client";
import { useEffect, useRef, useState } from "react";

export function useResizeObserver<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      setWidth(entries[0].contentRect.width);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return { ref, width };
}
```

- [ ] **Step 2: Write the failing render test**

Create `web/components/charts/LineChart.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { LineChart } from "@/components/charts/LineChart";

describe("LineChart", () => {
  it("renders one path per series when given explicit dimensions", () => {
    const { container } = render(
      <LineChart
        ariaLabel="test chart"
        width={400}
        height={200}
        series={[
          { id: "a", label: "A", color: "#00d68f", points: [[0, 1], [1, 2], [2, 1.5]] },
          { id: "b", label: "B", color: "#5b8def", points: [[0, 0.5], [1, 1], [2, 2]] },
        ]}
      />,
    );
    expect(container.querySelectorAll("path.series-line")).toHaveLength(2);
    expect(container.querySelector("svg")?.getAttribute("aria-label")).toBe("test chart");
  });
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `npx vitest run components/charts/LineChart.test.tsx`
Expected: FAIL — cannot resolve `@/components/charts/LineChart`.

- [ ] **Step 4: Implement `LineChart`**

Create `web/components/charts/LineChart.tsx`:
```tsx
"use client";
import { useMemo } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { extent } from "d3-array";
import { useResizeObserver } from "@/components/charts/useResizeObserver";

export interface Series {
  id: string;
  label: string;
  color: string;
  points: [number, number][];
}

export interface LineChartProps {
  series: Series[];
  ariaLabel: string;
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xType?: "linear" | "time";
  yUnit?: string;
  zeroBaseline?: boolean;
}

const M = { top: 12, right: 16, bottom: 36, left: 48 };

export function LineChart(props: LineChartProps) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = props.width ?? measured;
  const height = props.height ?? 280;

  const content = useMemo(() => {
    if (width === 0) return null;
    const iw = Math.max(1, width - M.left - M.right);
    const ih = Math.max(1, height - M.top - M.bottom);

    const allX = props.series.flatMap((s) => s.points.map((p) => p[0]));
    const allY = props.series.flatMap((s) => s.points.map((p) => p[1]));
    if (props.zeroBaseline) allY.push(0);
    const [x0, x1] = extent(allX) as [number, number];
    const [y0, y1] = extent(allY) as [number, number];

    const x = scaleLinear().domain([x0 ?? 0, x1 ?? 1]).range([0, iw]).nice();
    const y = scaleLinear().domain([y0 ?? 0, y1 ?? 1]).range([ih, 0]).nice();

    const gen = line<[number, number]>()
      .x((p) => x(p[0]))
      .y((p) => y(p[1]));

    const xTicks = x.ticks(6);
    const yTicks = y.ticks(5);
    const fmtX = (v: number) =>
      props.xType === "time" ? new Date(v).toLocaleDateString(undefined, { year: "2-digit", month: "short" }) : String(v);
    const fmtY = (v: number) => `${v}${props.yUnit ?? ""}`;

    return { iw, ih, x, y, gen, xTicks, yTicks, fmtX, fmtY };
  }, [props.series, props.zeroBaseline, props.xType, props.yUnit, width, height]);

  return (
    <div ref={ref} className="w-full">
      {content && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={props.ariaLabel}
          className="overflow-visible"
        >
          <g transform={`translate(${M.left},${M.top})`}>
            {/* gridlines + y ticks */}
            {content.yTicks.map((t) => (
              <g key={`y${t}`} transform={`translate(0,${content.y(t)})`}>
                <line x1={0} x2={content.iw} stroke="var(--grid)" />
                <text x={-8} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)" className="tabnum">
                  {content.fmtY(t)}
                </text>
              </g>
            ))}
            {/* x ticks */}
            {content.xTicks.map((t) => (
              <g key={`x${t}`} transform={`translate(${content.x(t)},${content.ih})`}>
                <line y1={0} y2={6} stroke="var(--muted)" />
                <text y={20} textAnchor="middle" fontSize={11} fill="var(--muted)" className="tabnum">
                  {content.fmtX(t)}
                </text>
              </g>
            ))}
            {/* zero baseline */}
            {props.zeroBaseline && (
              <line x1={0} x2={content.iw} y1={content.y(0)} y2={content.y(0)} stroke="var(--muted)" strokeDasharray="3 3" />
            )}
            {/* series */}
            {props.series.map((s) => (
              <path
                key={s.id}
                className="series-line"
                d={content.gen(s.points) ?? ""}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
              />
            ))}
            {/* axis labels */}
            {props.xLabel && (
              <text x={content.iw / 2} y={content.ih + 32} textAnchor="middle" fontSize={12} fill="var(--muted)">
                {props.xLabel}
              </text>
            )}
            {props.yLabel && (
              <text transform={`translate(${-36},${content.ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fill="var(--muted)">
                {props.yLabel}
              </text>
            )}
          </g>
        </svg>
      )}
      {props.series.length > 1 && (
        <div className="mt-2 flex flex-wrap gap-4 text-xs text-[var(--muted)]">
          {props.series.map((s) => (
            <span key={s.id} className="inline-flex items-center gap-1.5">
              <span className="inline-block h-2 w-3 rounded-sm" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run components/charts/LineChart.test.tsx`
Expected: test passes (two `path.series-line` elements).

- [ ] **Step 6: Commit**

```bash
cd ..
git add web/components/charts
git commit -m "feat(web): responsive D3 LineChart with render test"
cd web
```

---

## Task 7: Home page

**Files:** modify `web/app/page.tsx`.

- [ ] **Step 1: Implement the Home page**

Replace `web/app/page.tsx` with:
```tsx
import Link from "next/link";
import { headers } from "next/headers";
import { Card } from "@/components/ui/Card";
import { Kpi } from "@/components/ui/Kpi";
import { computeCurveKpis } from "@/lib/finance";
import { Kpis } from "@/lib/types";

export const dynamic = "force-dynamic";

async function getKpis(): Promise<{ kpis: Kpis | null; date: string | null }> {
  try {
    const h = await headers();
    const host = h.get("host");
    const proto = h.get("x-forwarded-proto") ?? "http";
    const res = await fetch(`${proto}://${host}/api/treasury/latest`, { cache: "no-store" });
    if (!res.ok) return { kpis: null, date: null };
    const { row } = await res.json();
    if (!row) return { kpis: null, date: null };
    return { kpis: computeCurveKpis(row), date: row.date };
  } catch {
    return { kpis: null, date: null };
  }
}

const pct = (v: number | null) => (v === null ? "—" : `${v.toFixed(2)}%`);
const bps = (v: number | null) => (v === null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(0)} bps`);

export default async function Home() {
  const { kpis, date } = await getKpis();
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-4xl font-semibold text-[var(--accent)]">BONDVIZ</h1>
        <p className="mt-1 text-lg text-[var(--muted)]">Fixed-income research terminal</p>
        <p className="mt-3 max-w-2xl text-[var(--text)]">
          Live U.S. Treasury data to visualize the yield curve and price bonds. A front-end-focused
          demo built with Next.js and hand-rolled D3 charts.
        </p>
      </section>

      <Card>
        <h2 className="mb-3 text-lg">Snapshot{date ? ` · ${date}` : ""}</h2>
        {kpis === null ? (
          <p className="text-[var(--muted)]">Live Treasury snapshot unavailable — open the tools from the nav.</p>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Kpi label="10Y Treasury" value={pct(kpis.tenYear)} />
            <Kpi label="2s10s spread" value={bps(kpis.twos10s)} />
            <Kpi label="3m10y spread" value={bps(kpis.threeM10Y)} />
          </div>
        )}
      </Card>

      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Link href="/yield-curve">
          <Card className="h-full transition-colors hover:border-l-[var(--text)]">
            <h3 className="text-[var(--accent)]">Yield Curve →</h3>
            <p className="mt-1 text-sm text-[var(--muted)]">Latest curve, shifts vs the past, and key spreads.</p>
          </Card>
        </Link>
        <Link href="/pricing">
          <Card className="h-full transition-colors hover:border-l-[var(--text)]">
            <h3 className="text-[var(--accent)]">Bond Pricing →</h3>
            <p className="mt-1 text-sm text-[var(--muted)]">Continuous-compounding present value calculator.</p>
          </Card>
        </Link>
      </section>
    </div>
  );
}
```

- [ ] **Step 2: Verify build**

Run: `npm run build`
Expected: builds with no type errors.

- [ ] **Step 3: Commit**

```bash
cd ..
git add web/app/page.tsx
git commit -m "feat(web): home page with live KPI snapshot"
cd web
```

---

## Task 8: Yield Curve explorer page

**Files:** create `web/app/yield-curve/page.tsx`, `web/app/yield-curve/YieldCurveClient.tsx`.

- [ ] **Step 1: Create the server page wrapper**

Create `web/app/yield-curve/page.tsx`:
```tsx
import { YieldCurveClient } from "./YieldCurveClient";

export const metadata = { title: "Yield Curve · BondViz" };

export default function YieldCurvePage() {
  return <YieldCurveClient />;
}
```

- [ ] **Step 2: Create the client component**

Create `web/app/yield-curve/YieldCurveClient.tsx`:
```tsx
"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { LineChart, Series } from "@/components/charts/LineChart";
import { rowToCurve, spreadSeries, describeCurve } from "@/lib/finance";
import { YieldRow } from "@/lib/types";

const COMPARE = [
  { label: "1M ago", months: 1, color: "#5b8def" },
  { label: "3M ago", months: 3, color: "#f5a623" },
  { label: "6M ago", months: 6, color: "#e5484d" },
  { label: "1Y ago", months: 12, color: "#9b59b6" },
];

function iso(d: Date) {
  return d.toISOString().slice(0, 10);
}

function nearest(rows: YieldRow[], target: string): YieldRow | null {
  if (rows.length === 0) return null;
  let best = rows[0];
  let bestDiff = Infinity;
  const tt = new Date(target).getTime();
  for (const r of rows) {
    const diff = Math.abs(new Date(r.date).getTime() - tt);
    if (diff < bestDiff) { bestDiff = diff; best = r; }
  }
  return best;
}

export function YieldCurveClient() {
  const [rows, setRows] = useState<YieldRow[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const end = new Date();
    const start = new Date();
    start.setFullYear(start.getFullYear() - 1);
    start.setDate(start.getDate() - 14); // pad so "1Y ago" has a neighbor
    fetch(`/api/treasury/range?start=${iso(start)}&end=${iso(end)}`)
      .then((r) => r.json())
      .then((d) => setRows(d.rows ?? []))
      .catch(() => setError(true));
  }, []);

  const view = useMemo(() => {
    if (!rows || rows.length === 0) return null;
    const latest = rows[rows.length - 1];
    const latestCurve = rowToCurve(latest);

    const curveSeries: Series[] = [
      { id: "latest", label: latest.date, color: "#00d68f", points: latestCurve.map((p) => [p.years, p.yield]) },
    ];
    for (const c of COMPARE) {
      const target = new Date(latest.date);
      target.setMonth(target.getMonth() - c.months);
      const row = nearest(rows, iso(target));
      if (row) {
        const pts = rowToCurve(row).map((p) => [p.years, p.yield] as [number, number]);
        if (pts.length) curveSeries.push({ id: c.label, label: c.label, color: c.color, points: pts });
      }
    }

    const { twos10s, threeM10Y } = spreadSeries(rows);
    const spreadSeriesData: Series[] = [
      { id: "2s10s", label: "2s10s", color: "#00d68f", points: twos10s.map(([t, v]) => [t, v * 100]) },
      { id: "3m10y", label: "3m10y", color: "#5b8def", points: threeM10Y.map(([t, v]) => [t, v * 100]) },
    ];

    return { latest, latestCurve, curveSeries, spreadSeriesData };
  }, [rows]);

  if (error) return <p className="text-[var(--muted)]">Treasury data is unavailable right now.</p>;
  if (!view) return <p className="text-[var(--muted)]">Loading Treasury data…</p>;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl">Yield Curve Explorer</h1>

      <Card>
        <h2 className="mb-2 text-lg">Latest curve · {view.latest.date}</h2>
        <LineChart
          ariaLabel="Latest Treasury yield curve by maturity"
          series={[view.curveSeries[0]]}
          xLabel="Maturity (years)"
          yLabel="Yield (%)"
          yUnit="%"
        />
        <p className="mt-2 text-sm text-[var(--muted)]">{describeCurve(view.latestCurve)}</p>
      </Card>

      <Card>
        <h2 className="mb-2 text-lg">Curve shifts vs the past</h2>
        <LineChart
          ariaLabel="Latest yield curve compared with prior periods"
          series={view.curveSeries}
          xLabel="Maturity (years)"
          yLabel="Yield (%)"
          yUnit="%"
        />
      </Card>

      <Card>
        <h2 className="mb-2 text-lg">Key spreads over time</h2>
        <LineChart
          ariaLabel="2s10s and 3m10y spreads over time"
          series={view.spreadSeriesData}
          xType="time"
          xLabel="Date"
          yLabel="Spread (bps)"
          yUnit=""
          zeroBaseline
        />
        <p className="mt-2 text-sm text-[var(--muted)]">
          Negative spreads indicate inversion. Lines below the dashed zero line mark inverted regimes.
        </p>
      </Card>
    </div>
  );
}
```

- [ ] **Step 3: Verify build**

Run: `npm run build`
Expected: builds with no type errors.

- [ ] **Step 4: Commit**

```bash
cd ..
git add web/app/yield-curve
git commit -m "feat(web): yield curve explorer (latest, shifts, spreads)"
cd web
```

---

## Task 9: Bond Pricing page

**Files:** create `web/app/pricing/page.tsx`, `web/app/pricing/PricingClient.tsx`.

- [ ] **Step 1: Create the server page wrapper**

Create `web/app/pricing/page.tsx`:
```tsx
import { PricingClient } from "./PricingClient";

export const metadata = { title: "Bond Pricing · BondViz" };

export default function PricingPage() {
  return <PricingClient />;
}
```

- [ ] **Step 2: Create the client component**

Create `web/app/pricing/PricingClient.tsx`:
```tsx
"use client";
import { useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Kpi } from "@/components/ui/Kpi";
import { LineChart } from "@/components/charts/LineChart";
import { pvContinuous, discountFactors } from "@/lib/finance";

function NumberField({ label, value, step, onChange }: { label: string; value: number; step: number; onChange: (v: number) => void }) {
  return (
    <label className="block">
      <span className="text-sm text-[var(--muted)]">{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="tabnum mt-1 w-full rounded border border-[var(--panel-border)] bg-[var(--bg)] px-3 py-2 text-[var(--text)]"
      />
    </label>
  );
}

export function PricingClient() {
  const [face, setFace] = useState(1000);
  const [coupon, setCoupon] = useState(0.05);
  const [ytm, setYtm] = useState(0.04);
  const [years, setYears] = useState(10);

  const { pv, dfPoints } = useMemo(() => {
    const safe = (n: number) => (Number.isFinite(n) ? n : 0);
    const pv = pvContinuous(safe(face), safe(coupon), safe(ytm), safe(years));
    const grid = Array.from({ length: 11 }, (_, i) => (safe(years) * i) / 10);
    const dfPoints = discountFactors(safe(ytm), grid).map((d) => [d.t, d.df] as [number, number]);
    return { pv, dfPoints };
  }, [face, coupon, ytm, years]);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl">Bond Pricing</h1>
      <p className="text-[var(--muted)]">Present value of a fixed-coupon bond under continuous compounding.</p>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <Card>
          <h2 className="mb-3 text-lg">Inputs</h2>
          <div className="grid grid-cols-2 gap-4">
            <NumberField label="Face" value={face} step={100} onChange={setFace} />
            <NumberField label="Coupon rate" value={coupon} step={0.005} onChange={setCoupon} />
            <NumberField label="Continuous yield" value={ytm} step={0.005} onChange={setYtm} />
            <NumberField label="Years to maturity" value={years} step={1} onChange={setYears} />
          </div>
          <div className="mt-5">
            <Kpi label="Present Value" value={pv.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} />
          </div>
        </Card>

        <Card>
          <h2 className="mb-2 text-lg">Discount factors</h2>
          <LineChart
            ariaLabel="Continuous-compounding discount factor by maturity"
            series={[{ id: "df", label: "Discount factor", color: "#00d68f", points: dfPoints }]}
            xLabel="Years"
            yLabel="Discount factor"
          />
        </Card>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify build**

Run: `npm run build`
Expected: builds with no type errors.

- [ ] **Step 4: Commit**

```bash
cd ..
git add web/app/pricing
git commit -m "feat(web): bond pricing calculator with live PV and discount curve"
cd web
```

---

## Task 10: Deployment docs

**Files:** create `web/README.md`; modify root `README.md`.

- [ ] **Step 1: Write `web/README.md`**

Create `web/README.md`:
```markdown
# BondViz Web (Next.js front-end)

Native front-end for BondViz: Home, Yield Curve explorer, and Bond Pricing, built with
Next.js (App Router), TypeScript, Tailwind, and hand-rolled D3 charts. Data comes from the
keyless U.S. Treasury par-yield XML feed via server route handlers (cached hourly).

## Develop

```bash
cd web
npm install
npm run dev      # http://localhost:3000
npm test         # Vitest unit tests
npm run build    # production build
```

## Deploy to Vercel (at bondviz.nikolayvalev.com)

1. Import the GitHub repo in Vercel. Set **Root Directory = `web/`** (framework preset: Next.js). No env vars.
2. Deploy. Vercel builds `web/` and serves the app on a `*.vercel.app` URL.
3. Add the custom domain in **Project → Settings → Domains**: `bondviz.nikolayvalev.com`.
   Vercel shows a target like `cname.vercel-dns.com`.
4. In **Cloudflare DNS** for `nikolayvalev.com`, add:
   - Type `CNAME`, Name `bondviz`, Target `cname.vercel-dns.com`, Proxy status **DNS only** (grey cloud).
   Vercel then provisions the TLS certificate; the app is live at the custom domain.
```

- [ ] **Step 2: Add a pointer to the root `README.md`**

In the root `README.md`, immediately after the "Deploy to Streamlit Community Cloud" section (before `### License`), add:
```markdown
## Web front-end (Vercel)

A native Next.js + D3 front-end lives in [`web/`](web/) and deploys to Vercel at a custom domain
(see [`web/README.md`](web/README.md)). It reuses the same Treasury data and is independent of the
Streamlit app.
```

- [ ] **Step 3: Commit**

```bash
cd ..
git add web/README.md README.md
git commit -m "docs(web): add run + Vercel/Cloudflare deploy instructions"
```

---

## Task 11: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Full test suite**

Run (from `web/`): `npm test`
Expected: all suites pass (finance, treasury, routes, LineChart, smoke).

- [ ] **Step 2: Lint + production build**

Run: `npm run lint` (expected: no errors) and `npm run build` (expected: succeeds; route list shows `/`, `/yield-curve`, `/pricing`, `/api/treasury/latest`, `/api/treasury/range`).

- [ ] **Step 3: Manual smoke (live data)**

Run: `npm run dev`, open http://localhost:3000 and confirm:
- Home shows the hero and a KPI snapshot (or a graceful "unavailable" message).
- `/yield-curve` renders the latest curve, the shifts overlay (multiple colored lines + legend), and the spreads chart with a dashed zero line.
- `/pricing` updates the Present Value and discount-factor curve live as inputs change.
- All charts are dark-themed and resize when the window width changes.

- [ ] **Step 4: Final commit (only if tweaks were needed)**

```bash
cd ..
git add -A
git commit -m "chore(web): verification fixes"
```

---

## Self-Review Notes

- **Spec coverage:** Next/TS/Tailwind in `web/` (Task 1–2); D3 charts React-owns-SVG (Task 6, used in 8–9); Home + KPIs (Task 7); Yield Curve latest/shifts/spreads + blurbs (Task 8); Bond Pricing live PV + discount curve (Task 9); keyless Treasury route handlers cached hourly (Task 4–5); ported finance lib + treasury parse, both Vitest-tested (Task 3–4); route tests with mocked fetch (Task 5); graceful degradation (Task 7 KPIs, Task 8 error/loading states, route 503/400s); accessibility (palette contrast in Task 2, `role="img"`/`aria-label` on charts in Task 6); Vercel + Cloudflare deploy (Task 10). PCA/Stocks/heatmap/bootstrap/FRED are out of scope per the spec — no tasks, intentionally.
- **Name consistency:** `Series`/`LineChart` props (Task 6) are imported unchanged in Tasks 8–9. `computeCurveKpis`/`rowToCurve`/`spreadSeries`/`pvContinuous`/`discountFactors`/`describeCurve` (Task 3) and `parseTreasuryXml`/`fetchTreasuryYear` (Task 4) are used with identical signatures downstream. KPI field names `tenYear`/`twos10s`/`threeM10Y` are consistent between `types.ts`, `finance.ts`, and `page.tsx`.
- **Greenfield:** no existing patterns to match beyond reusing the `theme.py` palette values.
```
