# Inversion & Recession Signal Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/signal` page that frames yield-curve inversion as a recession indicator — current status & streak, a deep-history (1990–today) 10y–3m / 2s10s spread chart with inversion shading and NBER recession bands, and a table of every inversion episode with whether a recession followed.

**Architecture:** A pure, unit-tested analysis module (`lib/signal.ts`) derives slim spread points, current status, and inversion episodes (cross-referenced against a static NBER recession list). A slim spread-only API route (`/api/treasury/spreads`) serves 35 years without large payloads, reusing `fetchTreasuryYear`. A dedicated `SpreadHistoryChart` renders the shaded long chart. A fail-soft client page wires it together; nav + home get links.

**Tech Stack:** Next.js 16, React 19, TypeScript, Tailwind 4, d3-scale/d3-shape/d3-array, vitest + Testing Library.

## Global Constraints

- Surface is `web/`; run npm commands from `web/`.
- **Next.js 16 has breaking changes** — per `web/AGENTS.md`, read the route/page guide in `node_modules/next/dist/docs/` before writing the API route and page.
- Headline signal is **10y–3m** (`BC_10YEAR − BC_3MONTH`); secondary is **2s10s** (`BC_10YEAR − BC_2YEAR`). Spreads are in percentage points internally; convert to bps only at the display edge (`× 100`).
- Inversion test: `s10y3m <= 0` (flat counts as inverted). A `null` spread breaks a run (not inverted, not carried).
- `recessionFollowed`: an NBER recession `start` falls within `[episode.start, episode.start + 24 months]`.
- NBER_RECESSIONS (peak→trough, ISO month precision): `1990-07-01→1991-03-01`, `2001-03-01→2001-11-01`, `2007-12-01→2009-06-01`, `2020-02-01→2020-04-01`.
- Fail-soft everywhere: skipped years degrade gracefully; empty data shows an "unavailable" message, never throws.
- Run tests `npm test`; lint `npm run lint`; types `npx tsc --noEmit`.

---

### Task 1: Signal analysis module

**Files:**
- Create: `web/lib/signal.ts`
- Test: `web/lib/signal.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks (reads `BC_*` numeric columns off plain row objects).
- Produces:
  - `interface SpreadPoint { date: string; s10y3m: number | null; s2s10s: number | null; }`
  - `interface SignalStatus { date: string | null; s10y3m: number | null; s2s10s: number | null; inverted: boolean; streakDays: number; }`
  - `interface InversionEpisode { start: string; end: string; days: number; maxDepthBps: number; recessionFollowed: boolean; }`
  - `interface NberRecession { start: string; end: string; }`
  - `const NBER_RECESSIONS: NberRecession[]`
  - `function toSpreadPoints(rows: Record<string, unknown>[]): SpreadPoint[]`
  - `function currentStatus(points: SpreadPoint[]): SignalStatus`
  - `function inversionEpisodes(points: SpreadPoint[]): InversionEpisode[]`

- [ ] **Step 1: Write the failing tests**

Create `web/lib/signal.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import {
  toSpreadPoints,
  currentStatus,
  inversionEpisodes,
  NBER_RECESSIONS,
} from "@/lib/signal";

const row = (date: string, y3m: number | null, y2: number | null, y10: number | null) => ({
  date,
  BC_3MONTH: y3m,
  BC_2YEAR: y2,
  BC_10YEAR: y10,
});

describe("toSpreadPoints", () => {
  it("computes 10y3m and 2s10s spreads", () => {
    const pts = toSpreadPoints([row("2025-01-02", 5.0, 4.0, 4.5)]);
    expect(pts[0].s10y3m).toBeCloseTo(-0.5, 9); // 4.5 - 5.0
    expect(pts[0].s2s10s).toBeCloseTo(0.5, 9); // 4.5 - 4.0
  });

  it("yields null when a leg is missing", () => {
    const pts = toSpreadPoints([row("2025-01-02", null, 4.0, 4.5)]);
    expect(pts[0].s10y3m).toBeNull();
    expect(pts[0].s2s10s).toBeCloseTo(0.5, 9);
  });
});

describe("currentStatus", () => {
  it("zeroes out for empty input", () => {
    expect(currentStatus([])).toEqual({
      date: null, s10y3m: null, s2s10s: null, inverted: false, streakDays: 0,
    });
  });

  it("counts the trailing inverted streak and stops at a positive point", () => {
    const pts: { date: string; s10y3m: number | null; s2s10s: number | null }[] = [
      { date: "d1", s10y3m: 0.5, s2s10s: 0.2 },
      { date: "d2", s10y3m: 0.1, s2s10s: 0.1 },
      { date: "d3", s10y3m: -0.2, s2s10s: -0.1 },
      { date: "d4", s10y3m: -0.3, s2s10s: -0.1 },
    ];
    const s = currentStatus(pts);
    expect(s.inverted).toBe(true);
    expect(s.streakDays).toBe(2);
    expect(s.date).toBe("d4");
  });

  it("reports not inverted when the latest point is positive", () => {
    const s = currentStatus([{ date: "d1", s10y3m: 0.3, s2s10s: 0.1 }]);
    expect(s.inverted).toBe(false);
    expect(s.streakDays).toBe(0);
  });
});

describe("inversionEpisodes", () => {
  const mk = (vals: (number | null)[], start = 2020): { date: string; s10y3m: number | null; s2s10s: number | null }[] =>
    vals.map((v, i) => ({ date: `${start}-01-${String(i + 1).padStart(2, "0")}`, s10y3m: v, s2s10s: 0 }));

  it("returns [] for an all-positive series", () => {
    expect(inversionEpisodes(mk([0.1, 0.2, 0.3]))).toEqual([]);
  });

  it("detects a single inverted day as a 1-day episode", () => {
    const eps = inversionEpisodes(mk([0.1, -0.2, 0.3]));
    expect(eps).toHaveLength(1);
    expect(eps[0].days).toBe(1);
    expect(eps[0].start).toBe("2020-01-02");
    expect(eps[0].end).toBe("2020-01-02");
    expect(eps[0].maxDepthBps).toBeCloseTo(-20, 6);
  });

  it("detects a multi-day run with correct boundaries and depth", () => {
    const eps = inversionEpisodes(mk([0.1, -0.2, -0.5, -0.3, 0.2]));
    expect(eps).toHaveLength(1);
    expect(eps[0].start).toBe("2020-01-02");
    expect(eps[0].end).toBe("2020-01-04");
    expect(eps[0].days).toBe(3);
    expect(eps[0].maxDepthBps).toBeCloseTo(-50, 6);
  });

  it("closes an episode that is still inverted at the last point", () => {
    const eps = inversionEpisodes(mk([0.1, -0.2, -0.3]));
    expect(eps).toHaveLength(1);
    expect(eps[0].end).toBe("2020-01-03");
    expect(eps[0].days).toBe(2);
  });

  it("splits two runs separated by a positive (or null) point", () => {
    const eps = inversionEpisodes(mk([-0.1, 0.2, null, -0.3]));
    expect(eps).toHaveLength(2);
    expect(eps[0].start).toBe("2020-01-01");
    expect(eps[1].start).toBe("2020-01-04");
  });

  it("flags recessionFollowed when an NBER recession starts within 24 months", () => {
    // 2007-12 recession start; an inversion starting 2006-06 is ~18 months prior.
    const eps = inversionEpisodes([
      { date: "2006-06-01", s10y3m: -0.1, s2s10s: 0 },
      { date: "2006-06-02", s10y3m: -0.2, s2s10s: 0 },
    ]);
    expect(eps[0].recessionFollowed).toBe(true);
  });

  it("does not flag recessionFollowed when no recession is within 24 months", () => {
    // 2013 inversion: nearest NBER start (2020-02) is > 24 months away.
    const eps = inversionEpisodes([
      { date: "2013-01-01", s10y3m: -0.1, s2s10s: 0 },
      { date: "2013-01-02", s10y3m: -0.2, s2s10s: 0 },
    ]);
    expect(eps[0].recessionFollowed).toBe(false);
  });
});

describe("NBER_RECESSIONS", () => {
  it("lists the four recessions since 1990", () => {
    expect(NBER_RECESSIONS).toHaveLength(4);
    expect(NBER_RECESSIONS[0].start).toBe("1990-07-01");
    expect(NBER_RECESSIONS[3].end).toBe("2020-04-01");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd web && npm test -- signal`
Expected: FAIL — module `@/lib/signal` not found.

- [ ] **Step 3: Implement `web/lib/signal.ts`**

```ts
export interface SpreadPoint {
  date: string;
  s10y3m: number | null; // 10Y − 3M, percentage points
  s2s10s: number | null; // 10Y − 2Y, percentage points
}

export interface SignalStatus {
  date: string | null;
  s10y3m: number | null;
  s2s10s: number | null;
  inverted: boolean;
  streakDays: number;
}

export interface InversionEpisode {
  start: string;
  end: string;
  days: number;
  maxDepthBps: number; // most negative s10y3m over the run, in bps (<= 0)
  recessionFollowed: boolean;
}

export interface NberRecession {
  start: string;
  end: string;
}

export const NBER_RECESSIONS: NberRecession[] = [
  { start: "1990-07-01", end: "1991-03-01" },
  { start: "2001-03-01", end: "2001-11-01" },
  { start: "2007-12-01", end: "2009-06-01" },
  { start: "2020-02-01", end: "2020-04-01" },
];

function num(v: unknown): number | null {
  return typeof v === "number" && !Number.isNaN(v) ? v : null;
}

function spread(a: number | null, b: number | null): number | null {
  return a !== null && b !== null ? a - b : null;
}

/** Map raw yield rows (BC_* columns) to slim spread points, sorted by date. */
export function toSpreadPoints(rows: Record<string, unknown>[]): SpreadPoint[] {
  return rows
    .map((r) => {
      const y10 = num(r.BC_10YEAR);
      const y3m = num(r.BC_3MONTH);
      const y2 = num(r.BC_2YEAR);
      return {
        date: String(r.date),
        s10y3m: spread(y10, y3m),
        s2s10s: spread(y10, y2),
      };
    })
    .sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
}

const isInverted = (v: number | null): boolean => v !== null && v <= 0;

/** Latest values plus the trailing consecutive-inverted streak. */
export function currentStatus(points: SpreadPoint[]): SignalStatus {
  if (points.length === 0) {
    return { date: null, s10y3m: null, s2s10s: null, inverted: false, streakDays: 0 };
  }
  const last = points[points.length - 1];
  let streak = 0;
  for (let i = points.length - 1; i >= 0; i--) {
    if (isInverted(points[i].s10y3m)) streak++;
    else break;
  }
  return {
    date: last.date,
    s10y3m: last.s10y3m,
    s2s10s: last.s2s10s,
    inverted: isInverted(last.s10y3m),
    streakDays: streak,
  };
}

/** Months between two ISO dates (approximate, calendar-based). */
function monthsBetween(fromIso: string, toIso: string): number {
  const a = new Date(fromIso);
  const b = new Date(toIso);
  return (b.getUTCFullYear() - a.getUTCFullYear()) * 12 + (b.getUTCMonth() - a.getUTCMonth());
}

function recessionWithin24m(episodeStart: string): boolean {
  return NBER_RECESSIONS.some((r) => {
    const m = monthsBetween(episodeStart, r.start);
    return m >= 0 && m <= 24;
  });
}

/** Maximal consecutive runs of inverted (s10y3m <= 0) points. */
export function inversionEpisodes(points: SpreadPoint[]): InversionEpisode[] {
  const episodes: InversionEpisode[] = [];
  let runStart = -1;
  let minVal = Infinity;

  const close = (endIdx: number) => {
    const start = points[runStart].date;
    episodes.push({
      start,
      end: points[endIdx].date,
      days: endIdx - runStart + 1,
      maxDepthBps: minVal * 100,
      recessionFollowed: recessionWithin24m(start),
    });
    runStart = -1;
    minVal = Infinity;
  };

  for (let i = 0; i < points.length; i++) {
    const v = points[i].s10y3m;
    if (isInverted(v)) {
      if (runStart === -1) runStart = i;
      if ((v as number) < minVal) minVal = v as number;
    } else if (runStart !== -1) {
      close(i - 1);
    }
  }
  if (runStart !== -1) close(points.length - 1);
  return episodes;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd web && npm test -- signal`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
cd web && git add lib/signal.ts lib/signal.test.ts
git commit -m "feat(web): inversion & recession signal analysis lib with tests"
```

---

### Task 2: Slim spreads API route

**Files:**
- Create: `web/app/api/treasury/spreads/route.ts`
- Test: `web/app/api/treasury/spreads/route.test.ts`

**Interfaces:**
- Consumes: `fetchTreasuryYear` from `@/lib/treasury`; `toSpreadPoints`/`SpreadPoint` from `@/lib/signal`.
- Produces: `GET(req: Request): Promise<NextResponse>` returning `{ points: SpreadPoint[] }`; `400` without `start`/`end`; `503` on total failure.

- [ ] **Step 1: Read the Next.js 16 route guide**

Run: `ls node_modules/next/dist/docs/` (from `web/`) and read the route-handler guidance referenced by `web/AGENTS.md`. Confirm the existing `app/api/treasury/range/route.ts` pattern (a `GET(req: Request)` returning `NextResponse.json`) is current — mirror it.

- [ ] **Step 2: Write the failing test**

Create `web/app/api/treasury/spreads/route.test.ts`:

```ts
import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { GET } from "@/app/api/treasury/spreads/route";

const xml = readFileSync(
  fileURLToPath(new URL("../../../../test/fixtures/treasury-sample.xml", import.meta.url)),
  "utf-8",
);

afterEach(() => vi.restoreAllMocks());

describe("/api/treasury/spreads", () => {
  it("returns slim spread points filtered to the window", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(xml, { status: 200 })));
    const req = new Request("http://x/api/treasury/spreads?start=2025-01-03&end=2025-01-03");
    const res = await GET(req);
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.points).toHaveLength(1);
    expect(body.points[0].date).toBe("2025-01-03");
    // fixture 2025-01-03: 10Y 4.55, 3M 5.01, 2Y 4.05
    expect(body.points[0].s10y3m).toBeCloseTo(-0.46, 6);
    expect(body.points[0].s2s10s).toBeCloseTo(0.5, 6);
  });

  it("400s without start/end", async () => {
    const res = await GET(new Request("http://x/api/treasury/spreads"));
    expect(res.status).toBe(400);
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd web && npm test -- spreads`
Expected: FAIL — route module not found.

- [ ] **Step 4: Implement `web/app/api/treasury/spreads/route.ts`**

```ts
import { NextResponse } from "next/server";
import { fetchTreasuryYear } from "@/lib/treasury";
import { toSpreadPoints } from "@/lib/signal";
import { YieldRow } from "@/lib/types";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const start = searchParams.get("start");
  const end = searchParams.get("end");
  if (!start || !end) return NextResponse.json({ points: [] }, { status: 400 });

  try {
    const y0 = new Date(start).getUTCFullYear();
    const y1 = new Date(end).getUTCFullYear();
    const all: YieldRow[] = [];
    for (let y = y0; y <= y1; y++) {
      const rows = await fetchTreasuryYear(y).catch(() => [] as YieldRow[]);
      all.push(...rows);
    }
    const rows = all.filter((r) => r.date >= start && r.date <= end);
    return NextResponse.json({ points: toSpreadPoints(rows) });
  } catch {
    return NextResponse.json({ points: [] }, { status: 503 });
  }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd web && npm test -- spreads`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd web && git add app/api/treasury/spreads/route.ts app/api/treasury/spreads/route.test.ts
git commit -m "feat(web): slim treasury spreads API route with tests"
```

---

### Task 3: Spread-history chart with inversion & recession shading

**Files:**
- Create: `web/components/charts/SpreadHistoryChart.tsx`
- Test: `web/components/charts/SpreadHistoryChart.test.tsx`

**Interfaces:**
- Consumes: `useResizeObserver` from `@/components/charts/useResizeObserver`; `SpreadPoint`/`NberRecession` from `@/lib/signal`; `scaleTime`/`scaleLinear` from `d3-scale`, `line`/`area` from `d3-shape`.
- Produces: `function SpreadHistoryChart(props: { points: SpreadPoint[]; recessions: NberRecession[]; ariaLabel: string; width?: number; height?: number; }): JSX.Element`
  - Renders an svg `role="img"` with the aria-label; one `path.series-10y3m` and one `path.series-2s10s`; gray `rect.recession` bands; a red `path.inversion-fill` for the negative region; a dashed zero baseline.

- [ ] **Step 1: Write the failing test**

Create `web/components/charts/SpreadHistoryChart.test.tsx`:

```tsx
import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { SpreadHistoryChart } from "@/components/charts/SpreadHistoryChart";

const points = [
  { date: "2019-01-01", s10y3m: 0.3, s2s10s: 0.2 },
  { date: "2019-06-01", s10y3m: -0.1, s2s10s: 0.05 },
  { date: "2020-01-01", s10y3m: -0.2, s2s10s: -0.1 },
  { date: "2020-06-01", s10y3m: 0.5, s2s10s: 0.4 },
];

const recessions = [{ start: "2020-02-01", end: "2020-04-01" }];

describe("SpreadHistoryChart", () => {
  it("renders both series, a recession band, and the inversion fill", () => {
    const { container, getByRole } = render(
      <SpreadHistoryChart width={800} ariaLabel="spread history" points={points} recessions={recessions} />,
    );
    expect(getByRole("img", { name: "spread history" })).toBeTruthy();
    expect(container.querySelector("path.series-10y3m")).toBeTruthy();
    expect(container.querySelector("path.series-2s10s")).toBeTruthy();
    expect(container.querySelectorAll("rect.recession").length).toBe(1);
    expect(container.querySelector("path.inversion-fill")).toBeTruthy();
  });

  it("renders nothing fatal for empty points", () => {
    const { container } = render(
      <SpreadHistoryChart width={800} ariaLabel="empty" points={[]} recessions={recessions} />,
    );
    expect(container).toBeTruthy();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npm test -- SpreadHistoryChart`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `web/components/charts/SpreadHistoryChart.tsx`**

```tsx
"use client";
import { useMemo } from "react";
import { scaleTime, scaleLinear } from "d3-scale";
import { line, area } from "d3-shape";
import { extent } from "d3-array";
import { useResizeObserver } from "@/components/charts/useResizeObserver";
import type { SpreadPoint, NberRecession } from "@/lib/signal";

export interface SpreadHistoryChartProps {
  points: SpreadPoint[];
  recessions: NberRecession[];
  ariaLabel: string;
  width?: number;
  height?: number;
}

const M = { top: 14, right: 18, bottom: 30, left: 50 };

interface XY { t: number; v: number; }

export function SpreadHistoryChart(props: SpreadHistoryChartProps) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = props.width ?? measured;
  const height = props.height ?? 320;

  const c = useMemo(() => {
    if (width === 0 || props.points.length === 0) return null;
    const iw = Math.max(1, width - M.left - M.right);
    const ih = Math.max(1, height - M.top - M.bottom);

    const ms = (d: string) => new Date(d).getTime();
    const s10 = props.points.filter((p) => p.s10y3m !== null).map((p) => ({ t: ms(p.date), v: p.s10y3m as number * 100 }));
    const s2 = props.points.filter((p) => p.s2s10s !== null).map((p) => ({ t: ms(p.date), v: p.s2s10s as number * 100 }));

    const allT = props.points.map((p) => ms(p.date));
    const allV = [...s10.map((p) => p.v), ...s2.map((p) => p.v), 0];
    const [t0, t1] = extent(allT) as [number, number];
    const x = scaleTime().domain([t0, t1]).range([0, iw]);
    const y = scaleLinear().domain([Math.min(...allV), Math.max(...allV)]).range([ih, 0]).nice();

    const lineGen = line<XY>().x((p) => x(p.t)).y((p) => y(p.v));

    // Inversion fill: area between the 10y3m line and zero, clipped to negatives.
    const zeroY = y(0);
    const invArea = area<XY>()
      .x((p) => x(p.t))
      .y0(() => zeroY)
      .y1((p) => (p.v < 0 ? y(p.v) : zeroY));

    const recBands = props.recessions
      .map((r) => {
        const rx0 = x(ms(r.start));
        const rx1 = x(ms(r.end));
        return { key: r.start, x: Math.min(rx0, rx1), w: Math.max(1, Math.abs(rx1 - rx0)) };
      })
      .filter((b) => b.x + b.w >= 0 && b.x <= iw);

    return {
      iw, ih, x, y, zeroY, s10, s2, lineGen, invArea, recBands,
      xTicks: x.ticks(8), yTicks: y.ticks(5),
    };
  }, [props.points, props.recessions, width, height]);

  return (
    <div ref={ref} className="w-full">
      {c && (
        <svg width={width} height={height} role="img" aria-label={props.ariaLabel} className="overflow-visible">
          <g transform={`translate(${M.left},${M.top})`}>
            {/* recession bands (behind everything) */}
            {c.recBands.map((b) => (
              <rect key={b.key} className="recession" x={b.x} y={0} width={b.w} height={c.ih} fill="var(--muted)" opacity={0.12} />
            ))}
            {/* y grid + labels */}
            {c.yTicks.map((t) => (
              <g key={`y${t}`} transform={`translate(0,${c.y(t)})`}>
                <line x1={0} x2={c.iw} stroke="var(--grid)" />
                <text x={-8} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)" className="tabnum">{t}</text>
              </g>
            ))}
            {/* x ticks */}
            {c.xTicks.map((t) => (
              <g key={`x${+t}`} transform={`translate(${c.x(t)},${c.ih})`}>
                <line y1={0} y2={6} stroke="var(--faint)" />
                <text y={20} textAnchor="middle" fontSize={11} fill="var(--muted)" className="tabnum">
                  {new Date(+t).getUTCFullYear()}
                </text>
              </g>
            ))}
            {/* inversion fill (red, below zero) */}
            <path className="inversion-fill" d={c.invArea(c.s10) ?? ""} fill="var(--neg)" opacity={0.18} />
            {/* zero baseline */}
            <line x1={0} x2={c.iw} y1={c.zeroY} y2={c.zeroY} stroke="var(--panel-border-strong)" strokeDasharray="3 3" />
            {/* series */}
            <path className="series-2s10s" d={c.lineGen(c.s2) ?? ""} fill="none" stroke="#5b8def" strokeWidth={1.5} opacity={0.85} />
            <path className="series-10y3m" d={c.lineGen(c.s10) ?? ""} fill="none" stroke="var(--accent)" strokeWidth={2} />
          </g>
        </svg>
      )}
      <div className="mt-2 flex flex-wrap gap-4 text-xs text-[var(--muted)]">
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 rounded bg-[var(--accent)]" /> 10y–3m</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 rounded" style={{ background: "#5b8def" }} /> 2s10s</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded-sm bg-[var(--neg)] opacity-30" /> inverted</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded-sm bg-[var(--muted)] opacity-30" /> NBER recession</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npm test -- SpreadHistoryChart`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd web && git add components/charts/SpreadHistoryChart.tsx components/charts/SpreadHistoryChart.test.tsx
git commit -m "feat(web): spread-history chart with inversion & recession shading"
```

---

### Task 4: Signal page (client + route)

**Files:**
- Create: `web/app/signal/SignalClient.tsx`
- Create: `web/app/signal/page.tsx`

**Interfaces:**
- Consumes: `currentStatus`, `inversionEpisodes`, `NBER_RECESSIONS`, `SpreadPoint` from `@/lib/signal`; `SpreadHistoryChart` from `@/components/charts/SpreadHistoryChart`; `Card` from `@/components/ui/Card`; `iso` from `@/lib/format`.
- Data: GET `/api/treasury/spreads?start=1990-01-01&end=<today>` → `{ points: SpreadPoint[] }`.

- [ ] **Step 1: Implement `web/app/signal/SignalClient.tsx`**

```tsx
"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { SpreadHistoryChart } from "@/components/charts/SpreadHistoryChart";
import { currentStatus, inversionEpisodes, NBER_RECESSIONS, SpreadPoint } from "@/lib/signal";
import { iso } from "@/lib/format";

const bps = (pp: number | null) => (pp === null ? "—" : `${pp >= 0 ? "+" : ""}${(pp * 100).toFixed(0)} bps`);

export function SignalClient() {
  const [points, setPoints] = useState<SpreadPoint[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch(`/api/treasury/spreads?start=1990-01-01&end=${iso(new Date())}`)
      .then((r) => r.json())
      .then((d) => setPoints(d.points ?? []))
      .catch(() => setError(true));
  }, []);

  const view = useMemo(() => {
    if (!points) return null;
    const status = currentStatus(points);
    const episodes = inversionEpisodes(points).slice().reverse(); // most recent first
    const forward = inversionEpisodes(points);
    const withRec = forward.filter((e) => e.recessionFollowed).length;
    return { status, episodes, total: forward.length, withRec };
  }, [points]);

  if (error) return <p className="text-[var(--muted)]">Treasury data is unavailable right now.</p>;
  if (!view) return <p className="text-[var(--muted)]">Loading 35 years of Treasury data… (first load can take a moment)</p>;

  const { status } = view;
  const summary =
    status.streakDays > 0
      ? `10y–3m has inverted before ${view.withRec} of the last ${NBER_RECESSIONS.length} recessions; currently inverted ${status.streakDays} trading days.`
      : `10y–3m is currently positive (${bps(status.s10y3m)}). ${view.total} inversion episodes since 1990, ${view.withRec} followed by recession within 24 months.`;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl">Inversion &amp; Recession Signal</h1>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card>
          <div className="text-xs uppercase tracking-wide text-[var(--muted)]">10y–3m spread</div>
          <div className="mt-1 text-3xl font-bold tabnum" style={{ color: status.inverted ? "var(--neg)" : "var(--pos)" }}>
            {bps(status.s10y3m)}
          </div>
          <div className="mt-1 text-sm text-[var(--muted)]">{status.date ?? "—"}</div>
        </Card>
        <Card>
          <div className="text-xs uppercase tracking-wide text-[var(--muted)]">Status</div>
          <div className="mt-1 text-3xl font-bold" style={{ color: status.inverted ? "var(--neg)" : "var(--pos)" }}>
            {status.inverted ? "Inverted" : "Normal"}
          </div>
          <div className="mt-1 text-sm text-[var(--muted)] tabnum">
            {status.inverted ? `${status.streakDays} trading days` : "not inverted"}
          </div>
        </Card>
        <Card>
          <div className="text-xs uppercase tracking-wide text-[var(--muted)]">2s10s spread</div>
          <div className="mt-1 text-3xl font-bold tabnum text-[var(--text)]">{bps(status.s2s10s)}</div>
          <div className="mt-1 text-sm text-[var(--muted)]">secondary signal</div>
        </Card>
      </div>

      <Card>
        <h2 className="mb-2 text-lg">Spread history since 1990</h2>
        <SpreadHistoryChart
          ariaLabel="10y minus 3m and 2s10s spreads since 1990 with inversion and recession shading"
          points={points!}
          recessions={NBER_RECESSIONS}
        />
        <p className="mt-2 text-sm text-[var(--muted)]">{summary}</p>
      </Card>

      <Card>
        <h2 className="mb-3 text-lg">Inversion episodes (10y–3m)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm tabnum">
            <thead>
              <tr className="text-left text-[var(--muted)]">
                <th className="py-1 pr-4 font-medium">Start</th>
                <th className="py-1 pr-4 font-medium">End</th>
                <th className="py-1 pr-4 font-medium">Days</th>
                <th className="py-1 pr-4 font-medium">Max depth</th>
                <th className="py-1 pr-4 font-medium">Recession ≤24m</th>
              </tr>
            </thead>
            <tbody>
              {view.episodes.map((e) => (
                <tr key={e.start} className="border-t border-[var(--panel-border)]">
                  <td className="py-1.5 pr-4 text-[var(--text)]">{e.start}</td>
                  <td className="py-1.5 pr-4">{e.end}</td>
                  <td className="py-1.5 pr-4">{e.days}</td>
                  <td className="py-1.5 pr-4" style={{ color: "var(--neg)" }}>{e.maxDepthBps.toFixed(0)} bps</td>
                  <td className="py-1.5 pr-4" style={{ color: e.recessionFollowed ? "var(--neg)" : "var(--muted)" }}>
                    {e.recessionFollowed ? "✓" : "✗"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-[var(--faint)]">
          An episode is a maximal run of days with 10y–3m ≤ 0. &ldquo;Recession ≤24m&rdquo; marks episodes whose start
          preceded an NBER recession by no more than 24 months.
        </p>
      </Card>
    </div>
  );
}
```

- [ ] **Step 2: Implement `web/app/signal/page.tsx`**

```tsx
import { SignalClient } from "./SignalClient";

export const metadata = { title: "Inversion & Recession Signal · BondViz" };

export default function SignalPage() {
  return <SignalClient />;
}
```

- [ ] **Step 3: Verify build, types, lint, tests**

Run: `cd web && npm run lint && npx tsc --noEmit && npm run build && npm test`
Expected: all succeed; build output lists the `/signal` route; all tests pass.

- [ ] **Step 4: Manually verify (optional if no browser)**

If a browser is available: `npm run dev`, open `http://localhost:3000/signal`. Confirm the hero shows current values, the chart shows recession bands + red inversion fill, and the episode table lists the 2019 and 2022–24 inversions with ✓ where applicable. (First load fetches 35 years and may take several seconds.)

- [ ] **Step 5: Commit**

```bash
cd web && git add app/signal/page.tsx app/signal/SignalClient.tsx
git commit -m "feat(web): inversion & recession signal page"
```

---

### Task 5: Navigation + Home link

**Files:**
- Modify: `web/components/Nav.tsx` (the `LINKS` array)
- Modify: `web/app/page.tsx` (the card `<section>`)

**Interfaces:**
- Consumes: existing `LINKS` array and card grid section.

- [ ] **Step 1: Add the nav link**

In `web/components/Nav.tsx`, insert into `LINKS` after the `/carry` entry:

```tsx
  { href: "/carry", label: "Carry & Roll" },
  { href: "/signal", label: "Recession Signal" },
  { href: "/portfolio", label: "Portfolio" },
```

- [ ] **Step 2: Add a Home card**

In `web/app/page.tsx`, add after the `/carry` card `<Link>` (inside the same card `<section>`):

```tsx
        <Link href="/signal" className="group">
          <Card className="h-full transition-all group-hover:border-l-[var(--accent)] group-hover:bg-[var(--panel)]">
            <h3 className="flex items-center justify-between text-[var(--accent)]">
              Recession Signal <span className="transition-transform group-hover:translate-x-1">→</span>
            </h3>
            <p className="mt-2 text-sm text-[var(--muted)]">Curve inversion vs NBER recessions since 1990, with episode history.</p>
          </Card>
        </Link>
```

- [ ] **Step 3: Verify lint, types, tests**

Run: `cd web && npm run lint && npx tsc --noEmit && npm test`
Expected: no errors; all tests pass.

- [ ] **Step 4: Commit**

```bash
cd web && git add components/Nav.tsx app/page.tsx
git commit -m "feat(web): link Recession Signal from nav and home"
```

---

## Self-Review

**Spec coverage:**
- Slim spreads API route, year-loop, fail-soft, 400/503, mirrors range route → Task 2 ✓
- `lib/signal.ts` (toSpreadPoints, currentStatus w/ streak, inversionEpisodes, NBER_RECESSIONS, 24m lookahead) → Task 1 ✓
- Inversion test `<= 0`, null breaks run, maxDepthBps ≤ 0, bps at display edge → Tasks 1, 3, 4 ✓
- Status hero, spread-history chart w/ inversion fill + recession bands, episode table, auto-summary → Tasks 3, 4 ✓
- Dedicated `SpreadHistoryChart` (not overloading LineChart) → Task 3 ✓
- Fail-soft loading/error/empty states → Tasks 2, 4 ✓
- Nav + Home links → Task 5 ✓
- Tests: spreads, streak, episode boundaries (single/multi/ongoing/split), maxDepth, recessionFollowed true/false, empty → Task 1 ✓; route 200/400 → Task 2 ✓; chart elements/empty → Task 3 ✓

**Placeholder scan:** No TBD/TODO. Manual browser step (Task 4 Step 4) is explicitly optional and gated on browser availability; all automated verification (lint/tsc/build/test) is mandatory in Task 4 Step 3.

**Type consistency:** `SpreadPoint {date, s10y3m, s2s10s}` used identically across Tasks 1–4. `SignalStatus`/`InversionEpisode` fields match between Task 1 definition and Task 4 consumption (`status.streakDays`, `status.inverted`, `e.maxDepthBps`, `e.recessionFollowed`, `e.start/end/days`). `NBER_RECESSIONS: NberRecession[]` consumed by Tasks 3 (`recessions` prop) and 4. `toSpreadPoints` consumed by Task 2. Chart prop names (`points`, `recessions`, `ariaLabel`, `width`) consistent between Task 3 definition and Task 4 use. ✓
