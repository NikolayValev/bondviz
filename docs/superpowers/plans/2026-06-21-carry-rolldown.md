# Carry & Roll-Down Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/carry` page to the BondViz web terminal that decomposes the expected horizon return of each Treasury tenor into carry and roll-down, with a breakeven cushion, charts, and a table.

**Architecture:** A pure, unit-tested math module (`lib/carry.ts`) computes per-tenor carry/roll/total/breakeven from the latest par curve, reusing `modifiedDuration`/`bondCashflows` from `lib/finance.ts`. A reusable diverging `CategoryBarChart` renders per-tenor bars. A fail-soft client page (`app/carry`) fetches the latest curve from the existing `/api/treasury/range` route, drives a horizon segmented control, and renders charts + table + an auto-summary. Nav and Home get links.

**Tech Stack:** Next.js 16, React 19, TypeScript, Tailwind 4, d3-scale/d3-shape/d3-array, vitest + Testing Library.

## Global Constraints

- Surface is `web/`; all commands run from `web/` (`cd web` first).
- **Next.js 16 has breaking changes** — per `web/AGENTS.md`, read the relevant guide in `node_modules/next/dist/docs/` before writing page/route code.
- Yields are handled in **decimals internally**; convert to bps/percent only at the display edge.
- Continuous compounding for all duration math (matches `lib/finance.ts`).
- Running-yield carry convention: `carry = y(T)·h`. Static-curve roll-down. No forward-based carry, no repo/leverage.
- Accent color `#00d68f`; use existing CSS vars (`--muted`, `--accent`, `--panel-border`, etc.) for styling consistency.
- Run tests with `npm test` (vitest). Lint with `npm run lint`.

---

### Task 1: Carry & roll-down math module

**Files:**
- Create: `web/lib/carry.ts`
- Test: `web/lib/carry.test.ts`

**Interfaces:**
- Consumes: `bondCashflows`, `modifiedDuration` from `@/lib/finance`.
- Produces:
  - `interface CarryPoint { label: string; years: number; yieldPct: number; carryBps: number; rollBps: number; totalBps: number; breakevenBps: number; totalPct: number; }`
  - `function carryRollDown(curve: { years: number; yieldPct: number; label: string }[], horizonYears: number): CarryPoint[]`

- [ ] **Step 1: Write the failing tests**

Create `web/lib/carry.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { carryRollDown } from "@/lib/carry";

const FLAT = [
  { years: 1, yieldPct: 5, label: "1Y" },
  { years: 2, yieldPct: 5, label: "2Y" },
  { years: 5, yieldPct: 5, label: "5Y" },
  { years: 10, yieldPct: 5, label: "10Y" },
];

const UPWARD = [
  { years: 1, yieldPct: 1, label: "1Y" },
  { years: 2, yieldPct: 2, label: "2Y" },
  { years: 5, yieldPct: 3, label: "5Y" },
  { years: 10, yieldPct: 4, label: "10Y" },
];

const INVERTED = [
  { years: 1, yieldPct: 5, label: "1Y" },
  { years: 2, yieldPct: 4, label: "2Y" },
  { years: 5, yieldPct: 3, label: "5Y" },
  { years: 10, yieldPct: 2, label: "10Y" },
];

describe("carryRollDown", () => {
  it("returns empty for empty input", () => {
    expect(carryRollDown([], 0.25)).toEqual([]);
  });

  it("flat curve: roll-down ≈ 0 and total ≈ carry", () => {
    const out = carryRollDown(FLAT, 0.25);
    expect(out.length).toBeGreaterThan(0);
    for (const p of out) {
      expect(p.rollBps).toBeCloseTo(0, 6);
      expect(p.totalBps).toBeCloseTo(p.carryBps, 6);
    }
  });

  it("flat 5% curve: carry over 3M ≈ 125 bps", () => {
    const p = carryRollDown(FLAT, 0.25).find((x) => x.label === "10Y")!;
    expect(p.carryBps).toBeCloseTo(125, 6); // 0.05 * 0.25 = 0.0125 = 125 bps
  });

  it("upward curve: roll-down is positive", () => {
    for (const p of carryRollDown(UPWARD, 0.25)) expect(p.rollBps).toBeGreaterThan(0);
  });

  it("inverted curve: roll-down is negative", () => {
    for (const p of carryRollDown(INVERTED, 0.25)) expect(p.rollBps).toBeLessThan(0);
  });

  it("carry scales linearly with horizon", () => {
    const a = carryRollDown(FLAT, 0.25).find((x) => x.label === "10Y")!;
    const b = carryRollDown(FLAT, 0.5).find((x) => x.label === "10Y")!;
    expect(b.carryBps).toBeCloseTo(a.carryBps * 2, 6);
  });

  it("breakeven sign matches total sign", () => {
    for (const p of carryRollDown(UPWARD, 0.25)) {
      expect(Math.sign(p.breakevenBps)).toBe(Math.sign(p.totalBps));
    }
  });

  it("excludes tenors with maturity ≤ horizon", () => {
    const curve = [
      { years: 0.5, yieldPct: 5, label: "6M" },
      { years: 1, yieldPct: 5, label: "1Y" },
      { years: 2, yieldPct: 5, label: "2Y" },
    ];
    const labels = carryRollDown(curve, 1).map((p) => p.label);
    expect(labels).not.toContain("6M");
    expect(labels).not.toContain("1Y"); // 1Y == horizon, excluded
    expect(labels).toContain("2Y");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd web && npm test -- carry`
Expected: FAIL — `carryRollDown` not found / module missing.

- [ ] **Step 3: Implement `web/lib/carry.ts`**

```ts
import { bondCashflows, modifiedDuration } from "@/lib/finance";

export interface CarryPoint {
  label: string;
  years: number;
  yieldPct: number; // y(T) in percent
  carryBps: number; // running-yield carry over the horizon, in bps of return
  rollBps: number; // roll-down return over the horizon, in bps of return
  totalBps: number; // carryBps + rollBps
  breakevenBps: number; // yield sell-off (bps) that zeroes the horizon return
  totalPct: number; // total horizon return, percent
}

export interface CurveInput {
  years: number;
  yieldPct: number;
  label: string;
}

/** Linear interpolation of y at x over sorted (xs, ys); flat-extrapolated.
 *  Mirrors the interp convention in lib/curve.ts. */
function interp(x: number, xs: number[], ys: number[]): number {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) {
      const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
      return ys[i - 1] + (ys[i] - ys[i - 1]) * t;
    }
  }
  return ys[ys.length - 1];
}

/**
 * Carry & roll-down per tenor over a holding horizon, assuming a STATIC curve.
 *   carry = y(T)·h                       (running-yield income over horizon)
 *   roll  = Dur(T−h)·(y(T) − y(T−h))     (price gain as the bond ages down the curve)
 * Dur is modified duration (continuous compounding) of a par bond at maturity T−h.
 * Breakeven = total / Dur = bps yields can rise over the horizon before return = 0.
 * Tenors with maturity ≤ horizon are excluded (a bond shorter than the horizon
 * cannot roll for the full period).
 */
export function carryRollDown(curve: CurveInput[], horizonYears: number): CarryPoint[] {
  const clean = curve
    .filter((p) => Number.isFinite(p.years) && Number.isFinite(p.yieldPct))
    .sort((a, b) => a.years - b.years);
  if (clean.length === 0) return [];

  const xs = clean.map((p) => p.years);
  const ys = clean.map((p) => p.yieldPct);
  const h = horizonYears;
  const eps = 1e-9;
  const out: CarryPoint[] = [];

  for (const p of clean) {
    if (p.years <= h + eps) continue;
    const yT = p.yieldPct / 100;
    const rollMat = p.years - h;
    const yRoll = interp(rollMat, xs, ys) / 100;

    // Modified duration of the rolled bond, priced as a par bond at its own yield.
    const { cashflows, times } = bondCashflows(100, yRoll, rollMat, 2);
    const dur = modifiedDuration(cashflows, times, yRoll);

    const carry = yT * h;
    const roll = dur * (yT - yRoll);
    const total = carry + roll;
    const breakeven = dur > eps ? total / dur : 0;

    out.push({
      label: p.label,
      years: p.years,
      yieldPct: p.yieldPct,
      carryBps: carry * 10_000,
      rollBps: roll * 10_000,
      totalBps: total * 10_000,
      breakevenBps: breakeven * 10_000,
      totalPct: total * 100,
    });
  }
  return out;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd web && npm test -- carry`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
cd web && git add lib/carry.ts lib/carry.test.ts
git commit -m "feat(web): carry & roll-down math lib with tests"
```

---

### Task 2: Reusable diverging category bar chart

**Files:**
- Create: `web/components/charts/CategoryBarChart.tsx`
- Test: `web/components/charts/CategoryBarChart.test.tsx`

**Interfaces:**
- Consumes: `useResizeObserver` from `@/components/charts/useResizeObserver`; `scaleBand`, `scaleLinear` from `d3-scale`.
- Produces:
  - `interface BarSeries { id: string; label: string; color: string; values: (number | null)[]; }`
  - `function CategoryBarChart(props: { categories: string[]; series: BarSeries[]; ariaLabel: string; yUnit?: string; width?: number; height?: number; }): JSX.Element`
  - Behavior: signed values stack diverging from a zero baseline (positives up, negatives down); `series[i].values[j]` aligns with `categories[j]`; `null` is treated as 0.

- [ ] **Step 1: Write the failing test**

Create `web/components/charts/CategoryBarChart.test.tsx`:

```tsx
import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { CategoryBarChart } from "@/components/charts/CategoryBarChart";

describe("CategoryBarChart", () => {
  it("renders an svg with one rect per (series, category) cell", () => {
    const { container, getByRole } = render(
      <CategoryBarChart
        width={600}
        ariaLabel="test bars"
        categories={["2Y", "5Y", "10Y"]}
        series={[
          { id: "carry", label: "Carry", color: "#00d68f", values: [10, 20, 30] },
          { id: "roll", label: "Roll", color: "#5b8def", values: [5, -5, 15] },
        ]}
      />,
    );
    expect(getByRole("img", { name: "test bars" })).toBeTruthy();
    // 2 series × 3 categories = 6 bar rects (class "bar")
    expect(container.querySelectorAll("rect.bar").length).toBe(6);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npm test -- CategoryBarChart`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `web/components/charts/CategoryBarChart.tsx`**

```tsx
"use client";
import { useMemo } from "react";
import { scaleBand, scaleLinear } from "d3-scale";
import { useResizeObserver } from "@/components/charts/useResizeObserver";

export interface BarSeries {
  id: string;
  label: string;
  color: string;
  values: (number | null)[];
}

export interface CategoryBarChartProps {
  categories: string[];
  series: BarSeries[];
  ariaLabel: string;
  yUnit?: string;
  width?: number;
  height?: number;
}

const M = { top: 12, right: 16, bottom: 32, left: 52 };

export function CategoryBarChart(props: CategoryBarChartProps) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = props.width ?? measured;
  const height = props.height ?? 280;

  const c = useMemo(() => {
    if (width === 0 || props.categories.length === 0) return null;
    const iw = Math.max(1, width - M.left - M.right);
    const ih = Math.max(1, height - M.top - M.bottom);

    const val = (s: BarSeries, j: number) => s.values[j] ?? 0;

    // Diverging stack totals per category to size the y domain.
    let lo = 0;
    let hi = 0;
    props.categories.forEach((_, j) => {
      let pos = 0;
      let neg = 0;
      for (const s of props.series) {
        const v = val(s, j);
        if (v >= 0) pos += v;
        else neg += v;
      }
      hi = Math.max(hi, pos);
      lo = Math.min(lo, neg);
    });

    const x = scaleBand<string>().domain(props.categories).range([0, iw]).padding(0.25);
    const y = scaleLinear().domain([lo, hi]).range([ih, 0]).nice();

    // Pre-compute rects: stack positives upward from 0, negatives downward.
    interface Rect { key: string; x: number; y: number; w: number; h: number; fill: string; }
    const rects: Rect[] = [];
    const bw = x.bandwidth();
    props.categories.forEach((cat, j) => {
      let posAcc = 0;
      let negAcc = 0;
      for (const s of props.series) {
        const v = val(s, j);
        if (v === 0) continue;
        let y0: number;
        let y1: number;
        if (v >= 0) {
          y0 = y(posAcc);
          posAcc += v;
          y1 = y(posAcc);
        } else {
          y0 = y(negAcc);
          negAcc += v;
          y1 = y(negAcc);
        }
        const top = Math.min(y0, y1);
        rects.push({
          key: `${s.id}-${cat}`,
          x: (x(cat) ?? 0),
          y: top,
          w: bw,
          h: Math.max(1, Math.abs(y1 - y0)),
          fill: s.color,
        });
      }
    });

    return { iw, ih, x, y, rects, yTicks: y.ticks(5), bw };
  }, [props.categories, props.series, width, height]);

  const fmtY = (v: number) => `${Math.round(v)}${props.yUnit ?? ""}`;

  return (
    <div ref={ref} className="w-full">
      {c && (
        <svg width={width} height={height} role="img" aria-label={props.ariaLabel} className="overflow-visible">
          <g transform={`translate(${M.left},${M.top})`}>
            {c.yTicks.map((t) => (
              <g key={`y${t}`} transform={`translate(0,${c.y(t)})`}>
                <line x1={0} x2={c.iw} stroke="var(--grid)" />
                <text x={-8} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)" className="tabnum">
                  {fmtY(t)}
                </text>
              </g>
            ))}
            <line x1={0} x2={c.iw} y1={c.y(0)} y2={c.y(0)} stroke="var(--panel-border-strong)" />
            {c.rects.map((r) => (
              <rect key={r.key} className="bar" x={r.x} y={r.y} width={r.w} height={r.h} fill={r.fill} rx={1.5} />
            ))}
            {props.categories.map((cat) => (
              <text
                key={`x${cat}`}
                x={(c.x(cat) ?? 0) + c.bw / 2}
                y={c.ih + 18}
                textAnchor="middle"
                fontSize={11}
                fill="var(--muted)"
                className="tabnum"
              >
                {cat}
              </text>
            ))}
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

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npm test -- CategoryBarChart`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd web && git add components/charts/CategoryBarChart.tsx components/charts/CategoryBarChart.test.tsx
git commit -m "feat(web): reusable diverging CategoryBarChart"
```

---

### Task 3: Carry page (client + route)

**Files:**
- Create: `web/app/carry/CarryClient.tsx`
- Create: `web/app/carry/page.tsx`

**Interfaces:**
- Consumes: `carryRollDown`, `CarryPoint` from `@/lib/carry`; `CategoryBarChart` from `@/components/charts/CategoryBarChart`; `Card` from `@/components/ui/Card`; `Segmented` from `@/components/ui/Segmented`; `rowToCurve` from `@/lib/finance`; `iso` from `@/lib/format`; `YieldRow` from `@/lib/types`.
- Data: GET `/api/treasury/range?start=<iso>&end=<iso>` → `{ rows: YieldRow[] }` (same shape the Yield Curve page consumes); use the last row as the latest curve.

- [ ] **Step 1: Read the Next.js 16 page/metadata guide**

Run: `ls node_modules/next/dist/docs/` (from `web/`) and read the page/metadata guidance referenced by `web/AGENTS.md` before writing the route. Confirm `app/<route>/page.tsx` + client-component split matches the existing `app/pricing/page.tsx` pattern (it does).

- [ ] **Step 2: Implement `web/app/carry/CarryClient.tsx`**

```tsx
"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Segmented } from "@/components/ui/Segmented";
import { CategoryBarChart } from "@/components/charts/CategoryBarChart";
import { carryRollDown, CarryPoint } from "@/lib/carry";
import { rowToCurve } from "@/lib/finance";
import { iso } from "@/lib/format";
import { YieldRow } from "@/lib/types";

const HORIZONS = [
  { label: "1M", value: 1 / 12 },
  { label: "3M", value: 0.25 },
  { label: "6M", value: 0.5 },
  { label: "1Y", value: 1 },
];

function summarize(points: CarryPoint[], horizonLabel: string): string {
  if (points.length === 0) return "No tenors longer than the horizon in this snapshot.";
  const best = points.reduce((a, b) => (b.breakevenBps > a.breakevenBps ? b : a));
  const sign = best.breakevenBps >= 0 ? "+" : "";
  return `Best breakeven cushion over ${horizonLabel}: ${best.label} at ${sign}${best.breakevenBps.toFixed(0)} bps.`;
}

export function CarryClient() {
  const [rows, setRows] = useState<YieldRow[] | null>(null);
  const [error, setError] = useState(false);
  const [horizon, setHorizon] = useState(0.25);

  useEffect(() => {
    const end = new Date();
    const start = new Date();
    start.setDate(start.getDate() - 30);
    fetch(`/api/treasury/range?start=${iso(start)}&end=${iso(end)}`)
      .then((r) => r.json())
      .then((d) => setRows(d.rows ?? []))
      .catch(() => setError(true));
  }, []);

  const horizonLabel = HORIZONS.find((h) => h.value === horizon)?.label ?? "3M";

  const view = useMemo(() => {
    if (!rows || rows.length === 0) return null;
    const latest = rows[rows.length - 1];
    const curve = rowToCurve(latest).map((p) => ({ years: p.years, yieldPct: p.yield, label: p.label }));
    const points = carryRollDown(curve, horizon);
    return { date: latest.date, points };
  }, [rows, horizon]);

  if (error) return <p className="text-[var(--muted)]">Treasury data is unavailable right now.</p>;
  if (!view) return <p className="text-[var(--muted)]">Loading Treasury data…</p>;

  const cats = view.points.map((p) => p.label);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl">Carry & Roll-Down</h1>
        <Segmented
          ariaLabel="Holding horizon"
          options={HORIZONS}
          value={horizon}
          onChange={(v) => setHorizon(v as number)}
        />
      </div>

      <Card>
        <div className="mb-2 flex items-baseline justify-between">
          <h2 className="text-lg">Carry vs roll-down · {horizonLabel}</h2>
          <span className="tabnum text-xs text-[var(--faint)]">as of {view.date}</span>
        </div>
        <CategoryBarChart
          ariaLabel="Carry and roll-down per tenor, stacked, in basis points of horizon return"
          yUnit=" bps"
          categories={cats}
          series={[
            { id: "carry", label: "Carry", color: "#00d68f", values: view.points.map((p) => p.carryBps) },
            { id: "roll", label: "Roll-down", color: "#5b8def", values: view.points.map((p) => p.rollBps) },
          ]}
        />
        <p className="mt-2 text-sm text-[var(--muted)]">{summarize(view.points, horizonLabel)}</p>
      </Card>

      <Card>
        <h2 className="mb-2 text-lg">Breakeven cushion · {horizonLabel}</h2>
        <CategoryBarChart
          ariaLabel="Breakeven yield sell-off per tenor, in basis points"
          yUnit=" bps"
          categories={cats}
          series={[{ id: "be", label: "Breakeven", color: "#f5a623", values: view.points.map((p) => p.breakevenBps) }]}
        />
        <p className="mt-2 text-sm text-[var(--muted)]">
          How far yields can rise over the horizon before the position returns zero. Higher is more defensive.
        </p>
      </Card>

      <Card>
        <h2 className="mb-3 text-lg">Detail</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm tabnum">
            <thead>
              <tr className="text-left text-[var(--muted)]">
                <th className="py-1 pr-4 font-medium">Tenor</th>
                <th className="py-1 pr-4 font-medium">Yield</th>
                <th className="py-1 pr-4 font-medium">Carry (bps)</th>
                <th className="py-1 pr-4 font-medium">Roll (bps)</th>
                <th className="py-1 pr-4 font-medium">Total (bps)</th>
                <th className="py-1 pr-4 font-medium">Breakeven (bps)</th>
                <th className="py-1 pr-4 font-medium">Return</th>
              </tr>
            </thead>
            <tbody>
              {view.points.map((p) => (
                <tr key={p.label} className="border-t border-[var(--panel-border)]">
                  <td className="py-1.5 pr-4 text-[var(--text)]">{p.label}</td>
                  <td className="py-1.5 pr-4">{p.yieldPct.toFixed(2)}%</td>
                  <td className="py-1.5 pr-4">{p.carryBps.toFixed(0)}</td>
                  <td className="py-1.5 pr-4" style={{ color: p.rollBps >= 0 ? "var(--accent)" : "var(--err)" }}>
                    {p.rollBps >= 0 ? "+" : ""}{p.rollBps.toFixed(0)}
                  </td>
                  <td className="py-1.5 pr-4">{p.totalBps.toFixed(0)}</td>
                  <td className="py-1.5 pr-4">{p.breakevenBps >= 0 ? "+" : ""}{p.breakevenBps.toFixed(0)}</td>
                  <td className="py-1.5 pr-4">{p.totalPct.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-[var(--faint)]">
          Static-curve, unlevered. Carry = running yield × horizon; roll-down ≈ duration × (yield − rolled yield)
          on today&apos;s curve. Tenors with maturity ≤ horizon are omitted.
        </p>
      </Card>
    </div>
  );
}
```

> Note: if `--err` is not a defined CSS var, use `--e5484d`-equivalent existing token. Verify against `app/globals.css` in Step 4; the codebase uses `var(--accent)` for positive. If no negative token exists, use the literal `#e5484d` (same red used for the "6M ago" series).

- [ ] **Step 3: Implement `web/app/carry/page.tsx`**

```tsx
import { CarryClient } from "./CarryClient";

export const metadata = { title: "Carry & Roll-Down · BondViz" };

export default function CarryPage() {
  return <CarryClient />;
}
```

- [ ] **Step 4: Verify the negative-color token, build, and lint**

Run: `cd web && grep -n "\-\-err" app/globals.css` — if no match, replace `var(--err)` in `CarryClient.tsx` with the literal `#e5484d`.
Run: `cd web && npm run lint && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Manually verify the page renders**

Run: `cd web && npm run dev`, open `http://localhost:3000/carry`. Confirm: horizon control switches charts/table; upward curve shows positive roll-down; table populated.

- [ ] **Step 6: Commit**

```bash
cd web && git add app/carry/page.tsx app/carry/CarryClient.tsx
git commit -m "feat(web): carry & roll-down page with charts and table"
```

---

### Task 4: Navigation + Home links

**Files:**
- Modify: `web/components/Nav.tsx:5-11`
- Modify: `web/app/page.tsx:60-77`

**Interfaces:**
- Consumes: existing `LINKS` array in `Nav.tsx`; existing card grid `<section>` in `page.tsx`.

- [ ] **Step 1: Add the nav link**

In `web/components/Nav.tsx`, change the `LINKS` array to insert Carry after Bond Pricing:

```tsx
const LINKS = [
  { href: "/yield-curve", label: "Yield Curve" },
  { href: "/pricing", label: "Bond Pricing" },
  { href: "/carry", label: "Carry & Roll" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/pca", label: "PCA" },
  { href: "/stocks", label: "Stocks" },
];
```

- [ ] **Step 2: Add a Home card**

In `web/app/page.tsx`, inside the card `<section className="grid ...">` (the one at line ~60), add a third `<Link>` card after the Pricing card:

```tsx
        <Link href="/carry" className="group">
          <Card className="h-full transition-all group-hover:border-l-[var(--accent)] group-hover:bg-[var(--panel)]">
            <h3 className="flex items-center justify-between text-[var(--accent)]">
              Carry &amp; Roll-Down <span className="transition-transform group-hover:translate-x-1">→</span>
            </h3>
            <p className="mt-2 text-sm text-[var(--muted)]">Horizon carry, roll-down and breakeven cushion across the curve.</p>
          </Card>
        </Link>
```

- [ ] **Step 3: Build, lint, and verify navigation**

Run: `cd web && npm run lint && npx tsc --noEmit`
Expected: no errors.
Then `npm run dev` and confirm the nav shows "Carry & Roll", links to `/carry`, and the Home card appears and links correctly.

- [ ] **Step 4: Run the full test suite**

Run: `cd web && npm test`
Expected: all tests pass (including Task 1 & 2 tests).

- [ ] **Step 5: Commit**

```bash
cd web && git add components/Nav.tsx app/page.tsx
git commit -m "feat(web): link Carry & Roll-Down from nav and home"
```

---

## Self-Review

**Spec coverage:**
- `lib/carry.ts` math (carry/roll/total/breakeven, conventions) → Task 1 ✓
- Static-curve, running-yield, modified duration of rolled bond, linear interp → Task 1 ✓
- `/carry` page + client, horizon segmented control, fail-soft, reuses `/api/treasury/range` → Task 3 ✓
- Stacked carry/roll bars + breakeven bars + table + auto-summary → Tasks 2 & 3 ✓
- Deferred charting decision resolved: new reusable `CategoryBarChart` → Task 2 ✓
- Nav entry + Home card → Task 4 ✓
- Tests: upward/inverted/flat, carry scales with horizon, breakeven sign, T≤h excluded, empty → Task 1 ✓
- Error handling: empty curve → "unavailable"; Dur≈0 guard; T≤h skip → Tasks 1 & 3 ✓
- Out of scope items (leverage, TIPS, forward carry, custom bonds) → not built ✓

**Placeholder scan:** No TBD/TODO. The one conditional (`--err` token vs literal `#e5484d`) is resolved deterministically by a grep step in Task 3 Step 4 — not a placeholder.

**Type consistency:** `CarryPoint` fields (`carryBps`, `rollBps`, `totalBps`, `breakevenBps`, `totalPct`, `yieldPct`, `label`, `years`) used identically in Tasks 1 and 3. `BarSeries` (`id`/`label`/`color`/`values`) consistent between Tasks 2 and 3. `carryRollDown(curve, horizonYears)` signature matches its call site. `rowToCurve` returns `CurvePoint[]` with `.yield`/`.years`/`.label`, mapped to `carryRollDown`'s `{years, yieldPct, label}` input in Task 3. ✓
