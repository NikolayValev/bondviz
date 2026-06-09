"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { LineChart, Series } from "@/components/charts/LineChart";
import { rowToCurve, spreadSeries, describeCurve, toBps } from "@/lib/finance";
import { bootstrapZeros } from "@/lib/curve";
import { Heatmap } from "@/components/charts/Heatmap";
import { YieldRow } from "@/lib/types";
import { iso } from "@/lib/format";

const COMPARE = [
  { label: "1M ago", months: 1, color: "#5b8def" },
  { label: "3M ago", months: 3, color: "#f5a623" },
  { label: "6M ago", months: 6, color: "#e5484d" },
  { label: "1Y ago", months: 12, color: "#9b59b6" },
];

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
    start.setDate(start.getDate() - 14);
    fetch(`/api/treasury/range?start=${iso(start)}&end=${iso(end)}`)
      .then((r) => r.json())
      .then((d) => setRows(d.rows ?? []))
      .catch(() => setError(true));
  }, []);

  const view = useMemo(() => {
    if (!rows || rows.length === 0) return null;
    const latest = rows[rows.length - 1];
    const latestCurve = rowToCurve(latest);
    const boot = bootstrapZeros(latestCurve.map((p) => ({ years: p.years, yieldPct: p.yield })));
    const bootstrapSeries: Series[] = [
      { id: "par", label: "Par", color: "#00d68f", points: latestCurve.map((p) => [p.years, p.yield]) },
      { id: "zero", label: "Zero", color: "#5b8def", points: boot.grid.map((t, i) => [t, boot.zero[i] * 100]) },
      { id: "fwd", label: "Forward (6M)", color: "#f5a623", points: boot.grid.map((t, i) => [t, boot.forward[i] * 100]) },
    ];

    const tenorsPresent = latestCurve.map((p) => p.label);
    const heatmap = {
      dates: rows.map((r) => r.date),
      tenors: tenorsPresent,
      values: rows.map((r) => {
        const m = new Map(rowToCurve(r).map((p) => [p.label, p.yield] as const));
        return tenorsPresent.map((lab) => m.get(lab) ?? null);
      }),
    };

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
      { id: "2s10s", label: "2s10s", color: "#00d68f", points: twos10s.map(([t, v]) => [t, toBps(v)]) },
      { id: "3m10y", label: "3m10y", color: "#5b8def", points: threeM10Y.map(([t, v]) => [t, toBps(v)]) },
    ];

    return { latest, latestCurve, curveSeries, spreadSeriesData, bootstrapSeries, heatmap };
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
        <h2 className="mb-2 text-lg">Zero & forward curve (bootstrapped)</h2>
        <LineChart
          ariaLabel="Bootstrapped zero and forward curves vs par"
          series={view.bootstrapSeries}
          xLabel="Maturity (years)"
          yLabel="Rate (%)"
          yUnit="%"
        />
        <p className="mt-2 text-sm text-[var(--muted)]">
          Par yields are treated as semiannual par coupons and bootstrapped into discount factors;
          zero rates are annual-compounded, and forwards are the implied 6-month rates.
        </p>
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

      <Card>
        <h2 className="mb-2 text-lg">Yield heatmap (time × tenor)</h2>
        <Heatmap
          ariaLabel="Treasury yields by tenor over time"
          dates={view.heatmap.dates}
          tenors={view.heatmap.tenors}
          values={view.heatmap.values}
        />
        <p className="mt-2 text-sm text-[var(--muted)]">
          Each row is a date (old → new), each column a tenor; brighter cells are higher yields.
          Watch horizontal bands for tenor-specific moves and diagonal gradients for rolling
          steepeners or flatteners.
        </p>
      </Card>
    </div>
  );
}
