"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { LineChart, Series } from "@/components/charts/LineChart";
import { rowToCurve, spreadSeries, describeCurve, toBps } from "@/lib/finance";
import { bootstrapZeros, type ParInterpolation } from "@/lib/curve";
import { SERIES } from "@/lib/chartColors";
import { Heatmap } from "@/components/charts/Heatmap";
import { YieldRow } from "@/lib/types";
import { iso } from "@/lib/format";
import { SHOCK_SIZES, SHOCKS, type ShockKind } from "@/lib/curveShocks";
import { useUrlState } from "@/components/useUrlState";
import { oneOf, type Schema } from "@/lib/urlState";
import { ShockScenario } from "./ShockScenario";
import { LazyMount } from "@/components/LazyMount";
import { nearestRow } from "@/lib/snapshot";
import { shiftMonths } from "@/lib/dates";
import { Segmented } from "@/components/ui/Segmented";

const COMPARE = [
  { label: "1M ago", months: 1, color: SERIES[1] },
  { label: "3M ago", months: 3, color: SERIES[2] },
  { label: "6M ago", months: 6, color: SERIES[3] },
  { label: "1Y ago", months: 12, color: SERIES[4] },
];

// The shock scenario and the bootstrap's par interpolation live in the URL:
// /yield-curve?shock=twist&bp=100&interp=linear
const URL_DEFAULTS = { shock: "bear-steepener" as ShockKind, bp: 50, interp: "pchip" as ParInterpolation };
const URL_SCHEMA: Schema<typeof URL_DEFAULTS> = {
  shock: oneOf(SHOCKS.map((s) => s.kind)),
  bp: oneOf(SHOCK_SIZES),
  interp: oneOf<ParInterpolation>(["pchip", "linear"]),
};

const INTERPOLATIONS = [
  { label: "Monotone cubic", value: "pchip" as const },
  { label: "Linear", value: "linear" as const },
];

export function YieldCurveClient() {
  const [rows, setRows] = useState<YieldRow[] | null>(null);
  const [error, setError] = useState(false);
  const [url, setUrl, shareUrl] = useUrlState(URL_SCHEMA, URL_DEFAULTS);

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
      { id: "latest", label: latest.date, color: SERIES[0], points: latestCurve.map((p) => [p.years, p.yield]) },
    ];
    for (const c of COMPARE) {
      const row = nearestRow(rows, shiftMonths(latest.date, -c.months));
      if (row) {
        const pts = rowToCurve(row).map((p) => [p.years, p.yield] as [number, number]);
        if (pts.length) curveSeries.push({ id: c.label, label: c.label, color: c.color, points: pts });
      }
    }

    const { twos10s, threeM10Y } = spreadSeries(rows);
    const spreadSeriesData: Series[] = [
      { id: "2s10s", label: "2s10s", color: SERIES[0], points: twos10s.map(([t, v]) => [t, toBps(v)]) },
      { id: "3m10y", label: "3m10y", color: SERIES[1], points: threeM10Y.map(([t, v]) => [t, toBps(v)]) },
    ];

    return { latest, latestCurve, curveSeries, spreadSeriesData, heatmap };
  }, [rows]);

  const latestCurve = view?.latestCurve;
  const bootstrapSeries = useMemo(() => {
    if (!latestCurve) return [];
    const boot = bootstrapZeros(
      latestCurve.map((p) => ({ years: p.years, yieldPct: p.yield })),
      { interpolation: url.interp },
    );
    const series: Series[] = [
      { id: "par", label: "Par", color: SERIES[0], points: latestCurve.map((p) => [p.years, p.yield]) },
      { id: "zero", label: "Zero", color: SERIES[1], points: boot.grid.map((t, i) => [t, boot.zero[i] * 100]) },
      { id: "fwd", label: "Forward (6M)", color: SERIES[2], points: boot.grid.map((t, i) => [t, boot.forward[i] * 100]) },
    ];
    return series;
  }, [latestCurve, url.interp]);

  // The heading renders in every state so it's in the prerendered HTML (it's the LCP element)
  // and the page keeps a heading while loading or on error.
  const header = <h1 className="text-2xl">Yield Curve Explorer</h1>;
  if (error)
    return (
      <div className="space-y-6">
        {header}
        <p className="text-[var(--muted)]">Treasury data is unavailable right now.</p>
      </div>
    );
  if (!view)
    return (
      <div className="space-y-6">
        {header}
        <div role="status" aria-label="Loading Treasury data" className="space-y-6">
          <Skeleton className="h-96" />
          <Skeleton className="h-80" />
        </div>
      </div>
    );

  return (
    <div className="space-y-6">
      {header}

      <Card>
        <h2 className="mb-2 text-lg">Latest curve · {view.latest.date}</h2>
        <LineChart
          ariaLabel="Latest Treasury yield curve by maturity"
          exportTitle={`Treasury yield curve ${view.latest.date}`}
          series={[view.curveSeries[0]]}
          xLabel="Maturity (years)"
          xScale="sqrt"
          yLabel="Yield (%)"
          yUnit="%"
        />
        <p className="mt-2 text-sm text-[var(--muted)]">{describeCurve(view.latestCurve)}</p>
      </Card>

      <Card>
        <h2 className="mb-2 text-lg">Curve shifts vs the past</h2>
        <LineChart
          ariaLabel="Latest yield curve compared with prior periods"
          exportTitle="Curve shifts vs the past"
          series={view.curveSeries}
          xLabel="Maturity (years)"
          xScale="sqrt"
          yLabel="Yield (%)"
          yUnit="%"
        />
      </Card>

      <ShockScenario
        curve={view.latestCurve}
        date={view.latest.date}
        kind={url.shock}
        sizeBps={url.bp}
        onChange={({ kind, sizeBps }) => setUrl({ shock: kind, bp: sizeBps })}
        shareUrl={shareUrl}
      />

      <Card>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg">Zero & forward curve (bootstrapped)</h2>
          <Segmented
            ariaLabel="Par interpolation between tenors"
            options={INTERPOLATIONS}
            value={url.interp}
            onChange={(v) => setUrl({ interp: v as ParInterpolation })}
          />
        </div>
        <LazyMount minHeight={320}>
          <LineChart
            ariaLabel={`Bootstrapped zero and forward curves vs par, ${url.interp === "pchip" ? "monotone cubic" : "linear"} par interpolation`}
            series={bootstrapSeries}
            exportTitle={`Zero and forward curve (${url.interp === "pchip" ? "monotone cubic" : "linear"})`}
            xLabel="Maturity (years)"
            xScale="sqrt"
            yLabel="Rate (%)"
            yUnit="%"
          />
        </LazyMount>
        <p className="mt-2 text-sm text-[var(--muted)]">
          Par yields are treated as semiannual par coupons and bootstrapped into discount factors;
          zero rates are annual-compounded, and forwards are the implied 6-month rates. Between
          quoted tenors the par curve is filled in with a monotone cubic by default. Linear
          interpolation (the Python reference&apos;s choice) bends the curve at every tenor, and the
          forward curve turns each bend into a jump, most visibly at 20Y.
        </p>
      </Card>

      <Card>
        <h2 className="mb-2 text-lg">Key spreads over time</h2>
        <LazyMount minHeight={320}>
          <LineChart
            ariaLabel="2s10s and 3m10y spreads over time"
            exportTitle="Key spreads over time"
            series={view.spreadSeriesData}
            xType="time"
            xLabel="Date"
            yLabel="Spread (bps)"
            yUnit=""
            zeroBaseline
          />
        </LazyMount>
        <p className="mt-2 text-sm text-[var(--muted)]">
          Negative spreads indicate inversion. Lines below the dashed zero line mark inverted regimes.
        </p>
      </Card>

      <Card>
        <h2 className="mb-2 text-lg">Yield heatmap (time × tenor)</h2>
        <LazyMount minHeight={300}>
          <Heatmap
            ariaLabel="Treasury yields by tenor over time"
            exportTitle="Yield heatmap"
            dates={view.heatmap.dates}
            tenors={view.heatmap.tenors}
            values={view.heatmap.values}
            unit="%"
          />
        </LazyMount>
        <p className="mt-2 text-sm text-[var(--muted)]">
          Each row is a date (oldest at the top), each column a tenor; brighter cells are higher
          yields (see the scale below the chart).
          Watch horizontal bands for tenor-specific moves and diagonal gradients for rolling
          steepeners or flatteners.
        </p>
      </Card>
    </div>
  );
}
