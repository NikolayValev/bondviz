"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { Segmented } from "@/components/ui/Segmented";
import { Skeleton } from "@/components/ui/Skeleton";
import { LineChart, Series } from "@/components/charts/LineChart";
import { pca } from "@/lib/pca";
import { TENOR_YEARS } from "@/lib/finance";
import { SERIES } from "@/lib/chartColors";
import { YieldRow } from "@/lib/types";
import { iso } from "@/lib/format";
import { useUrlState } from "@/components/useUrlState";
import { oneOf, type Schema } from "@/lib/urlState";

const COMPONENT_COLORS = [SERIES[0], SERIES[1], SERIES[2]];

const LOOKBACKS = [
  { label: "1Y", value: 1 },
  { label: "2Y", value: 2 },
  { label: "5Y", value: 5 },
];
const URL_DEFAULTS = { lookback: "2Y" };
const URL_SCHEMA: Schema<typeof URL_DEFAULTS> = { lookback: oneOf(LOOKBACKS.map((l) => l.label)) };

export function PcaClient() {
  const [url, setUrl] = useUrlState(URL_SCHEMA, URL_DEFAULTS);
  const years = LOOKBACKS.find((l) => l.label === url.lookback)?.value ?? 2;
  const setYears = (v: number) => setUrl({ lookback: LOOKBACKS.find((l) => l.value === v)?.label ?? "2Y" });
  // Track which lookback each result/error belongs to so switching windows shows
  // the loading state again (derived, rather than synchronously resetting state
  // inside the effect) and stale in-flight responses are ignored.
  const [loaded, setLoaded] = useState<{ years: number; rows: YieldRow[] } | null>(null);
  const [erroredYears, setErroredYears] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const end = new Date();
    const start = new Date();
    start.setFullYear(start.getFullYear() - years);
    start.setDate(start.getDate() - 14);
    fetch(`/api/treasury/range?start=${iso(start)}&end=${iso(end)}`)
      .then((r) => r.json())
      .then((d) => {
        if (!cancelled) setLoaded({ years, rows: d.rows ?? [] });
      })
      .catch(() => {
        if (!cancelled) setErroredYears(years);
      });
    return () => {
      cancelled = true;
    };
  }, [years]);

  const rows = loaded?.years === years ? loaded.rows : null;
  const error = erroredYears === years;

  const result = useMemo(() => (rows ? pca(rows, 3) : null), [rows]);

  const charts = useMemo(() => {
    if (!result) return null;
    const loadingSeries: Series[] = result.loadings.map((vec, c) => ({
      id: `pc${c + 1}`,
      label: `PC${c + 1} · ${(result.explained[c] * 100).toFixed(1)}%`,
      color: COMPONENT_COLORS[c % COMPONENT_COLORS.length],
      points: result.tenors.map((t, j) => [TENOR_YEARS[t], vec[j]] as [number, number]),
    }));
    const scoreSeries: Series[] = result.loadings.map((_, c) => ({
      id: `pc${c + 1}-score`,
      label: `PC${c + 1}`,
      color: COMPONENT_COLORS[c % COMPONENT_COLORS.length],
      points: result.scores.map((s) => [new Date(s.date).getTime(), s.values[c]] as [number, number]),
    }));
    return { loadingSeries, scoreSeries };
  }, [result]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl">Yield Curve PCA</h1>
        <Segmented
          ariaLabel="Lookback window"
          options={LOOKBACKS}
          value={years}
          onChange={(v) => setYears(v as number)}
        />
      </div>
      <p className="text-[var(--muted)]">
        Principal-component analysis of standardized daily Treasury yields. The first three
        components typically map to the level, slope, and curvature of the curve.
      </p>

      {error && <p className="text-[var(--muted)]">Treasury data is unavailable right now.</p>}
      {!error && !rows && (
        // Sized like the loaded cards (variance, loadings, scores) so the footer doesn't jump.
        <div role="status" aria-label="Loading Treasury data" className="space-y-6">
          <Skeleton className="h-48" />
          <Skeleton className="h-96" />
          <Skeleton className="h-96" />
        </div>
      )}
      {!error && rows && !result && (
        <p className="text-[var(--muted)]">Not enough complete data in this window to run PCA.</p>
      )}

      {result && charts && (
        <>
          <Card>
            <h2 className="mb-3 text-lg">Explained variance</h2>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              {result.explained.map((e, c) => (
                <Metric key={c} label={`PC${c + 1}`} value={`${(e * 100).toFixed(1)}%`} tone="accent" />
              ))}
            </div>
            <p className="mt-3 text-sm text-[var(--muted)]">
              Top {result.explained.length} components explain{" "}
              {(result.explained.reduce((a, b) => a + b, 0) * 100).toFixed(1)}% of the standardized
              yield variance.
            </p>
          </Card>

          <Card>
            <h2 className="mb-2 text-lg">Factor loadings</h2>
            <LineChart
              ariaLabel="PCA factor loadings by maturity"
              exportTitle="PCA factor loadings"
              series={charts.loadingSeries}
              xLabel="Maturity (years)"
              yLabel="Loading"
              zeroBaseline
            />
            <p className="mt-2 text-sm text-[var(--muted)]">
              A roughly flat PC1 is a level shift; a monotonic PC2 is slope; a U-shaped PC3 is
              curvature.
            </p>
          </Card>

          <Card>
            <h2 className="mb-2 text-lg">Factor scores over time</h2>
            <LineChart
              ariaLabel="PCA factor scores over time"
              exportTitle="PCA factor scores"
              series={charts.scoreSeries}
              xType="time"
              xLabel="Date"
              yLabel="Score"
              zeroBaseline
            />
          </Card>
        </>
      )}
    </div>
  );
}
