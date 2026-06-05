"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { Segmented } from "@/components/ui/Segmented";
import { LineChart, Series } from "@/components/charts/LineChart";
import { pca } from "@/lib/pca";
import { TENOR_YEARS } from "@/lib/finance";
import { YieldRow } from "@/lib/types";

const COMPONENT_COLORS = ["#00d68f", "#5b8def", "#f5a623"];

function iso(d: Date) {
  return d.toISOString().slice(0, 10);
}

export function PcaClient() {
  const [years, setYears] = useState(2);
  const [rows, setRows] = useState<YieldRow[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    setRows(null);
    setError(false);
    const end = new Date();
    const start = new Date();
    start.setFullYear(start.getFullYear() - years);
    start.setDate(start.getDate() - 14);
    fetch(`/api/treasury/range?start=${iso(start)}&end=${iso(end)}`)
      .then((r) => r.json())
      .then((d) => setRows(d.rows ?? []))
      .catch(() => setError(true));
  }, [years]);

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
          options={[
            { label: "1Y", value: 1 },
            { label: "2Y", value: 2 },
            { label: "5Y", value: 5 },
          ]}
          value={years}
          onChange={(v) => setYears(v as number)}
        />
      </div>
      <p className="text-[var(--muted)]">
        Principal-component analysis of standardized daily Treasury yields. The first three
        components typically map to the level, slope, and curvature of the curve.
      </p>

      {error && <p className="text-[var(--muted)]">Treasury data is unavailable right now.</p>}
      {!error && !rows && <p className="text-[var(--muted)]">Loading Treasury data…</p>}
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
