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
        <h1 className="text-2xl">Carry &amp; Roll-Down</h1>
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
            { id: "carry", label: "Carry", color: "var(--accent)", values: view.points.map((p) => p.carryBps) },
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
          series={[{ id: "be", label: "Breakeven", color: "var(--warn)", values: view.points.map((p) => p.breakevenBps) }]}
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
                  <td className="py-1.5 pr-4" style={{ color: p.rollBps >= 0 ? "var(--pos)" : "var(--neg)" }}>
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
