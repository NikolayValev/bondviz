"use client";
import { useMemo } from "react";
import { Card } from "@/components/ui/Card";
import { Segmented } from "@/components/ui/Segmented";
import { LineChart, Series } from "@/components/charts/LineChart";
import { CategoryBarChart } from "@/components/charts/CategoryBarChart";
import { SERIES } from "@/lib/chartColors";
import { applyShock, benchmarkImpacts, shockBps, SHOCK_SIZES, SHOCKS, type ShockKind } from "@/lib/curveShocks";
import { CopyLinkButton } from "@/components/ui/CopyLinkButton";
import { signedPct } from "@/lib/format";
import { CurvePoint } from "@/lib/types";
import { ScrollRegion } from "@/components/ui/ScrollRegion";
import { ExportButtons } from "@/components/ui/ExportButtons";

const BENCHMARKS = ["2Y", "5Y", "10Y", "30Y"];

const signedBps = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(0)} bps`;

/** Apply a named shock to today's curve: overlay, per-tenor shifts, and benchmark reprices. */
export function ShockScenario({
  curve,
  date,
  kind,
  sizeBps,
  onChange,
  shareUrl,
}: {
  curve: CurvePoint[];
  date: string;
  kind: ShockKind;
  sizeBps: number;
  onChange: (next: { kind: ShockKind; sizeBps: number }) => void;
  /** When given, shows a "copy link" button for this scenario. */
  shareUrl?: () => string;
}) {
  const view = useMemo(() => {
    const spec = { kind, sizeBps };
    const shocked = applyShock(curve, spec);
    const series: Series[] = [
      { id: "today", label: `Today (${date})`, color: SERIES[0], points: curve.map((p) => [p.years, p.yield]) },
      { id: "shocked", label: "Shocked", color: SERIES[2], points: shocked.map((p) => [p.years, p.yield]) },
    ];
    const shifts = curve.map((p) => shockBps(spec, p.years));
    return {
      series,
      categories: curve.map((p) => p.label),
      // Split by sign so rises and falls get their own color; drop a side that never occurs
      // so the legend doesn't advertise it.
      bars: [
        { id: "up", label: "Yield up", color: SERIES[3], values: shifts.map((v) => (v > 0 ? v : null)) },
        { id: "down", label: "Yield down", color: SERIES[1], values: shifts.map((v) => (v < 0 ? v : null)) },
      ].filter((b) => b.values.some((v) => v !== null)),
      impacts: benchmarkImpacts(curve, spec, BENCHMARKS),
    };
  }, [curve, date, kind, sizeBps]);

  const meta = SHOCKS.find((s) => s.kind === kind);

  return (
    <Card>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg">Scenario: shock the curve</h2>
          <p className="mt-1 text-sm text-[var(--muted)]">
            Apply a named move to today&apos;s curve and see what it does to each tenor and to benchmark bonds.
          </p>
        </div>
        {shareUrl && <CopyLinkButton getUrl={shareUrl} />}
      </div>

      <div className="mt-4 space-y-3">
        <Segmented
          ariaLabel="Shock shape"
          variant="chips"
          options={SHOCKS.map((s) => ({ label: s.label, value: s.kind }))}
          value={kind}
          onChange={(v) => onChange({ kind: v as ShockKind, sizeBps })}
        />
        <div className="flex flex-wrap items-center gap-3">
          <span className="eyebrow">Size</span>
          <Segmented
            ariaLabel="Shock size in basis points"
            options={SHOCK_SIZES.map((s) => ({ label: `${s} bp`, value: s }))}
            value={sizeBps}
            onChange={(v) => onChange({ kind, sizeBps: v as number })}
          />
        </div>
        {meta && <p className="text-sm text-[var(--text)]">{meta.description}</p>}
      </div>

      <div className="mt-4 grid gap-6 lg:grid-cols-2">
        <div>
          <div className="eyebrow mb-2">Curve</div>
          <LineChart
            ariaLabel={`Today's yield curve and the curve after a ${sizeBps} bp ${meta?.label ?? kind} shock`}
            exportTitle={`${sizeBps} bp ${meta?.label ?? kind} curve`}
            series={view.series}
            xLabel="Maturity (years)"
            xScale="sqrt"
            yLabel="Yield (%)"
            yUnit="%"
            height={260}
          />
        </div>
        <div>
          <div className="eyebrow mb-2">Shift by tenor</div>
          <CategoryBarChart
            ariaLabel={`Yield shift in basis points at each tenor for a ${sizeBps} bp ${meta?.label ?? kind} shock`}
            exportTitle={`${sizeBps} bp ${meta?.label ?? kind} shift by tenor`}
            categoryLabel="Tenor"
            categories={view.categories}
            series={view.bars}
            yUnit=" bp"
            height={260}
          />
        </div>
      </div>

      <ScrollRegion label="Benchmark bond reprices" className="mt-6">
        <table className="w-full text-sm tabnum">
          <caption className="eyebrow mb-2 text-left">Benchmark bonds (coupon = today&apos;s yield)</caption>
          <thead className="text-left text-[var(--muted)]">
            <tr>
              <th className="py-1.5 pr-4 font-medium">Bond</th>
              <th className="py-1.5 pr-4 font-medium">Yield today</th>
              <th className="py-1.5 pr-4 font-medium">Shocked</th>
              <th className="py-1.5 pr-4 font-medium">Shift</th>
              <th className="py-1.5 pr-4 font-medium">Price change</th>
              <th className="py-1.5 font-medium">D + C estimate</th>
            </tr>
          </thead>
          <tbody>
            {view.impacts.map((b) => (
              <tr key={b.label} className="border-t border-[var(--panel-border)]">
                <td className="py-1.5 pr-4 text-[var(--text)]">{b.label}</td>
                <td className="py-1.5 pr-4">{b.yieldBefore.toFixed(2)}%</td>
                <td className="py-1.5 pr-4">{b.yieldAfter.toFixed(2)}%</td>
                <td className="py-1.5 pr-4">{signedBps(b.shiftBps)}</td>
                <td className={`py-1.5 pr-4 ${b.pctChange >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]"}`}>
                  {signedPct(b.pctChange)}
                </td>
                <td className="py-1.5 text-[var(--muted)]">{signedPct(b.approxPctChange)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollRegion>
      <ExportButtons
        className="mt-3 justify-end"
        title={`${sizeBps} bp ${meta?.label ?? kind} benchmark reprices`}
        csv={() => ({
          columns: ["Bond", "Yield today (%)", "Shocked yield (%)", "Shift (bps)", "Price change (%)", "D + C estimate (%)"],
          rows: view.impacts.map((b) => [b.label, b.yieldBefore, b.yieldAfter, b.shiftBps, b.pctChange * 100, b.approxPctChange * 100]),
        })}
      />
    </Card>
  );
}
