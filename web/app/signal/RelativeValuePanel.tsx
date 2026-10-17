"use client";
import { useMemo } from "react";
import { Card } from "@/components/ui/Card";
import { Segmented } from "@/components/ui/Segmented";
import { spreadRelativeValue, type RelativeValue } from "@/lib/relativeValue";
import type { SpreadPoint } from "@/lib/signal";

export const RV_WINDOWS = [
  { label: "5Y", years: 5 },
  { label: "10Y", years: 10 },
  { label: "Since 1990", years: null },
] as const;
export type RvWindow = (typeof RV_WINDOWS)[number]["label"];

const bp = (pp: number) => `${pp >= 0 ? "+" : "−"}${Math.abs(pp * 100).toFixed(0)} bps`;
const sigma = (z: number) => `${z >= 0 ? "+" : "−"}${Math.abs(z).toFixed(1)}σ`;
function ordinal(n: number): string {
  const r = Math.round(n);
  const s = r % 100 >= 11 && r % 100 <= 13 ? "th" : ["th", "st", "nd", "rd"][r % 10] ?? "th";
  return `${r}${s}`;
}
function verdict(rv: RelativeValue): string {
  if (rv.z >= 1) return "steeper than usual";
  if (rv.z <= -1) return "flatter than usual";
  return "within its normal range";
}

/** Min–max track with the window mean (tick) and today (dot). Values are also given as text. */
function RangeBar({ rv }: { rv: RelativeValue }) {
  const span = rv.max - rv.min || 1;
  const pos = (v: number) => `${((v - rv.min) / span) * 100}%`;
  return (
    <div aria-hidden="true" className="relative mt-3 h-2 rounded-full bg-[var(--panel-2)]">
      {rv.min < 0 && rv.max > 0 && (
        <div className="absolute inset-y-0 rounded-l-full bg-[var(--neg)] opacity-25" style={{ left: 0, width: pos(0) }} />
      )}
      <div className="absolute -top-1 h-4 w-px bg-[var(--muted)]" style={{ left: pos(rv.mean) }} />
      <div
        className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-[var(--bg)] bg-[var(--accent)]"
        style={{ left: pos(rv.current) }}
      />
    </div>
  );
}

function Row({ name, rv, windowLabel }: { name: string; rv: RelativeValue | null; windowLabel: string }) {
  if (!rv) return <p className="text-sm text-[var(--muted)]">{name}: not enough data.</p>;
  const over = windowLabel === "Since 1990" ? "since 1990" : `over the last ${windowLabel.replace("Y", " years")}`;
  return (
    <div>
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h3 className="text-base text-[var(--text)]">{name}</h3>
        <div className="tabnum flex flex-wrap gap-x-5 text-sm">
          <span>
            <span className="text-[var(--muted)]">today </span>
            {bp(rv.current)}
          </span>
          <span>
            <span className="text-[var(--muted)]">z </span>
            {sigma(rv.z)}
          </span>
          <span>
            <span className="text-[var(--muted)]">percentile </span>
            {ordinal(rv.percentile)}
          </span>
        </div>
      </div>
      <RangeBar rv={rv} />
      <div className="tabnum mt-1.5 flex justify-between text-xs text-[var(--faint)]">
        <span>min {bp(rv.min)}</span>
        <span>mean {bp(rv.mean)} · σ {Math.abs(rv.std * 100).toFixed(0)} bps</span>
        <span>max {bp(rv.max)}</span>
      </div>
      <p className="mt-2 text-sm text-[var(--muted)]">
        {name} is in the {ordinal(rv.percentile)} percentile {over} ({sigma(rv.z)}): {verdict(rv)}.
      </p>
    </div>
  );
}

/** Today's 10y–3m and 2s10s against their own history: z-score, percentile and range. */
export function RelativeValuePanel({
  points,
  span,
  onSpan,
}: {
  points: SpreadPoint[];
  span: RvWindow;
  onSpan: (w: RvWindow) => void;
}) {
  const years = RV_WINDOWS.find((w) => w.label === span)?.years ?? 10;
  const rv = useMemo(
    () => ({ s10y3m: spreadRelativeValue(points, "s10y3m", years), s2s10s: spreadRelativeValue(points, "s2s10s", years) }),
    [points, years],
  );

  return (
    <Card>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg">Today vs history</h2>
          <p className="mt-1 text-sm text-[var(--muted)]">
            How unusual the current slope is: z-score and percentile against every trading day in the window.
          </p>
        </div>
        <Segmented
          ariaLabel="History window"
          options={RV_WINDOWS.map((w) => ({ label: w.label, value: w.label }))}
          value={span}
          onChange={(v) => onSpan(v as RvWindow)}
        />
      </div>
      <div className="mt-5 space-y-6">
        <Row name="10y–3m" rv={rv.s10y3m} windowLabel={span} />
        <Row name="2s10s" rv={rv.s2s10s} windowLabel={span} />
      </div>
    </Card>
  );
}
