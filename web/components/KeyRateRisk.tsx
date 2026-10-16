"use client";
import { useMemo, useState } from "react";
import { Segmented } from "@/components/ui/Segmented";
import { CategoryBarChart } from "@/components/charts/CategoryBarChart";
import { ACCENT } from "@/lib/chartColors";
import type { KeyRateDuration } from "@/lib/finance";
import { money } from "@/lib/format";

type Unit = "krd" | "dv01";

/** Bucketed rate risk: key-rate durations (or dollar DV01s) as bars, with the sum check. */
export function KeyRateRisk({ krds, subject }: { krds: KeyRateDuration[]; subject: string }) {
  const [unit, setUnit] = useState<Unit>("krd");

  const view = useMemo(() => {
    // Drop empty buckets past the last one carrying risk (a 5Y bond has none at 7Y–30Y).
    let last = krds.length - 1;
    while (last > 0 && Math.abs(krds[last].krd) < 1e-12) last--;
    const shown = krds.slice(0, last + 1);
    const total = krds.reduce((a, k) => a + k.krd, 0);
    const totalDv01 = krds.reduce((a, k) => a + k.dv01, 0);
    const top = krds.reduce((a, k) => (k.krd > a.krd ? k : a), krds[0]);
    return { shown, total, totalDv01, top };
  }, [krds]);

  const values = view.shown.map((k) => (unit === "krd" ? k.krd : k.dv01));
  // Ticks like 0, 0.5, 1.25: at most two decimals, trailing zeros dropped.
  const fmt = (v: number) => `${unit === "dv01" ? "$" : ""}${+v.toFixed(2)}`;

  return (
    <div>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg">Key-rate durations</h2>
          <p className="mt-1 max-w-xl text-sm text-[var(--muted)]">
            Where along the curve the rate risk sits: each bar is the price sensitivity to a 1bp move at
            one key tenor alone.
          </p>
        </div>
        <Segmented
          ariaLabel="Key-rate units"
          options={[
            { label: "Years", value: "krd" },
            { label: "DV01 $", value: "dv01" },
          ]}
          value={unit}
          onChange={(v) => setUnit(v as Unit)}
        />
      </div>
      <div className="mt-4">
        <CategoryBarChart
          ariaLabel={`Key-rate ${unit === "krd" ? "durations in years" : "DV01 in dollars"} of ${subject} by tenor`}
          categories={view.shown.map((k) => k.label)}
          series={[{ id: unit, label: unit === "krd" ? "Key-rate duration" : "Key-rate DV01", color: ACCENT, values }]}
          formatY={fmt}
          height={240}
        />
      </div>
      <p className="mt-3 text-sm text-[var(--muted)]">
        Largest bucket <span className="tabnum text-[var(--text)]">{view.top.label}</span>. Buckets sum to{" "}
        <span className="tabnum text-[var(--text)]">{view.total.toFixed(2)} yrs</span> /{" "}
        <span className="tabnum text-[var(--text)]">${money(view.totalDv01)}</span> per bp, the same as the
        parallel duration and DV01.
      </p>
    </div>
  );
}
