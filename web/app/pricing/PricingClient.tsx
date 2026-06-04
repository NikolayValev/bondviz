"use client";
import { useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { Segmented } from "@/components/ui/Segmented";
import { ScenarioChart } from "@/components/charts/ScenarioChart";
import {
  bondCashflows,
  convexity,
  modifiedDuration,
  priceFromCashflows,
  scenarioShift,
  symmetricShifts,
} from "@/lib/finance";

const money = (v: number) =>
  v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const signedMoney = (v: number) => `${v >= 0 ? "+" : "−"}${money(Math.abs(v))}`;
const signedPct = (v: number) => `${v >= 0 ? "+" : "−"}${(Math.abs(v) * 100).toFixed(2)}%`;

function Field({
  label,
  unit,
  value,
  onChange,
  step,
  min,
  max,
  slider = false,
}: {
  label: string;
  unit?: string;
  value: number;
  onChange: (v: number) => void;
  step: number;
  min?: number;
  max?: number;
  slider?: boolean;
}) {
  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <div className="mt-1.5 flex items-center rounded-[var(--radius)] border border-[var(--panel-border)] bg-[var(--bg)] focus-within:border-[var(--accent)]">
        <input
          type="number"
          value={Number.isFinite(value) ? value : ""}
          step={step}
          min={min}
          max={max}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="tabnum w-full bg-transparent px-3 py-2.5 text-[var(--text)] outline-none"
        />
        {unit && <span className="pr-3 text-sm text-[var(--faint)]">{unit}</span>}
      </div>
      {slider && (
        <input
          type="range"
          value={Number.isFinite(value) ? value : min ?? 0}
          step={step}
          min={min}
          max={max}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="mt-2 w-full accent-[var(--accent)]"
          aria-label={`${label} slider`}
        />
      )}
    </label>
  );
}

const FREQ = [
  { label: "Annual", value: 1 },
  { label: "Semi", value: 2 },
  { label: "Quarterly", value: 4 },
];
const MAGNITUDES = [25, 50, 100, 200];

export function PricingClient() {
  const [face, setFace] = useState(1000);
  const [couponPct, setCouponPct] = useState(5);
  const [yieldPct, setYieldPct] = useState(4);
  const [years, setYears] = useState(10);
  const [freq, setFreq] = useState(2);
  const [mags, setMags] = useState<number[]>([25, 50, 100]);

  const safe = (n: number) => (Number.isFinite(n) ? n : 0);

  const model = useMemo(() => {
    const coupon = safe(couponPct) / 100;
    const y = safe(yieldPct) / 100;
    const yrs = Math.max(0.5, safe(years));
    const { cashflows, times } = bondCashflows(safe(face), coupon, yrs, freq);
    const pv = priceFromCashflows(cashflows, times, y);
    const dur = modifiedDuration(cashflows, times, y);
    const conv = convexity(cashflows, times, y);
    const dropFor1pct = (dur * 0.01 - 0.5 * conv * 0.01 * 0.01) * 100;

    const shifts = symmetricShifts(mags);
    const rows = scenarioShift({ face: safe(face), coupon, yield_: y, years: yrs, freq }, shifts);
    return { pv, dur, conv, dropFor1pct, rows };
  }, [face, couponPct, yieldPct, years, freq, mags]);

  return (
    <div className="space-y-8">
      <header className="rise">
        <div className="eyebrow">Fixed income · continuous compounding</div>
        <h1 className="mt-1.5 text-3xl sm:text-4xl">Bond Pricing &amp; Risk</h1>
        <p className="mt-2 max-w-2xl text-[var(--muted)]">
          Present value, duration and convexity for a fixed-coupon bond, plus a parallel-shift
          scenario showing how the quadratic approximation tracks a full reprice.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,340px)_1fr]">
        {/* Inputs */}
        <Card className="rise" >
          <h2 className="mb-4 text-lg">Inputs</h2>
          <div className="grid grid-cols-2 gap-4">
            <Field label="Face value" unit="$" value={face} step={100} min={0} onChange={setFace} />
            <Field label="Maturity" unit="yr" value={years} step={1} min={0.5} max={30} onChange={setYears} slider />
            <Field label="Coupon" unit="%" value={couponPct} step={0.25} min={0} max={15} onChange={setCouponPct} slider />
            <Field label="Yield" unit="%" value={yieldPct} step={0.25} min={0} max={15} onChange={setYieldPct} slider />
          </div>
          <div className="mt-5">
            <span className="eyebrow">Coupon frequency</span>
            <div className="mt-2">
              <Segmented
                ariaLabel="Coupon frequency"
                options={FREQ}
                value={freq}
                onChange={(v) => setFreq(v as number)}
              />
            </div>
          </div>
        </Card>

        {/* Results */}
        <Card className="rise">
          <h2 className="mb-4 text-lg">Valuation</h2>
          <Metric label="Present value" value={money(model.pv)} tone="accent" size="lg" />
          <div className="mt-4 grid grid-cols-2 gap-3">
            <Metric label="Modified duration" value={`${model.dur.toFixed(2)} yrs`} />
            <Metric label="Convexity" value={model.conv.toFixed(2)} />
          </div>
          <p className="mt-4 rounded-[var(--radius)] border border-[var(--panel-border)] bg-[var(--accent-dim)] px-4 py-3 text-sm text-[var(--text)]">
            A <span className="tabnum">1%</span> rise in yield ≈ a{" "}
            <span className="tabnum font-semibold text-[var(--accent)]">{model.dropFor1pct.toFixed(2)}%</span> drop in price
            <span className="text-[var(--muted)]"> (duration + convexity estimate).</span>
          </p>
        </Card>
      </div>

      {/* Scenario analysis */}
      <Card className="rise">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-lg">Scenario analysis</h2>
            <p className="mt-1 text-sm text-[var(--muted)]">Parallel yield shifts — full reprice vs. approximation.</p>
          </div>
          <Segmented
            ariaLabel="Shift magnitudes in basis points"
            multi
            options={MAGNITUDES.map((m) => ({ label: `±${m}`, value: m }))}
            value={mags}
            onChange={(v) => setMags(v as number[])}
          />
        </div>

        <div className="mt-6">
          <ScenarioChart rows={model.rows} basePrice={model.pv} />
        </div>

        {/* Table — real table on >= sm, stacked cards on mobile */}
        <div className="mt-6 hidden overflow-hidden rounded-[var(--radius)] border border-[var(--panel-border)] sm:block">
          <table className="w-full text-sm">
            <thead className="bg-[var(--panel-2)] text-left text-[var(--muted)]">
              <tr>
                <th className="px-4 py-2.5 font-medium">Shift</th>
                <th className="px-4 py-2.5 font-medium">New yield</th>
                <th className="px-4 py-2.5 text-right font-medium">New price</th>
                <th className="px-4 py-2.5 text-right font-medium">$ change</th>
                <th className="px-4 py-2.5 text-right font-medium">% change</th>
              </tr>
            </thead>
            <tbody className="tabnum">
              {model.rows.map((r) => {
                const tone = r.shiftBps === 0 ? "text-[var(--muted)]" : r.dollarChange >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]";
                return (
                  <tr key={r.shiftBps} className={`border-t border-[var(--panel-border)] ${r.shiftBps === 0 ? "bg-white/[0.02]" : ""}`}>
                    <td className="px-4 py-2.5">{r.shiftBps > 0 ? `+${r.shiftBps}` : r.shiftBps} bps</td>
                    <td className="px-4 py-2.5">{(r.newYield * 100).toFixed(3)}%</td>
                    <td className="px-4 py-2.5 text-right">{money(r.newPrice)}</td>
                    <td className={`px-4 py-2.5 text-right ${tone}`}>{r.shiftBps === 0 ? "—" : signedMoney(r.dollarChange)}</td>
                    <td className={`px-4 py-2.5 text-right ${tone}`}>{r.shiftBps === 0 ? "—" : signedPct(r.pctChange)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <ul className="mt-6 grid grid-cols-1 gap-2 sm:hidden">
          {model.rows.map((r) => {
            const tone = r.shiftBps === 0 ? "text-[var(--muted)]" : r.dollarChange >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]";
            return (
              <li key={r.shiftBps} className="flex items-center justify-between rounded-[var(--radius)] border border-[var(--panel-border)] bg-[var(--panel-2)] px-4 py-3">
                <div>
                  <div className="tabnum font-semibold">{r.shiftBps > 0 ? `+${r.shiftBps}` : r.shiftBps} bps</div>
                  <div className="tabnum text-xs text-[var(--muted)]">y {(r.newYield * 100).toFixed(3)}%</div>
                </div>
                <div className="text-right">
                  <div className="tabnum">{money(r.newPrice)}</div>
                  <div className={`tabnum text-xs ${tone}`}>{r.shiftBps === 0 ? "—" : `${signedPct(r.pctChange)}`}</div>
                </div>
              </li>
            );
          })}
        </ul>
      </Card>
    </div>
  );
}
