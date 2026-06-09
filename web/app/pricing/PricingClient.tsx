"use client";
import { useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { Segmented } from "@/components/ui/Segmented";
import { ScenarioChart } from "@/components/charts/ScenarioChart";
import { PriceYieldChart } from "@/components/charts/PriceYieldChart";
import { CashflowChart } from "@/components/charts/CashflowChart";
import {
  bondCashflows,
  convexity,
  couponIncome,
  currentYield,
  discountedCashflows,
  dv01,
  horizonReturn,
  modifiedDuration,
  priceFromCashflows,
  priceYieldCurve,
  scenarioShift,
  solveYield,
  symmetricShifts,
} from "@/lib/finance";
import { money, money0, signedPct } from "@/lib/format";

const signedMoney = (v: number) => `${v >= 0 ? "+" : "−"}${money(Math.abs(v))}`;

function Field({
  label, unit, value, onChange, step, min, max, slider = false, disabled = false,
}: {
  label: string; unit?: string; value: number; onChange: (v: number) => void;
  step: number; min?: number; max?: number; slider?: boolean; disabled?: boolean;
}) {
  return (
    <label className={`block ${disabled ? "opacity-50" : ""}`}>
      <span className="eyebrow">{label}</span>
      <div className="mt-1.5 flex items-center rounded-[var(--radius)] border border-[var(--panel-border)] bg-[var(--bg)] focus-within:border-[var(--accent)]">
        <input
          type="number"
          value={Number.isFinite(value) ? value : ""}
          step={step} min={min} max={max} disabled={disabled}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="tabnum w-full bg-transparent px-3 py-2.5 text-[var(--text)] outline-none disabled:cursor-not-allowed"
        />
        {unit && <span className="pr-3 text-sm text-[var(--faint)]">{unit}</span>}
      </div>
      {slider && (
        <input
          type="range" value={Number.isFinite(value) ? value : min ?? 0}
          step={step} min={min} max={max} disabled={disabled}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="mt-2 w-full accent-[var(--accent)]" aria-label={`${label} slider`}
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
const MODE = [
  { label: "From yield", value: "price" },
  { label: "From price", value: "yield" },
];

export function PricingClient() {
  const [face, setFace] = useState(1000);
  const [couponPct, setCouponPct] = useState(5);
  const [yieldPct, setYieldPct] = useState(4);
  const [years, setYears] = useState(10);
  const [freq, setFreq] = useState(2);
  const [mode, setMode] = useState<"price" | "yield">("price");
  const [marketPrice, setMarketPrice] = useState(1000);
  const [mags, setMags] = useState<number[]>([25, 50, 100]);
  const [horizon, setHorizon] = useState(3);
  const [horizonShift, setHorizonShift] = useState(0);

  const safe = (n: number) => (Number.isFinite(n) ? n : 0);

  const m = useMemo(() => {
    const coupon = safe(couponPct) / 100;
    const yrs = Math.max(0.5, safe(years));
    const f = safe(face);
    const y = mode === "price" ? safe(yieldPct) / 100 : solveYield(f, coupon, yrs, freq, Math.max(1, safe(marketPrice)));

    const { cashflows, times } = bondCashflows(f, coupon, yrs, freq);
    const pv = priceFromCashflows(cashflows, times, y);
    const dur = modifiedDuration(cashflows, times, y);
    const conv = convexity(cashflows, times, y);
    const dollar01 = dv01(cashflows, times, y);
    const curY = currentYield(f, coupon, pv);
    const annualCoupon = couponIncome(f, coupon);
    const dropFor1pct = (dur * 0.01 - 0.5 * conv * 0.0001) * 100;

    // Price–yield bowl centred on the current yield.
    const lo = Math.max(0.0005, y - 0.05);
    const hi = y + 0.05;
    const yields = Array.from({ length: 61 }, (_, i) => lo + ((hi - lo) * i) / 60);
    const pyCurve = priceYieldCurve(f, coupon, yrs, freq, yields);

    const dcf = discountedCashflows(cashflows, times, y);
    const rows = scenarioShift({ face: f, coupon, yield_: y, years: yrs, freq }, symmetricShifts(mags));
    const hz = horizonReturn({ face: f, coupon, yield_: y, years: yrs, freq }, Math.min(horizon, yrs), horizonShift);

    return { y, pv, dur, conv, dollar01, curY, annualCoupon, dropFor1pct, pyCurve, dcf, rows, hz, yrs };
  }, [face, couponPct, yieldPct, years, freq, mode, marketPrice, mags, horizon, horizonShift]);

  const incomeShare = m.hz.totalValue > 0 ? (m.hz.couponIncome / m.hz.totalValue) * 100 : 0;

  return (
    <div className="space-y-8">
      <header className="rise">
        <div className="eyebrow">Fixed income · continuous compounding</div>
        <h1 className="mt-1.5 text-3xl sm:text-4xl">Bond Pricing &amp; Risk Lab</h1>
        <p className="mt-2 max-w-2xl text-[var(--muted)]">
          Price, duration, convexity, DV01, scenario reprices and holding-period returns — interactive,
          with hand-rolled D3. Move any input and every chart updates live.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,340px)_1fr]">
        {/* Inputs */}
        <Card className="rise">
          <h2 className="mb-4 text-lg">Inputs</h2>
          <div className="grid grid-cols-2 gap-4">
            <Field label="Face value" unit="$" value={face} step={100} min={0} onChange={setFace} />
            <Field label="Maturity" unit="yr" value={years} step={1} min={0.5} max={30} onChange={setYears} slider />
            <Field label="Coupon" unit="%" value={couponPct} step={0.25} min={0} max={15} onChange={setCouponPct} slider />
            <div>
              <span className="eyebrow">Coupon frequency</span>
              <div className="mt-1.5">
                <Segmented ariaLabel="Coupon frequency" options={FREQ} value={freq} onChange={(v) => setFreq(v as number)} />
              </div>
            </div>
          </div>

          <div className="mt-5">
            <span className="eyebrow">Solve for</span>
            <div className="mt-1.5">
              <Segmented ariaLabel="Solve mode" options={MODE} value={mode} onChange={(v) => setMode(v as "price" | "yield")} />
            </div>
          </div>

          <div className="mt-4">
            {mode === "price" ? (
              <Field label="Yield" unit="%" value={yieldPct} step={0.25} min={0} max={15} onChange={setYieldPct} slider />
            ) : (
              <>
                <Field label="Market price" unit="$" value={marketPrice} step={5} min={1} onChange={setMarketPrice} />
                <p className="mt-2 text-sm text-[var(--muted)]">
                  Implied yield <span className="tabnum font-semibold text-[var(--accent)]">{(m.y * 100).toFixed(3)}%</span>
                </p>
              </>
            )}
          </div>
        </Card>

        {/* Valuation */}
        <Card className="rise">
          <h2 className="mb-4 text-lg">Valuation &amp; risk</h2>
          <Metric label={mode === "price" ? "Present value" : "Price (input)"} value={money(mode === "price" ? m.pv : safe(marketPrice))} tone="accent" size="lg" />
          <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
            <Metric label="Modified duration" value={`${m.dur.toFixed(2)} yrs`} />
            <Metric label="Convexity" value={m.conv.toFixed(2)} />
            <Metric label="DV01" value={`$${m.dollar01.toFixed(3)}`} sub="per 1 bp" />
            <Metric label="Current yield" value={`${(m.curY * 100).toFixed(2)}%`} />
            <Metric label="Annual coupon" value={`$${money0(m.annualCoupon)}`} />
            <Metric label="Yield" value={`${(m.y * 100).toFixed(2)}%`} tone={mode === "yield" ? "accent" : "neutral"} />
          </div>
          <p className="mt-4 rounded-[var(--radius)] border border-[var(--panel-border)] bg-[var(--accent-dim)] px-4 py-3 text-sm text-[var(--text)]">
            A <span className="tabnum">1%</span> rise in yield ≈ a{" "}
            <span className="tabnum font-semibold text-[var(--accent)]">{m.dropFor1pct.toFixed(2)}%</span> drop in price
            <span className="text-[var(--muted)]"> (duration + convexity estimate).</span>
          </p>
        </Card>
      </div>

      {/* Price–yield + cashflows */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="rise">
          <h2 className="text-lg">Price vs. yield</h2>
          <p className="mt-1 text-sm text-[var(--muted)]">The convex curve vs. the straight-line duration estimate.</p>
          <div className="mt-4">
            <PriceYieldChart curve={m.pyCurve} currentYield={m.y} currentPrice={m.pv} dv01={m.dollar01} />
          </div>
        </Card>
        <Card className="rise">
          <h2 className="text-lg">Where the value sits</h2>
          <p className="mt-1 text-sm text-[var(--muted)]">Present value of each cashflow; the dashed line is duration.</p>
          <div className="mt-4">
            <CashflowChart data={m.dcf} duration={m.dur} />
          </div>
        </Card>
      </div>

      {/* Scenario analysis */}
      <Card className="rise">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-lg">Scenario analysis</h2>
            <p className="mt-1 text-sm text-[var(--muted)]">Parallel yield shifts — full reprice vs. approximation.</p>
          </div>
          <Segmented ariaLabel="Shift magnitudes in basis points" multi options={MAGNITUDES.map((mm) => ({ label: `±${mm}`, value: mm }))} value={mags} onChange={(v) => setMags(v as number[])} />
        </div>
        <div className="mt-6"><ScenarioChart rows={m.rows} basePrice={m.pv} /></div>

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
              {m.rows.map((r) => {
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
          {m.rows.map((r) => {
            const tone = r.shiftBps === 0 ? "text-[var(--muted)]" : r.dollarChange >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]";
            return (
              <li key={r.shiftBps} className="flex items-center justify-between rounded-[var(--radius)] border border-[var(--panel-border)] bg-[var(--panel-2)] px-4 py-3">
                <div>
                  <div className="tabnum font-semibold">{r.shiftBps > 0 ? `+${r.shiftBps}` : r.shiftBps} bps</div>
                  <div className="tabnum text-xs text-[var(--muted)]">y {(r.newYield * 100).toFixed(3)}%</div>
                </div>
                <div className="text-right">
                  <div className="tabnum">{money(r.newPrice)}</div>
                  <div className={`tabnum text-xs ${tone}`}>{r.shiftBps === 0 ? "—" : signedPct(r.pctChange)}</div>
                </div>
              </li>
            );
          })}
        </ul>
      </Card>

      {/* Horizon total return */}
      <Card className="rise">
        <h2 className="text-lg">Horizon total return</h2>
        <p className="mt-1 text-sm text-[var(--muted)]">
          Hold for a period, collect coupons, then value the remaining bond under a yield shift.
        </p>

        <div className="mt-5 grid gap-5 sm:grid-cols-2">
          <Field label={`Holding period (max ${m.yrs}y)`} unit="yr" value={horizon} step={0.5} min={0.5} max={m.yrs} onChange={setHorizon} slider />
          <Field label="Yield shift at horizon" unit="bps" value={horizonShift} step={25} min={-300} max={300} onChange={setHorizonShift} slider />
        </div>

        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Metric label="Coupon income" value={`$${money(m.hz.couponIncome)}`} />
          <Metric label="Bond value @ horizon" value={`$${money(m.hz.bondValue)}`} />
          <Metric label="Total return" value={signedPct(m.hz.totalReturn)} tone={m.hz.totalReturn >= 0 ? "pos" : "neg"} />
          <Metric label="Annualized" value={signedPct(m.hz.annualizedReturn)} tone={m.hz.annualizedReturn >= 0 ? "pos" : "neg"} />
        </div>

        {/* composition bar: income vs remaining bond value */}
        <div className="mt-5">
          <div className="flex h-3 overflow-hidden rounded-full border border-[var(--panel-border)]">
            <div className="bg-[var(--accent)]" style={{ width: `${incomeShare}%` }} title="Coupon income" />
            <div className="bg-[var(--warn)]" style={{ width: `${100 - incomeShare}%` }} title="Bond value at horizon" />
          </div>
          <div className="mt-2 flex justify-between text-xs text-[var(--muted)]">
            <span className="inline-flex items-center gap-1.5"><span className="inline-block h-2 w-2 rounded-sm bg-[var(--accent)]" /> Coupons {incomeShare.toFixed(0)}%</span>
            <span className="tabnum">Total value @ horizon: ${money(m.hz.totalValue)}</span>
            <span className="inline-flex items-center gap-1.5"><span className="inline-block h-2 w-2 rounded-sm bg-[var(--warn)]" /> Bond {(100 - incomeShare).toFixed(0)}%</span>
          </div>
        </div>
      </Card>
    </div>
  );
}
