"use client";
import { useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { Segmented } from "@/components/ui/Segmented";
import { ScenarioChart } from "@/components/charts/ScenarioChart";
import { symmetricShifts, type BondParams } from "@/lib/finance";
import { portfolioMetrics, portfolioScenario } from "@/lib/portfolio";

interface Holding {
  id: string;
  label: string;
  face: number;
  couponPct: number;
  yieldPct: number;
  years: number;
  freq: number;
}

let nextId = 0;
const mk = (label: string, face: number, couponPct: number, yieldPct: number, years: number): Holding => ({
  id: `h${nextId++}`, label, face, couponPct, yieldPct, years, freq: 2,
});

const SEED: Holding[] = [
  mk("2Y Note", 50_000, 4.5, 4.6, 2),
  mk("5Y Note", 50_000, 4.0, 4.2, 5),
  mk("10Y Bond", 50_000, 5.0, 4.4, 10),
];

const money = (v: number) => v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const money0 = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 });
const signedMoney = (v: number) => `${v >= 0 ? "+" : "−"}$${money(Math.abs(v))}`;
const signedPct = (v: number) => `${v >= 0 ? "+" : "−"}${(Math.abs(v) * 100).toFixed(2)}%`;
const MAGNITUDES = [25, 50, 100, 200];
const safe = (n: number) => (Number.isFinite(n) ? n : 0);

function Num({ label, unit, value, onChange, step, min }: {
  label: string; unit?: string; value: number; onChange: (v: number) => void; step: number; min?: number;
}) {
  return (
    <label className="block">
      <span className="eyebrow text-[0.6rem]">{label}</span>
      <div className="mt-1 flex items-center rounded-lg border border-[var(--panel-border)] bg-[var(--bg)] focus-within:border-[var(--accent)]">
        <input
          type="number" value={Number.isFinite(value) ? value : ""} step={step} min={min}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="tabnum w-full bg-transparent px-2.5 py-2 text-sm text-[var(--text)] outline-none"
        />
        {unit && <span className="pr-2 text-xs text-[var(--faint)]">{unit}</span>}
      </div>
    </label>
  );
}

export function PortfolioClient() {
  const [holdings, setHoldings] = useState<Holding[]>(SEED);
  const [mags, setMags] = useState<number[]>([25, 50, 100]);

  const update = (id: string, patch: Partial<Holding>) =>
    setHoldings((hs) => hs.map((h) => (h.id === id ? { ...h, ...patch } : h)));
  const remove = (id: string) => setHoldings((hs) => hs.filter((h) => h.id !== id));
  const add = () => setHoldings((hs) => [...hs, mk(`Bond ${hs.length + 1}`, 25_000, 4.5, 4.5, 7)]);

  const params: BondParams[] = useMemo(
    () => holdings.map((h) => ({
      face: safe(h.face), coupon: safe(h.couponPct) / 100, yield_: safe(h.yieldPct) / 100,
      years: Math.max(0.5, safe(h.years)), freq: h.freq,
    })),
    [holdings],
  );

  const metrics = useMemo(() => portfolioMetrics(params), [params]);
  const rows = useMemo(() => portfolioScenario(params, symmetricShifts(mags)), [params, mags]);

  return (
    <div className="space-y-8">
      <header className="rise">
        <div className="eyebrow">Fixed income · portfolio</div>
        <h1 className="mt-1.5 text-3xl sm:text-4xl">Portfolio Analyzer</h1>
        <p className="mt-2 max-w-2xl text-[var(--muted)]">
          Combine multiple bonds and see aggregate market value, value-weighted duration, convexity,
          yield and DV01 — then stress the whole book with a parallel rate shift.
        </p>
      </header>

      {/* Aggregate metrics */}
      <section className="rise grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Metric label="Market value" value={`$${money0(metrics.totalValue)}`} tone="accent" />
        <Metric label="Wtd duration" value={`${metrics.weightedDuration.toFixed(2)} yrs`} />
        <Metric label="Wtd convexity" value={metrics.weightedConvexity.toFixed(1)} />
        <Metric label="Portfolio DV01" value={`$${metrics.dv01.toFixed(2)}`} sub="per 1 bp" />
        <Metric label="Wtd yield" value={`${(metrics.weightedYield * 100).toFixed(2)}%`} />
        <Metric label="Avg maturity" value={`${metrics.weightedMaturity.toFixed(1)} yrs`} />
      </section>

      {/* Holdings */}
      <Card className="rise">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg">Holdings <span className="text-[var(--faint)]">({holdings.length})</span></h2>
          <button
            type="button" onClick={add}
            className="rounded-full border border-[var(--panel-border)] bg-[var(--panel-2)] px-3 py-1.5 text-sm text-[var(--accent)] transition-colors hover:bg-[var(--accent-dim)]"
          >
            + Add bond
          </button>
        </div>

        <div className="space-y-3">
          {holdings.map((h) => {
            const hm = metrics.holdings[holdings.indexOf(h)];
            return (
              <div key={h.id} className="rounded-[var(--radius)] border border-[var(--panel-border)] bg-[var(--panel-2)] p-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <input
                    value={h.label}
                    onChange={(e) => update(h.id, { label: e.target.value })}
                    className="w-40 rounded-md bg-transparent px-1 py-0.5 font-medium text-[var(--text)] outline-none focus:bg-[var(--bg)]"
                    aria-label="Holding label"
                  />
                  <button
                    type="button" onClick={() => remove(h.id)}
                    className="rounded-md px-2 py-1 text-sm text-[var(--faint)] transition-colors hover:bg-white/5 hover:text-[var(--neg)]"
                    aria-label={`Remove ${h.label}`}
                  >
                    ✕
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-5">
                  <Num label="Face" unit="$" value={h.face} step={1000} min={0} onChange={(v) => update(h.id, { face: v })} />
                  <Num label="Coupon" unit="%" value={h.couponPct} step={0.25} min={0} onChange={(v) => update(h.id, { couponPct: v })} />
                  <Num label="Yield" unit="%" value={h.yieldPct} step={0.25} min={0} onChange={(v) => update(h.id, { yieldPct: v })} />
                  <Num label="Maturity" unit="yr" value={h.years} step={1} min={0.5} onChange={(v) => update(h.id, { years: v })} />
                  <label className="block">
                    <span className="eyebrow text-[0.6rem]">Freq</span>
                    <div className="mt-1">
                      <Segmented
                        ariaLabel="Coupon frequency"
                        options={[{ label: "1", value: 1 }, { label: "2", value: 2 }, { label: "4", value: 4 }]}
                        value={h.freq}
                        onChange={(v) => update(h.id, { freq: v as number })}
                      />
                    </div>
                  </label>
                </div>
                {hm && (
                  <div className="mt-2.5 flex flex-wrap gap-x-5 gap-y-1 border-t border-[var(--panel-border)] pt-2.5 text-xs text-[var(--muted)]">
                    <span>MV <span className="tabnum text-[var(--text)]">${money0(hm.value)}</span></span>
                    <span>Weight <span className="tabnum text-[var(--text)]">{(hm.weight * 100).toFixed(1)}%</span></span>
                    <span>Duration <span className="tabnum text-[var(--text)]">{hm.duration.toFixed(2)}y</span></span>
                    <span>DV01 <span className="tabnum text-[var(--text)]">${hm.dv01.toFixed(2)}</span></span>
                  </div>
                )}
              </div>
            );
          })}
          {holdings.length === 0 && (
            <p className="rounded-[var(--radius)] border border-dashed border-[var(--panel-border)] px-4 py-8 text-center text-sm text-[var(--muted)]">
              No holdings — add a bond to start building a portfolio.
            </p>
          )}
        </div>
      </Card>

      {/* Portfolio scenario */}
      {holdings.length > 0 && (
        <Card className="rise">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-lg">Portfolio stress test</h2>
              <p className="mt-1 text-sm text-[var(--muted)]">Parallel shift applied to every holding — total value vs. approximation.</p>
            </div>
            <Segmented
              ariaLabel="Shift magnitudes in basis points" multi
              options={MAGNITUDES.map((mm) => ({ label: `±${mm}`, value: mm }))}
              value={mags} onChange={(v) => setMags(v as number[])}
            />
          </div>
          <div className="mt-6"><ScenarioChart rows={rows} basePrice={metrics.totalValue} /></div>

          <div className="mt-6 overflow-hidden rounded-[var(--radius)] border border-[var(--panel-border)]">
            <table className="w-full text-sm">
              <thead className="bg-[var(--panel-2)] text-left text-[var(--muted)]">
                <tr>
                  <th className="px-4 py-2.5 font-medium">Shift</th>
                  <th className="px-4 py-2.5 text-right font-medium">Portfolio value</th>
                  <th className="px-4 py-2.5 text-right font-medium">$ change</th>
                  <th className="px-4 py-2.5 text-right font-medium">% change</th>
                </tr>
              </thead>
              <tbody className="tabnum">
                {rows.map((r) => {
                  const tone = r.shiftBps === 0 ? "text-[var(--muted)]" : r.dollarChange >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]";
                  return (
                    <tr key={r.shiftBps} className={`border-t border-[var(--panel-border)] ${r.shiftBps === 0 ? "bg-white/[0.02]" : ""}`}>
                      <td className="px-4 py-2.5">{r.shiftBps > 0 ? `+${r.shiftBps}` : r.shiftBps} bps</td>
                      <td className="px-4 py-2.5 text-right">${money(r.newPrice)}</td>
                      <td className={`px-4 py-2.5 text-right ${tone}`}>{r.shiftBps === 0 ? "—" : signedMoney(r.dollarChange)}</td>
                      <td className={`px-4 py-2.5 text-right ${tone}`}>{r.shiftBps === 0 ? "—" : signedPct(r.pctChange)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
