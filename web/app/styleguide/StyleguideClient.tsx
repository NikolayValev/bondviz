"use client";
// Design-system catalog: every token, UI primitive, and chart in one place.
// Purely presentational and driven entirely by the CSS-var tokens + chartColors
// palette, so editing app/globals.css restyles this page live — the surface for
// iterating a new design system. Unlinked from the product nav; visit /styleguide.
import { useState } from "react";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { Kpi } from "@/components/ui/Kpi";
import { Segmented } from "@/components/ui/Segmented";
import { LineChart, Series } from "@/components/charts/LineChart";
import { CategoryBarChart } from "@/components/charts/CategoryBarChart";
import { Heatmap } from "@/components/charts/Heatmap";
import { ScenarioChart } from "@/components/charts/ScenarioChart";
import { PriceYieldChart } from "@/components/charts/PriceYieldChart";
import { CashflowChart } from "@/components/charts/CashflowChart";
import { SpreadHistoryChart } from "@/components/charts/SpreadHistoryChart";
import { SERIES } from "@/lib/chartColors";
import {
  bondCashflows,
  discountedCashflows,
  dv01,
  modifiedDuration,
  priceFromCashflows,
  priceYieldCurve,
  scenarioShift,
  symmetricShifts,
} from "@/lib/finance";
import { NBER_RECESSIONS, type SpreadPoint } from "@/lib/signal";

const SURFACE_TOKENS: [string, string][] = [
  ["bg", "--bg"], ["bg-2", "--bg-2"], ["panel", "--panel"], ["panel-2", "--panel-2"],
];
const TEXT_TOKENS: [string, string][] = [
  ["text", "--text"], ["muted", "--muted"], ["faint", "--faint"],
];
const SEMANTIC_TOKENS: [string, string][] = [
  ["accent", "--accent"], ["pos", "--pos"], ["neg", "--neg"], ["warn", "--warn"],
];
const SERIES_TOKENS: [string, string][] = [
  ["series-1", "--series-1"], ["series-2", "--series-2"], ["series-3", "--series-3"],
  ["series-4", "--series-4"], ["series-5", "--series-5"], ["series-6", "--series-6"],
];

function Swatch({ name, varName }: { name: string; varName: string }) {
  return (
    <div className="flex items-center gap-2.5">
      <span
        className="h-9 w-9 shrink-0 rounded-md border border-[var(--panel-border)]"
        style={{ background: `var(${varName})` }}
      />
      <div className="leading-tight">
        <div className="text-sm text-[var(--text)]">{name}</div>
        <div className="tabnum text-xs text-[var(--faint)]">{varName}</div>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-4">
      <h2 className="border-b border-[var(--panel-border)] pb-2 text-lg">{title}</h2>
      {children}
    </section>
  );
}

// --- Sample data (static; exercises each chart with realistic props) ---------
const FACE = 1000;
const COUPON = 0.05;
const YLD = 0.04;
const YEARS = 10;
const FREQ = 2;
const { cashflows, times } = bondCashflows(FACE, COUPON, YEARS, FREQ);
const SAMPLE_PV = priceFromCashflows(cashflows, times, YLD);
const SAMPLE_DUR = modifiedDuration(cashflows, times, YLD);
const SAMPLE_DV01 = dv01(cashflows, times, YLD);
const SAMPLE_DCF = discountedCashflows(cashflows, times, YLD);
const SAMPLE_PY = priceYieldCurve(FACE, COUPON, YEARS, FREQ, Array.from({ length: 41 }, (_, i) => 0.005 + (0.09 * i) / 40));
const SAMPLE_SCENARIO = scenarioShift({ face: FACE, coupon: COUPON, yield_: YLD, years: YEARS, freq: FREQ }, symmetricShifts([25, 50, 100]));

const LINE_SERIES: Series[] = [
  { id: "a", label: "Series A", color: SERIES[0], points: [[0, 4.1], [2, 4.3], [5, 4.0], [10, 4.4], [30, 4.6]] },
  { id: "b", label: "Series B", color: SERIES[1], points: [[0, 3.8], [2, 3.9], [5, 4.1], [10, 4.2], [30, 4.3]] },
];
const BAR_CATEGORIES = ["2Y", "5Y", "10Y", "30Y"];
const HEATMAP = {
  dates: ["2024-01", "2024-02", "2024-03", "2024-04"],
  tenors: ["1Y", "2Y", "5Y", "10Y"],
  values: [
    [4.9, 4.6, 4.2, 4.1],
    [4.8, 4.5, 4.1, 4.0],
    [4.7, 4.5, 4.3, 4.2],
    [4.6, 4.4, 4.4, 4.5],
  ],
};
const SPREAD_POINTS: SpreadPoint[] = [
  { date: "2005-01-01", s10y3m: 1.2, s2s10s: 0.5 },
  { date: "2006-06-01", s10y3m: 0.1, s2s10s: 0.0 },
  { date: "2007-06-01", s10y3m: -0.4, s2s10s: -0.2 },
  { date: "2008-09-01", s10y3m: 1.5, s2s10s: 1.0 },
  { date: "2012-01-01", s10y3m: 2.8, s2s10s: 1.6 },
  { date: "2019-08-01", s10y3m: -0.2, s2s10s: 0.1 },
  { date: "2020-03-01", s10y3m: 0.6, s2s10s: 0.4 },
  { date: "2023-01-01", s10y3m: -1.2, s2s10s: -0.7 },
  { date: "2024-06-01", s10y3m: -0.9, s2s10s: -0.4 },
];

export function StyleguideClient() {
  const [seg, setSeg] = useState<string>("semi");
  const [multi, setMulti] = useState<number[]>([25, 50]);

  return (
    <div className="space-y-10">
      <header className="rise">
        <div className="eyebrow">Design system</div>
        <h1 className="mt-1.5 text-3xl sm:text-4xl">Styleguide</h1>
        <p className="mt-2 max-w-2xl text-[var(--muted)]">
          Every design token, UI primitive, and chart in one place. Everything is driven by the CSS
          variables in <code className="tabnum">app/globals.css</code> and the{" "}
          <code className="tabnum">lib/chartColors.ts</code> palette — edit a token and this page
          restyles live.
        </p>
      </header>

      <Section title="Color tokens">
        <Card>
          <div className="space-y-5">
            <div>
              <div className="eyebrow mb-3">Surfaces</div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {SURFACE_TOKENS.map(([n, v]) => <Swatch key={v} name={n} varName={v} />)}
              </div>
            </div>
            <div>
              <div className="eyebrow mb-3">Text</div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {TEXT_TOKENS.map(([n, v]) => <Swatch key={v} name={n} varName={v} />)}
              </div>
            </div>
            <div>
              <div className="eyebrow mb-3">Semantic</div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {SEMANTIC_TOKENS.map(([n, v]) => <Swatch key={v} name={n} varName={v} />)}
              </div>
            </div>
            <div>
              <div className="eyebrow mb-3">Chart series</div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-6">
                {SERIES_TOKENS.map(([n, v]) => <Swatch key={v} name={n} varName={v} />)}
              </div>
            </div>
          </div>
        </Card>
      </Section>

      <Section title="Primitives">
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <div className="eyebrow mb-3">Metric · tones & sizes</div>
            <div className="grid grid-cols-2 gap-3">
              <Metric label="Accent" value="4.40%" tone="accent" />
              <Metric label="Positive" value="+18 bps" tone="pos" />
              <Metric label="Negative" value="−42 bps" tone="neg" />
              <Metric label="Neutral" value="100.00" tone="neutral" />
              <Metric label="Large" value="$1,012.50" tone="accent" size="lg" sub="with sub-label" />
              <Kpi label="Kpi (legacy)" value="4.40%" />
            </div>
          </Card>
          <Card>
            <div className="eyebrow mb-3">Segmented · single & multi</div>
            <div className="space-y-4">
              <Segmented
                ariaLabel="Single select demo"
                options={[{ label: "Annual", value: "annual" }, { label: "Semi", value: "semi" }, { label: "Quarterly", value: "quarterly" }]}
                value={seg}
                onChange={(v) => setSeg(v as string)}
              />
              <Segmented
                ariaLabel="Multi select demo"
                multi
                options={[25, 50, 100, 200].map((m) => ({ label: `±${m}`, value: m }))}
                value={multi}
                onChange={(v) => setMulti(v as number[])}
              />
              <p className="text-sm text-[var(--muted)]">
                Selected: <span className="tabnum text-[var(--text)]">{seg}</span> ·{" "}
                <span className="tabnum text-[var(--text)]">[{multi.join(", ")}]</span>
              </p>
            </div>
          </Card>
          <Card className="lg:col-span-2">
            <div className="eyebrow mb-3">Card + typography</div>
            <h3 className="text-[var(--accent)]">Card heading</h3>
            <p className="mt-2 text-sm text-[var(--muted)]">
              Body copy in muted. <span className="tabnum text-[var(--text)]">1,234.56</span> renders
              tabular. The <code>.eyebrow</code> label, <code>h1–h3</code>, and the accent link color
              all come from tokens.
            </p>
          </Card>
        </div>
      </Section>

      <Section title="Charts">
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <h3 className="mb-2 text-base">LineChart</h3>
            <LineChart ariaLabel="Sample line chart" series={LINE_SERIES} xLabel="Maturity (years)" yLabel="Yield" yUnit="%" />
          </Card>
          <Card>
            <h3 className="mb-2 text-base">CategoryBarChart (diverging stack)</h3>
            <CategoryBarChart
              ariaLabel="Sample bar chart"
              yUnit=" bps"
              categories={BAR_CATEGORIES}
              series={[
                { id: "carry", label: "Carry", color: SERIES[0], values: [12, 28, 41, 55] },
                { id: "roll", label: "Roll-down", color: SERIES[1], values: [8, 14, -6, -18] },
              ]}
            />
          </Card>
          <Card>
            <h3 className="mb-2 text-base">Heatmap</h3>
            <Heatmap ariaLabel="Sample heatmap" dates={HEATMAP.dates} tenors={HEATMAP.tenors} values={HEATMAP.values} />
          </Card>
          <Card>
            <h3 className="mb-2 text-base">ScenarioChart</h3>
            <ScenarioChart rows={SAMPLE_SCENARIO} basePrice={SAMPLE_PV} />
          </Card>
          <Card>
            <h3 className="mb-2 text-base">PriceYieldChart</h3>
            <PriceYieldChart curve={SAMPLE_PY} currentYield={YLD} currentPrice={SAMPLE_PV} dv01={SAMPLE_DV01} />
          </Card>
          <Card>
            <h3 className="mb-2 text-base">CashflowChart</h3>
            <CashflowChart data={SAMPLE_DCF} duration={SAMPLE_DUR} />
          </Card>
          <Card className="lg:col-span-2">
            <h3 className="mb-2 text-base">SpreadHistoryChart</h3>
            <SpreadHistoryChart ariaLabel="Sample spread history" points={SPREAD_POINTS} recessions={NBER_RECESSIONS} />
          </Card>
        </div>
      </Section>
    </div>
  );
}
