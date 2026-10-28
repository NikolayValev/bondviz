import Link from "next/link";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { LineChart, type Series } from "@/components/charts/LineChart";
import { describeCurve, toBps } from "@/lib/finance";
import { fetchTreasuryYear } from "@/lib/treasury";
import { marketSnapshot, type MarketSnapshot } from "@/lib/snapshot";
import { SERIES } from "@/lib/chartColors";
import { TOOLS, TOOL_GROUPS } from "@/lib/tools";
import { SOURCE_URL } from "@/lib/site";
import { PrefetchSignal } from "./PrefetchSignal";

// ISR: serve the page statically and refresh the KPI snapshot hourly (the
// Treasury feed itself is cached hourly in fetchTreasuryYear). force-dynamic
// blocked TTFB on an upstream XML fetch for every visitor.
export const revalidate = 3600;

/** This year's and last year's rows (each cached hourly): enough for "a month ago" and "a year
 *  ago" in January too. Either year failing still leaves a snapshot from the other. */
async function getSnapshot(): Promise<MarketSnapshot | null> {
  const year = new Date().getUTCFullYear();
  const [prev, cur] = await Promise.allSettled([fetchTreasuryYear(year - 1), fetchTreasuryYear(year)]);
  const rows = [prev, cur].flatMap((r) => (r.status === "fulfilled" ? r.value : []));
  return marketSnapshot(rows);
}

const pct = (v: number | null) => (v === null ? "—" : `${v.toFixed(2)}%`);
const bps = (v: number | null) => (v === null ? "—" : `${v >= 0 ? "+" : ""}${toBps(v).toFixed(0)} bps`);
const slopeTone = (v: number | null) => (v === null ? "neutral" : v >= 0 ? "pos" : "neg");
/** A month's change in percentage points, as signed bps: "+21 bps over the month". */
const monthMove = (v: number | null | undefined) =>
  v === null || v === undefined ? undefined : `${v >= 0 ? "+" : "−"}${Math.abs(toBps(v)).toFixed(0)} bps over the month`;

function curveSeries(snap: MarketSnapshot): Series[] {
  const series: Series[] = [
    { id: "today", label: `Today (${snap.date})`, color: SERIES[0], points: snap.curve.map((p) => [p.years, p.yield]) },
  ];
  if (snap.yearAgo)
    series.push({
      id: "year-ago",
      label: `A year ago (${snap.yearAgo.date})`,
      color: SERIES[1],
      points: snap.yearAgo.curve.map((p) => [p.years, p.yield]),
    });
  return series;
}

/** "The 10Y is 118 bps higher than a year ago." */
function yearOnYear(snap: MarketSnapshot): string {
  const now = snap.curve.find((p) => p.label === "10Y");
  const then = snap.yearAgo?.curve.find((p) => p.label === "10Y");
  if (!now || !then) return "";
  const d = Math.round(toBps(now.yield - then.yield));
  return d === 0 ? " The 10Y is unchanged on the year." : ` The 10Y is ${Math.abs(d)} bps ${d > 0 ? "higher" : "lower"} than a year ago.`;
}

export default async function Home() {
  const snap = await getSnapshot().catch(() => null);
  const kpis = snap?.kpis ?? null;
  return (
    <div className="space-y-10">
      <PrefetchSignal />
      <section className="rise">
        <div className="eyebrow">Fixed-income research terminal</div>
        <h1 className="mt-2 text-5xl font-bold tracking-tight sm:text-6xl">
          BOND<span className="text-[var(--accent)]">VIZ</span>
        </h1>
        <p className="mt-4 max-w-2xl text-lg text-[var(--muted)]">
          Live U.S. Treasury data, end to end: explore and decompose the yield curve, price and
          stress-test bonds and portfolios, track the inversion recession signal, or ask for a chart
          in plain English. Hand-rolled D3 charts on unit-tested bond math.
        </p>
        <div className="mt-5 flex flex-wrap gap-3 text-sm">
          <Link
            href="/yield-curve"
            className="rounded-full bg-[var(--accent)] px-4 py-2 font-medium text-[var(--on-accent)] transition-opacity hover:opacity-90"
          >
            Explore the curve
          </Link>
          <Link
            href="/about"
            className="rounded-full border border-[var(--panel-border)] px-4 py-2 text-[var(--muted)] transition-colors hover:bg-white/5 hover:text-[var(--text)]"
          >
            How the math works
          </Link>
          <a
            href={SOURCE_URL}
            target="_blank"
            rel="noreferrer"
            className="rounded-full border border-[var(--panel-border)] px-4 py-2 text-[var(--muted)] transition-colors hover:bg-white/5 hover:text-[var(--text)]"
          >
            View source
          </a>
        </div>
      </section>

      <section className="rise" style={{ animationDelay: "60ms" }}>
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="text-lg">Market snapshot</h2>
          {snap && <span className="tabnum text-xs text-[var(--faint)]">as of {snap.date}</span>}
        </div>
        {snap === null || kpis === null ? (
          <Card>
            <p className="text-[var(--muted)]">Live Treasury snapshot unavailable — open the tools from the nav.</p>
          </Card>
        ) : (
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 lg:grid-cols-1">
              <Metric label="10Y Treasury" value={pct(kpis.tenYear)} tone="accent" sub={monthMove(snap.monthChange?.tenYear)} />
              <Metric
                label="2s10s spread"
                value={bps(kpis.twos10s)}
                tone={slopeTone(kpis.twos10s)}
                sub={monthMove(snap.monthChange?.twos10s)}
              />
              <Metric
                label="3m10y spread"
                value={bps(kpis.threeM10Y)}
                tone={slopeTone(kpis.threeM10Y)}
                sub={monthMove(snap.monthChange?.threeM10Y)}
              />
            </div>
            <Card>
              <h3 className="text-base">{snap.yearAgo ? "The curve today vs a year ago" : "The curve today"}</h3>
              <div className="mt-3">
                <LineChart
                  ariaLabel={`Treasury par yield curve on ${snap.date}${snap.yearAgo ? ` and on ${snap.yearAgo.date}` : ""}`}
                  series={curveSeries(snap)}
                  xLabel="Maturity (years)"
                  yLabel="Yield (%)"
                  yUnit="%"
                  height={232}
                />
              </div>
              <p className="mt-2 text-sm text-[var(--muted)]">
                {describeCurve(snap.curve)}
                {yearOnYear(snap)}
              </p>
            </Card>
          </div>
        )}
      </section>

      {TOOL_GROUPS.map((group, gi) => (
        <section key={group} className="rise" style={{ animationDelay: `${120 + gi * 60}ms` }}>
          <h2 className="eyebrow mb-3">{group}</h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {TOOLS.filter((t) => t.group === group).map((t) => (
              <Link key={t.href} href={t.href} className="group">
                <Card className="h-full transition-all group-hover:border-l-[var(--accent)] group-hover:bg-[var(--panel)]">
                  <h3 className="flex items-center justify-between text-[var(--accent)]">
                    {t.title} <span className="transition-transform group-hover:translate-x-1">→</span>
                  </h3>
                  <p className="mt-2 text-sm text-[var(--muted)]">{t.description}</p>
                </Card>
              </Link>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
