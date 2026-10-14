import Link from "next/link";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { computeCurveKpis, toBps } from "@/lib/finance";
import { fetchTreasuryYear } from "@/lib/treasury";
import { Kpis } from "@/lib/types";
import { TOOLS, TOOL_GROUPS } from "@/lib/tools";
import { SOURCE_URL } from "@/lib/site";
import { PrefetchSignal } from "./PrefetchSignal";

// ISR: serve the page statically and refresh the KPI snapshot hourly (the
// Treasury feed itself is cached hourly in fetchTreasuryYear). force-dynamic
// blocked TTFB on an upstream XML fetch for every visitor.
export const revalidate = 3600;

async function getKpis(): Promise<{ kpis: Kpis | null; date: string | null }> {
  try {
    const year = new Date().getUTCFullYear();
    let rows = await fetchTreasuryYear(year);
    if (rows.length === 0) rows = await fetchTreasuryYear(year - 1);
    const row = rows.at(-1);
    if (!row) return { kpis: null, date: null };
    return { kpis: computeCurveKpis(row), date: row.date };
  } catch {
    return { kpis: null, date: null };
  }
}

const pct = (v: number | null) => (v === null ? "—" : `${v.toFixed(2)}%`);
const bps = (v: number | null) => (v === null ? "—" : `${v >= 0 ? "+" : ""}${toBps(v).toFixed(0)} bps`);
const slopeTone = (v: number | null) => (v === null ? "neutral" : v >= 0 ? "pos" : "neg");

export default async function Home() {
  const { kpis, date } = await getKpis();
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
          {date && <span className="tabnum text-xs text-[var(--faint)]">as of {date}</span>}
        </div>
        {kpis === null ? (
          <Card>
            <p className="text-[var(--muted)]">Live Treasury snapshot unavailable — open the tools from the nav.</p>
          </Card>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Metric label="10Y Treasury" value={pct(kpis.tenYear)} tone="accent" />
            <Metric label="2s10s spread" value={bps(kpis.twos10s)} tone={slopeTone(kpis.twos10s)} />
            <Metric label="3m10y spread" value={bps(kpis.threeM10Y)} tone={slopeTone(kpis.threeM10Y)} />
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
