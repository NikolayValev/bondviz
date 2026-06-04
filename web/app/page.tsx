import Link from "next/link";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { computeCurveKpis, toBps } from "@/lib/finance";
import { fetchTreasuryYear } from "@/lib/treasury";
import { Kpis } from "@/lib/types";

export const dynamic = "force-dynamic";

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
      <section className="rise">
        <div className="eyebrow">Fixed-income research terminal</div>
        <h1 className="mt-2 text-5xl font-bold tracking-tight sm:text-6xl">
          BOND<span className="text-[var(--accent)]">VIZ</span>
        </h1>
        <p className="mt-4 max-w-2xl text-lg text-[var(--muted)]">
          Live U.S. Treasury data to visualize the yield curve and price bonds — present value,
          duration, convexity and parallel-shift scenarios, with hand-rolled D3 charts.
        </p>
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

      <section className="grid grid-cols-1 gap-4 rise sm:grid-cols-2" style={{ animationDelay: "120ms" }}>
        <Link href="/yield-curve" className="group">
          <Card className="h-full transition-all group-hover:border-l-[var(--accent)] group-hover:bg-[var(--panel)]">
            <h3 className="flex items-center justify-between text-[var(--accent)]">
              Yield Curve <span className="transition-transform group-hover:translate-x-1">→</span>
            </h3>
            <p className="mt-2 text-sm text-[var(--muted)]">Latest curve, shifts versus the past, and key spreads over time.</p>
          </Card>
        </Link>
        <Link href="/pricing" className="group">
          <Card className="h-full transition-all group-hover:border-l-[var(--accent)] group-hover:bg-[var(--panel)]">
            <h3 className="flex items-center justify-between text-[var(--accent)]">
              Bond Pricing &amp; Risk <span className="transition-transform group-hover:translate-x-1">→</span>
            </h3>
            <p className="mt-2 text-sm text-[var(--muted)]">PV, duration, convexity and a parallel-shift scenario analyzer.</p>
          </Card>
        </Link>
      </section>
    </div>
  );
}
