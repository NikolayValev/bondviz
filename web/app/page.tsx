import Link from "next/link";
import { headers } from "next/headers";
import { Card } from "@/components/ui/Card";
import { Kpi } from "@/components/ui/Kpi";
import { computeCurveKpis } from "@/lib/finance";
import { Kpis } from "@/lib/types";

export const dynamic = "force-dynamic";

async function getKpis(): Promise<{ kpis: Kpis | null; date: string | null }> {
  try {
    const h = await headers();
    const host = h.get("host");
    const proto = h.get("x-forwarded-proto") ?? "http";
    const res = await fetch(`${proto}://${host}/api/treasury/latest`, { cache: "no-store" });
    if (!res.ok) return { kpis: null, date: null };
    const { row } = await res.json();
    if (!row) return { kpis: null, date: null };
    return { kpis: computeCurveKpis(row), date: row.date };
  } catch {
    return { kpis: null, date: null };
  }
}

const pct = (v: number | null) => (v === null ? "—" : `${v.toFixed(2)}%`);
const bps = (v: number | null) => (v === null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(0)} bps`);

export default async function Home() {
  const { kpis, date } = await getKpis();
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-4xl font-semibold text-[var(--accent)]">BONDVIZ</h1>
        <p className="mt-1 text-lg text-[var(--muted)]">Fixed-income research terminal</p>
        <p className="mt-3 max-w-2xl text-[var(--text)]">
          Live U.S. Treasury data to visualize the yield curve and price bonds. A front-end-focused
          demo built with Next.js and hand-rolled D3 charts.
        </p>
      </section>

      <Card>
        <h2 className="mb-3 text-lg">Snapshot{date ? ` · ${date}` : ""}</h2>
        {kpis === null ? (
          <p className="text-[var(--muted)]">Live Treasury snapshot unavailable — open the tools from the nav.</p>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Kpi label="10Y Treasury" value={pct(kpis.tenYear)} />
            <Kpi label="2s10s spread" value={bps(kpis.twos10s)} />
            <Kpi label="3m10y spread" value={bps(kpis.threeM10Y)} />
          </div>
        )}
      </Card>

      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Link href="/yield-curve">
          <Card className="h-full transition-colors hover:border-l-[var(--text)]">
            <h3 className="text-[var(--accent)]">Yield Curve →</h3>
            <p className="mt-1 text-sm text-[var(--muted)]">Latest curve, shifts vs the past, and key spreads.</p>
          </Card>
        </Link>
        <Link href="/pricing">
          <Card className="h-full transition-colors hover:border-l-[var(--text)]">
            <h3 className="text-[var(--accent)]">Bond Pricing →</h3>
            <p className="mt-1 text-sm text-[var(--muted)]">Continuous-compounding present value calculator.</p>
          </Card>
        </Link>
      </section>
    </div>
  );
}
