"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { SpreadHistoryChart } from "@/components/charts/SpreadHistoryChart";
import { currentStatus, inversionEpisodes, NBER_RECESSIONS, SpreadPoint } from "@/lib/signal";
import { iso } from "@/lib/format";

const bps = (pp: number | null) => (pp === null ? "—" : `${pp >= 0 ? "+" : ""}${(pp * 100).toFixed(0)} bps`);

export function SignalClient() {
  const [points, setPoints] = useState<SpreadPoint[] | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch(`/api/treasury/spreads?start=1990-01-01&end=${iso(new Date())}`)
      .then((r) => r.json())
      .then((d) => setPoints(d.points ?? []))
      .catch(() => setError(true));
  }, []);

  const view = useMemo(() => {
    if (!points) return null;
    const status = currentStatus(points);
    const forward = inversionEpisodes(points); // chronological
    const episodes = forward.slice().reverse(); // most recent first
    const withRec = forward.filter((e) => e.recessionFollowed).length;
    return { status, episodes, total: forward.length, withRec };
  }, [points]);

  if (error) return <p className="text-[var(--muted)]">Treasury data is unavailable right now.</p>;
  if (!view) return <p className="text-[var(--muted)]">Loading 35 years of Treasury data… (first load can take a moment)</p>;

  const { status } = view;
  const summary =
    status.streakDays > 0
      ? `10y–3m has inverted before ${view.withRec} of the last ${NBER_RECESSIONS.length} recessions; currently inverted ${status.streakDays} trading days.`
      : `10y–3m is currently positive (${bps(status.s10y3m)}). ${view.total} inversion episodes since 1990, ${view.withRec} followed by recession within 24 months.`;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl">Inversion &amp; Recession Signal</h1>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card>
          <div className="text-xs uppercase tracking-wide text-[var(--muted)]">10y–3m spread</div>
          <div className="mt-1 text-3xl font-bold tabnum" style={{ color: status.inverted ? "var(--neg)" : "var(--pos)" }}>
            {bps(status.s10y3m)}
          </div>
          <div className="mt-1 text-sm text-[var(--muted)]">{status.date ?? "—"}</div>
        </Card>
        <Card>
          <div className="text-xs uppercase tracking-wide text-[var(--muted)]">Status</div>
          <div className="mt-1 text-3xl font-bold" style={{ color: status.inverted ? "var(--neg)" : "var(--pos)" }}>
            {status.inverted ? "Inverted" : "Normal"}
          </div>
          <div className="mt-1 text-sm text-[var(--muted)] tabnum">
            {status.inverted ? `${status.streakDays} trading days` : "not inverted"}
          </div>
        </Card>
        <Card>
          <div className="text-xs uppercase tracking-wide text-[var(--muted)]">2s10s spread</div>
          <div className="mt-1 text-3xl font-bold tabnum text-[var(--text)]">{bps(status.s2s10s)}</div>
          <div className="mt-1 text-sm text-[var(--muted)]">secondary signal</div>
        </Card>
      </div>

      <Card>
        <h2 className="mb-2 text-lg">Spread history since 1990</h2>
        <SpreadHistoryChart
          ariaLabel="10y minus 3m and 2s10s spreads since 1990 with inversion and recession shading"
          points={points!}
          recessions={NBER_RECESSIONS}
        />
        <p className="mt-2 text-sm text-[var(--muted)]">{summary}</p>
      </Card>

      <Card>
        <h2 className="mb-3 text-lg">Inversion episodes (10y–3m)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm tabnum">
            <thead>
              <tr className="text-left text-[var(--muted)]">
                <th className="py-1 pr-4 font-medium">Start</th>
                <th className="py-1 pr-4 font-medium">End</th>
                <th className="py-1 pr-4 font-medium">Days</th>
                <th className="py-1 pr-4 font-medium">Max depth</th>
                <th className="py-1 pr-4 font-medium">Recession ≤24m</th>
              </tr>
            </thead>
            <tbody>
              {view.episodes.map((e) => (
                <tr key={e.start} className="border-t border-[var(--panel-border)]">
                  <td className="py-1.5 pr-4 text-[var(--text)]">{e.start}</td>
                  <td className="py-1.5 pr-4">{e.end}</td>
                  <td className="py-1.5 pr-4">{e.days}</td>
                  <td className="py-1.5 pr-4" style={{ color: "var(--neg)" }}>{e.maxDepthBps.toFixed(0)} bps</td>
                  <td className="py-1.5 pr-4" style={{ color: e.recessionFollowed ? "var(--neg)" : "var(--muted)" }}>
                    {e.recessionFollowed ? "✓" : "✗"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-[var(--faint)]">
          An episode is a maximal run of days with 10y–3m ≤ 0. &ldquo;Recession ≤24m&rdquo; marks episodes whose start
          preceded an NBER recession by no more than 24 months.
        </p>
      </Card>
    </div>
  );
}
