"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { SpreadHistoryChart } from "@/components/charts/SpreadHistoryChart";
import {
  currentStatus,
  describeRecord,
  inversionEpisodes,
  inversionPeriods,
  NBER_RECESSIONS,
  signalRecord,
  SIGNAL_WINDOW_MONTHS,
  type PeriodOutcome,
  type SpreadPoint,
} from "@/lib/signal";
import { loadSpreads } from "@/lib/spreadsCache";
import { useUrlState } from "@/components/useUrlState";
import { oneOf, type Schema } from "@/lib/urlState";
import { RelativeValuePanel, RV_WINDOWS, type RvWindow } from "./RelativeValuePanel";
import { ScrollRegion } from "@/components/ui/ScrollRegion";
import { ExportButtons } from "@/components/ui/ExportButtons";
import { monthYear } from "@/lib/dates";

const URL_DEFAULTS = { window: "10Y" as RvWindow };
const URL_SCHEMA: Schema<typeof URL_DEFAULTS> = { window: oneOf(RV_WINDOWS.map((w) => w.label)) };

const bps = (pp: number | null) => (pp === null ? "—" : `${pp >= 0 ? "+" : ""}${(pp * 100).toFixed(0)} bps`);


// The 2020 recession was the pandemic shock: the 2019 inversion counts as a hit by the rule, but
// few read it as having forecast that.
const PANDEMIC_RECESSION = "2020-02-01";

function outcomeText(o: PeriodOutcome): string {
  if (o.kind === "recession")
    return `Recession from ${monthYear(o.recessionStart)}, ${o.leadMonths} months after the first inversion${o.recessionStart === PANDEMIC_RECESSION ? " †" : ""}`;
  if (o.kind === "pending") return `Too early to tell: window open until ${monthYear(o.windowEnds)}`;
  return `No recession within ${SIGNAL_WINDOW_MONTHS} months`;
}

const outcomeColor = (o: PeriodOutcome) =>
  o.kind === "recession" ? "var(--neg)" : o.kind === "pending" ? "var(--warn)" : "var(--muted)";

export function SignalClient() {
  const [points, setPoints] = useState<SpreadPoint[] | null>(null);
  const [error, setError] = useState(false);
  const [url, setUrl] = useUrlState(URL_SCHEMA, URL_DEFAULTS);

  useEffect(() => {
    loadSpreads()
      .then(setPoints)
      .catch(() => setError(true));
  }, []);

  const view = useMemo(() => {
    if (!points) return null;
    const status = currentStatus(points);
    const first = points[0]?.date ?? "";
    const forward = inversionPeriods(inversionEpisodes(points), status.date ?? first); // chronological
    return {
      status,
      periods: forward.slice().reverse(), // most recent first
      record: forward.length ? describeRecord(signalRecord(forward, first), first) : "",
    };
  }, [points]);

  // The heading renders in every state (prerendered; LCP; heading on error).
  const header = <h1 className="text-2xl">Inversion &amp; Recession Signal</h1>;
  if (error)
    return (
      <div className="space-y-6">
        {header}
        <p className="text-[var(--muted)]">Treasury data is unavailable right now.</p>
      </div>
    );
  if (!view)
    return (
      <div className="space-y-6">
        {header}
        <div role="status" className="space-y-6">
          <p className="text-[var(--muted)]">Loading every trading day since 1990… (the first load can take a moment)</p>
          {/* Sized like the loaded page (status cards, history panel, chart) to avoid layout shift. */}
          <Skeleton className="h-32" />
          <Skeleton className="h-80" />
          <Skeleton className="h-[30rem]" />
        </div>
      </div>
    );

  const { status } = view;
  const summary = [
    status.inverted
      ? `10y–3m is inverted (${bps(status.s10y3m)}), ${status.streakDays} trading days in a row.`
      : `10y–3m is ${bps(status.s10y3m)} today.`,
    view.record,
  ].join(" ");

  return (
    <div className="space-y-6">
      {header}

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

      {points && <RelativeValuePanel points={points} span={url.window} onSpan={(w) => setUrl({ window: w })} />}

      <Card>
        <h2 className="mb-2 text-lg">Spread history since 1990</h2>
        <SpreadHistoryChart
          ariaLabel="10y minus 3m and 2s10s spreads since 1990 with inversion and recession shading"
          exportTitle="Spread history since 1990"
          points={points!}
          recessions={NBER_RECESSIONS}
        />
        <p className="mt-2 text-sm text-[var(--muted)]">{summary}</p>
      </Card>

      <Card>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg">Inversion periods (10y–3m)</h2>
          <ExportButtons
            title="Inversion periods 10y-3m"
            csv={() => ({
              columns: ["Start", "End", "Days inverted", "Separate runs", "Deepest (bps)", "Outcome", "Recession start", "Lead (months)"],
              rows: view.periods.map((p) => [
                p.start,
                p.end,
                p.invertedDays,
                p.runs,
                p.maxDepthBps,
                p.outcome.kind,
                p.outcome.kind === "recession" ? p.outcome.recessionStart : null,
                p.outcome.kind === "recession" ? p.outcome.leadMonths : null,
              ]),
            })}
          />
        </div>
        <ScrollRegion label="Inversion periods table">
          <table className="w-full text-sm tabnum">
            <thead>
              <tr className="text-left text-[var(--muted)]">
                <th className="py-1 pr-4 font-medium">Start</th>
                <th className="py-1 pr-4 font-medium">End</th>
                <th className="py-1 pr-4 text-right font-medium">Days inverted</th>
                <th className="py-1 pr-4 text-right font-medium">Runs</th>
                <th className="py-1 pr-4 text-right font-medium">Deepest</th>
                <th className="py-1 font-medium">What followed</th>
              </tr>
            </thead>
            <tbody>
              {view.periods.map((p) => (
                <tr key={p.start} className="border-t border-[var(--panel-border)]">
                  <td className="py-1.5 pr-4 text-[var(--text)]">{p.start}</td>
                  <td className="py-1.5 pr-4">{p.end}</td>
                  <td className="py-1.5 pr-4 text-right">{p.invertedDays}</td>
                  <td className="py-1.5 pr-4 text-right">{p.runs}</td>
                  <td className="py-1.5 pr-4 text-right" style={{ color: "var(--neg)" }}>
                    {p.maxDepthBps.toFixed(0)} bps
                  </td>
                  <td className="py-1.5" style={{ color: outcomeColor(p.outcome) }}>
                    {outcomeText(p.outcome)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollRegion>
        <div className="mt-3 space-y-1 text-xs text-[var(--faint)]">
          <p>
            A period is a stretch of days with 10y–3m ≤ 0; runs less than six months apart count as one
            period. A recession follows a period if it starts after the first inverted day and no later than{" "}
            {SIGNAL_WINDOW_MONTHS} months after the last; the lead is counted from the first inverted day. The 1990 recession is left out
            because the data starts six months before it.
          </p>
          {view.periods.some((p) => p.outcome.kind === "recession" && p.outcome.recessionStart === PANDEMIC_RECESSION) && (
            <p>† The 2020 recession was the pandemic shock. It counts as a hit by this rule, but few read the 2019 inversion as having forecast it.</p>
          )}
        </div>
      </Card>
    </div>
  );
}
