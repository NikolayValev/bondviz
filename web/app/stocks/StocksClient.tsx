"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { Segmented } from "@/components/ui/Segmented";
import { Skeleton } from "@/components/ui/Skeleton";
import { LineChart, Series } from "@/components/charts/LineChart";
import { StockBar } from "@/lib/types";
import { money, money0, signedPct } from "@/lib/format";
import { localIsoDate, shiftMonths } from "@/lib/dates";
import { SERIES } from "@/lib/chartColors";
import { useUrlState } from "@/components/useUrlState";
import { oneOf, text, type Schema } from "@/lib/urlState";
import { ScrollRegion } from "@/components/ui/ScrollRegion";
import { ExportButtons } from "@/components/ui/ExportButtons";
import { comparePerformance, rebaseTo100 } from "@/lib/relativePerformance";

interface CacheEntry {
  configured: boolean;
  bars: StockBar[];
  error?: string;
}

const LOOKBACKS = [
  { label: "1M", value: 1 },
  { label: "3M", value: 3 },
  { label: "6M", value: 6 },
  { label: "1Y", value: 12 },
];

const URL_DEFAULTS = { tickers: "AAPL, MSFT, GOOGL", lookback: "3M", active: "" };
const URL_SCHEMA: Schema<typeof URL_DEFAULTS> = {
  tickers: text(80),
  lookback: oneOf(LOOKBACKS.map((l) => l.label)),
  active: text(12),
};

/** One line per ticker on the comparison chart, so at most one per series color. */
const MAX_COMPARE = SERIES.length;

const cacheKeyOf = (ticker: string, from: string, to: string) => `${ticker}|${from}|${to}`;

function parseTickers(s: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of s.split(",")) {
    const t = raw.trim().toUpperCase();
    if (t && !seen.has(t)) {
      seen.add(t);
      out.push(t);
    }
  }
  return out;
}

export function StocksClient() {
  const [url, setUrl] = useUrlState(URL_SCHEMA, URL_DEFAULTS);
  const tickersInput = url.tickers;
  const setTickersInput = (v: string) => setUrl({ tickers: v });
  const lookback = LOOKBACKS.find((l) => l.label === url.lookback)?.value ?? 3;
  const setLookback = (v: number) => setUrl({ lookback: LOOKBACKS.find((l) => l.value === v)?.label ?? "3M" });
  const active = url.active || null;
  const setActive = (v: string) => setUrl({ active: v });
  const [cache, setCache] = useState<Record<string, CacheEntry>>({});
  // Bumped by "Retry" to re-run the fetch effect for tickers that failed (e.g. rate limited).
  const [attempt, setAttempt] = useState(0);
  const retryFailed = () => {
    setCache((prev) => Object.fromEntries(Object.entries(prev).filter(([, e]) => !e.error)));
    setAttempt((n) => n + 1);
  };

  const tickers = useMemo(() => parseTickers(tickersInput), [tickersInput]);
  const activeTicker = active && tickers.includes(active) ? active : tickers[0] ?? null;

  // Recomputed only when lookback changes, so the fetch effect has stable deps.
  const { from, to } = useMemo(() => {
    const today = localIsoDate();
    return { from: shiftMonths(today, -lookback), to: today };
  }, [lookback]);

  // Every compared ticker is fetched (plus the active one, which may sit past the cap).
  const compared = useMemo(() => tickers.slice(0, MAX_COMPARE), [tickers]);
  const wanted = useMemo(
    () => (activeTicker && !compared.includes(activeTicker) ? [...compared, activeTicker] : compared),
    [compared, activeTicker],
  );

  // The fetch effect reads the cache through a ref so that cache writes don't re-run it; it only
  // reacts to a new ticker list or window. Results are cached per ticker and window, and requests
  // in flight are remembered so a re-run never asks twice.
  const cacheRef = useRef(cache);
  useEffect(() => {
    cacheRef.current = cache;
  }, [cache]);
  const inFlight = useRef(new Set<string>());
  const settled = useRef(false);

  useEffect(() => {
    const missing = wanted.filter((t) => {
      const key = cacheKeyOf(t, from, to);
      // A failed load isn't final: it's tried again whenever the list, the window or "Retry" changes.
      return (!cacheRef.current[key] || cacheRef.current[key].error) && !inFlight.current.has(key);
    });
    if (missing.length === 0) return;
    // Wait for typing to pause so partial symbols ("MS", "MSF") aren't fetched against Polygon's
    // 5-calls-a-minute free tier. The first load goes at once; the re-render that swaps in a
    // shared link's tickers after hydration clears this timer before it fires.
    const id = window.setTimeout(
      () => {
        settled.current = true;
        for (const t of missing) {
          const key = cacheKeyOf(t, from, to);
          inFlight.current.add(key);
          fetch(`/api/stocks/aggregates?ticker=${encodeURIComponent(t)}&from=${from}&to=${to}`)
            .then((r) => r.json())
            .then((d) =>
              setCache((prev) => ({
                ...prev,
                [key]: { configured: d.configured ?? true, bars: d.bars ?? [], error: d.error },
              })),
            )
            .catch(() =>
              setCache((prev) => ({ ...prev, [key]: { configured: true, bars: [], error: "Failed to load" } })),
            )
            .finally(() => inFlight.current.delete(key));
        }
      },
      settled.current ? 500 : 0,
    );
    return () => window.clearTimeout(id);
  }, [wanted, from, to, attempt]);

  const entry = activeTicker ? cache[cacheKeyOf(activeTicker, from, to)] : undefined;

  const comparison = useMemo(() => {
    if (compared.length < 2) return null;
    const entries = compared.map((ticker) => ({ ticker, entry: cache[cacheKeyOf(ticker, from, to)] }));
    if (entries.some((e) => !e.entry)) return { loading: true as const };
    if (entries.some((e) => !e.entry.configured)) return null; // the key notice below covers it
    const loaded = entries.filter((e) => !e.entry.error && e.entry.bars.length > 0);
    const colorOf = (t: string) => SERIES[compared.indexOf(t) % SERIES.length];
    return {
      loading: false as const,
      series: loaded.map<Series>((e) => ({
        id: e.ticker,
        label: e.ticker,
        color: colorOf(e.ticker),
        points: rebaseTo100(e.entry.bars),
      })),
      ranking: comparePerformance(loaded.map((e) => ({ ticker: e.ticker, bars: e.entry.bars }))).map((r) => ({
        ...r,
        color: colorOf(r.ticker),
      })),
      failed: entries.filter((e) => e.entry.error).map((e) => `${e.ticker} (${e.entry.error})`),
      empty: entries.filter((e) => !e.entry.error && e.entry.bars.length === 0).map((e) => e.ticker),
    };
  }, [compared, cache, from, to]);

  const view = useMemo(() => {
    if (!entry || !entry.configured || entry.error || entry.bars.length === 0) return null;
    const bars = entry.bars;
    const first = bars[0];
    const last = bars[bars.length - 1];
    const series: Series[] = [
      {
        id: "close",
        label: "Close",
        color: SERIES[0],
        points: bars.map((b) => [new Date(b.date).getTime(), b.close] as [number, number]),
      },
    ];
    const pctChange = first.close !== 0 ? (last.close - first.close) / first.close : 0;
    const recent = [...bars].slice(-10).reverse();
    return { bars, last, pctChange, series, recent };
  }, [entry]);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl">Stocks</h1>
      <p className="text-[var(--muted)]">Daily price history from Polygon.io.</p>

      <Card>
        <div className="grid gap-4 sm:grid-cols-[1fr_auto]">
          <label className="block">
            <span className="eyebrow">Tickers (comma-separated)</span>
            <input
              type="text"
              value={tickersInput}
              onChange={(e) => setTickersInput(e.target.value)}
              className="tabnum mt-1.5 w-full rounded-[var(--radius)] border border-[var(--panel-border)] bg-[var(--bg)] px-3 py-2.5 text-[var(--text)] outline-none focus:border-[var(--accent)]"
            />
          </label>
          <label className="block">
            <span className="eyebrow">Lookback</span>
            <div className="mt-1.5">
              <Segmented
                ariaLabel="Lookback window"
                options={LOOKBACKS}
                value={lookback}
                onChange={(v) => setLookback(v as number)}
              />
            </div>
          </label>
        </div>
        {tickers.length > 1 && (
          <div className="mt-4">
            <Segmented
              ariaLabel="Active ticker"
              options={tickers.map((t) => ({ label: t, value: t }))}
              value={activeTicker ?? tickers[0]}
              onChange={(v) => setActive(v as string)}
            />
          </div>
        )}
      </Card>

      {tickers.length === 0 && <p className="text-[var(--muted)]">Enter at least one ticker.</p>}

      {comparison && (
        <Card>
          <h2 className="text-lg">Relative performance · {url.lookback}</h2>
          <p className="mt-1 text-sm text-[var(--muted)]">
            Each close is rebased to 100 at the start of the window, so the lines compare returns rather
            than prices.
          </p>
          {comparison.loading ? (
            <div role="status" aria-label="Loading comparison" className="mt-4">
              <Skeleton className="h-[280px]" />
            </div>
          ) : comparison.series.length > 0 ? (
            <div className="mt-4 grid gap-6 lg:grid-cols-[1fr_auto]">
              <LineChart
                ariaLabel={`Closes of ${comparison.series.map((s) => s.label).join(", ")} rebased to 100 over ${url.lookback}`}
                exportTitle={`Relative performance ${url.lookback}`}
                series={comparison.series}
                xType="time"
                xLabel="Date"
                yLabel="Index (start = 100)"
              />
              <table className="tabnum h-fit text-sm">
                <caption className="eyebrow mb-2 text-left">Return over {url.lookback}</caption>
                <thead className="text-left text-[var(--muted)]">
                  <tr>
                    <th className="py-1 pr-6 font-medium">Ticker</th>
                    <th className="py-1 pr-6 text-right font-medium">Return</th>
                    <th className="py-1 text-right font-medium">Last close</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.ranking.map((r) => (
                    <tr key={r.ticker} className="border-t border-[var(--panel-border)]">
                      <td className="py-1.5 pr-6 text-[var(--text)]">
                        <span className="mr-2 inline-block h-2 w-3 rounded-sm" style={{ background: r.color }} aria-hidden="true" />
                        {r.ticker}
                        {r.startsLate && <span title={`First bar ${r.firstDate}`}>*</span>}
                      </td>
                      <td className={`py-1.5 pr-6 text-right ${r.change >= 0 ? "text-[var(--pos)]" : "text-[var(--neg)]"}`}>
                        {signedPct(r.change)}
                      </td>
                      <td className="py-1.5 text-right">{money(r.lastClose)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="mt-4 text-sm text-[var(--muted)]">None of these tickers returned data for this window.</p>
          )}
          {!comparison.loading && (
            <div className="mt-3 space-y-1 text-xs text-[var(--faint)]">
              {comparison.ranking.some((r) => r.startsLate) && (
                <p>* Starts later than the others (first trade or first data inside the window), indexed from its own first close.</p>
              )}
              {comparison.empty.length > 0 && <p>No data: {comparison.empty.join(", ")} (check the symbol).</p>}
              {comparison.failed.length > 0 && (
                <p>
                  Couldn&apos;t load: {comparison.failed.join("; ")}.{" "}
                  <button type="button" onClick={retryFailed} className="text-[var(--accent)] underline underline-offset-4">
                    Retry
                  </button>
                </p>
              )}
              {tickers.length > MAX_COMPARE && <p>Comparing the first {MAX_COMPARE} tickers; pick any ticker above for its detail.</p>}
            </div>
          )}
        </Card>
      )}

      {activeTicker && (
        <>
          {!entry && (
            <div role="status" aria-label={`Loading ${activeTicker}`}>
              <Skeleton className="h-80" />
            </div>
          )}

          {entry && !entry.configured && (
            <Card>
              <h2 className="mb-2 text-lg">Polygon API key not set</h2>
              <p className="text-sm text-[var(--muted)]">
                Set <code>POLYGON_API_KEY</code> in <code>web/.env.local</code> for local development, or as a
                Vercel project environment variable, to load stock data. The rest of the app works without it.
              </p>
            </Card>
          )}

          {entry && entry.configured && entry.error && (
            <p className="text-[var(--muted)]">
              Couldn&apos;t load {activeTicker}: {entry.error}.{" "}
              <button type="button" onClick={retryFailed} className="text-[var(--accent)] underline underline-offset-4">
                Retry
              </button>
            </p>
          )}

          {entry && entry.configured && !entry.error && entry.bars.length === 0 && (
            <p className="text-[var(--muted)]">No data for {activeTicker} in this window (check the symbol).</p>
          )}

          {view && (
            <>
              <Card>
                <h2 className="mb-3 text-lg">{activeTicker} · close</h2>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                  <Metric label="Last close" value={money(view.last.close)} tone="accent" />
                  <Metric
                    label="Change (window)"
                    value={signedPct(view.pctChange)}
                    tone={view.pctChange >= 0 ? "pos" : "neg"}
                  />
                  <Metric label="Latest volume" value={money0(view.last.volume)} />
                </div>
                <div className="mt-4">
                  <LineChart
                    ariaLabel={`${activeTicker} daily close price`}
                    exportTitle={`${activeTicker} daily close`}
                    series={view.series}
                    xType="time"
                    xLabel="Date"
                    yLabel="Close"
                  />
                </div>
              </Card>

              <Card>
                <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
                  <h2 className="text-lg">Recent bars</h2>
                  <ExportButtons
                    title={`${activeTicker} daily bars ${url.lookback}`}
                    csv={() => ({
                      columns: ["Date", "Open", "High", "Low", "Close", "Volume", "VWAP"],
                      rows: view.bars.map((b) => [b.date, b.open, b.high, b.low, b.close, b.volume, b.vwap]),
                    })}
                  />
                </div>
                <ScrollRegion label="Recent daily bars">
                  <table className="tabnum w-full text-sm">
                    <thead>
                      <tr className="text-left text-[var(--muted)]">
                        <th className="py-1 pr-4">Date</th>
                        <th className="py-1 pr-4">Open</th>
                        <th className="py-1 pr-4">High</th>
                        <th className="py-1 pr-4">Low</th>
                        <th className="py-1 pr-4">Close</th>
                        <th className="py-1">Volume</th>
                      </tr>
                    </thead>
                    <tbody>
                      {view.recent.map((b) => (
                        <tr key={b.date} className="border-t border-[var(--panel-border)]">
                          <td className="py-1 pr-4">{b.date}</td>
                          <td className="py-1 pr-4">{money(b.open)}</td>
                          <td className="py-1 pr-4">{money(b.high)}</td>
                          <td className="py-1 pr-4">{money(b.low)}</td>
                          <td className="py-1 pr-4">{money(b.close)}</td>
                          <td className="py-1">{money0(b.volume)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </ScrollRegion>
              </Card>
            </>
          )}
        </>
      )}
    </div>
  );
}
