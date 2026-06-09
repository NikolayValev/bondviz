"use client";
import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Metric } from "@/components/ui/Metric";
import { Segmented } from "@/components/ui/Segmented";
import { LineChart, Series } from "@/components/charts/LineChart";
import { StockBar } from "@/lib/types";
import { iso, money, money0, signedPct } from "@/lib/format";

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
  const [tickersInput, setTickersInput] = useState("AAPL, MSFT, GOOGL");
  const [lookback, setLookback] = useState(3);
  const [active, setActive] = useState<string | null>(null);
  const [cache, setCache] = useState<Record<string, CacheEntry>>({});

  const tickers = useMemo(() => parseTickers(tickersInput), [tickersInput]);
  const activeTicker = active && tickers.includes(active) ? active : tickers[0] ?? null;

  // Recomputed only when lookback changes, so the fetch effect has stable deps.
  const { from, to } = useMemo(() => {
    const end = new Date();
    const start = new Date();
    start.setMonth(start.getMonth() - lookback);
    return { from: iso(start), to: iso(end) };
  }, [lookback]);

  const cacheKey = activeTicker ? `${activeTicker}|${from}|${to}` : null;

  useEffect(() => {
    if (!cacheKey || !activeTicker || cache[cacheKey]) return;
    let cancelled = false;
    fetch(`/api/stocks/aggregates?ticker=${encodeURIComponent(activeTicker)}&from=${from}&to=${to}`)
      .then((r) => r.json())
      .then((d) => {
        if (!cancelled)
          setCache((prev) => ({
            ...prev,
            [cacheKey]: { configured: d.configured ?? true, bars: d.bars ?? [], error: d.error },
          }));
      })
      .catch(() => {
        if (!cancelled)
          setCache((prev) => ({ ...prev, [cacheKey]: { configured: true, bars: [], error: "Failed to load" } }));
      });
    return () => {
      cancelled = true;
    };
  }, [cacheKey, activeTicker, from, to, cache]);

  const entry = cacheKey ? cache[cacheKey] : undefined;

  const view = useMemo(() => {
    if (!entry || !entry.configured || entry.error || entry.bars.length === 0) return null;
    const bars = entry.bars;
    const first = bars[0];
    const last = bars[bars.length - 1];
    const series: Series[] = [
      {
        id: "close",
        label: "Close",
        color: "#00d68f",
        points: bars.map((b) => [new Date(b.date).getTime(), b.close] as [number, number]),
      },
    ];
    const pctChange = first.close !== 0 ? (last.close - first.close) / first.close : 0;
    const recent = [...bars].slice(-10).reverse();
    return { last, pctChange, series, recent };
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

      {activeTicker && (
        <>
          {!entry && <p className="text-[var(--muted)]">Loading {activeTicker}…</p>}

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
              Couldn&apos;t load {activeTicker}: {entry.error}
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
                    series={view.series}
                    xType="time"
                    xLabel="Date"
                    yLabel="Close"
                  />
                </div>
              </Card>

              <Card>
                <h2 className="mb-2 text-lg">Recent bars</h2>
                <div className="overflow-x-auto">
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
                </div>
              </Card>
            </>
          )}
        </>
      )}
    </div>
  );
}
