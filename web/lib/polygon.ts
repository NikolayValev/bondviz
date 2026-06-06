import { StockBar } from "@/lib/types";

// Polygon aggregates (daily bars). Mirrors src/bondviz/stocks.py fetch_aggregates,
// but uses plain fetch (no SDK). The API key is supplied by the server route and
// must never be exposed to the browser.
export function aggregatesUrl(ticker: string, from: string, to: string, apiKey: string): string {
  const t = encodeURIComponent(ticker.toUpperCase());
  return (
    `https://api.polygon.io/v2/aggs/ticker/${t}/range/1/day/${from}/${to}` +
    `?adjusted=true&sort=asc&limit=50000&apiKey=${encodeURIComponent(apiKey)}`
  );
}

export function parseAggregates(json: unknown): StockBar[] {
  const results = (json as { results?: unknown } | null)?.results;
  if (!Array.isArray(results)) return [];
  const bars: StockBar[] = [];
  for (const r of results as Record<string, unknown>[]) {
    const t = r.t;
    if (typeof t !== "number") continue;
    bars.push({
      date: new Date(t).toISOString().slice(0, 10),
      open: Number(r.o),
      high: Number(r.h),
      low: Number(r.l),
      close: Number(r.c),
      volume: Number(r.v),
      vwap: r.vw === undefined || r.vw === null ? null : Number(r.vw),
    });
  }
  bars.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  return bars;
}

export async function fetchAggregates(
  ticker: string,
  from: string,
  to: string,
  apiKey: string,
): Promise<StockBar[]> {
  const res = await fetch(aggregatesUrl(ticker, from, to, apiKey), { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error(`Polygon aggregates for ${ticker} returned ${res.status}`);
  return parseAggregates(await res.json());
}
