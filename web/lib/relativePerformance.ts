import type { StockBar } from "@/lib/types";

// Several tickers on one chart: each close series is rebased so its first close in the window is
// 100, which puts a $30 stock and a $3,000 one on the same scale and reads directly as "% since
// the start of the window".

/** Closes as an index, first close = 100, keyed by epoch-ms at UTC midnight (a chart's time x). */
export function rebaseTo100(bars: StockBar[]): [number, number][] {
  const valid = bars.filter((b) => Number.isFinite(b.close) && b.close > 0);
  if (valid.length === 0) return [];
  const base = valid[0].close;
  return valid.map((b) => [Date.parse(b.date), (b.close / base) * 100]);
}

export interface TickerPerformance {
  ticker: string;
  firstDate: string;
  lastClose: number;
  /** Return over the window, decimal (0.05 = +5%). */
  change: number;
  /** First bar is after the earliest first bar among the compared tickers (listed mid-window or
   *  missing early data), so its index starts later than the others'. */
  startsLate: boolean;
}

/** Window return for each ticker with data, best first. */
export function comparePerformance(entries: { ticker: string; bars: StockBar[] }[]): TickerPerformance[] {
  const rows = entries
    .map(({ ticker, bars }) => {
      const valid = bars.filter((b) => Number.isFinite(b.close) && b.close > 0);
      if (valid.length === 0) return null;
      const first = valid[0];
      const last = valid[valid.length - 1];
      return { ticker, firstDate: first.date, lastClose: last.close, change: last.close / first.close - 1 };
    })
    .filter((r): r is Omit<TickerPerformance, "startsLate"> => r !== null);
  const earliest = rows.reduce((min, r) => (r.firstDate < min ? r.firstDate : min), rows[0]?.firstDate ?? "");
  return rows
    .map((r) => ({ ...r, startsLate: r.firstDate > earliest }))
    .sort((a, b) => b.change - a.change);
}
