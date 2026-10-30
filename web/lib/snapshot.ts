import { computeCurveKpis, rowToCurve } from "@/lib/finance";
import type { CurvePoint, Kpis, YieldRow } from "@/lib/types";
import { shiftMonths } from "@/lib/dates";

// The home page's market snapshot: the latest curve and KPIs, their change over a month, and
// the curve a year earlier for comparison.

const DAY_MS = 86_400_000;

/** The row dated closest to `targetIso` (rows in any order); null when there are none. */
export function nearestRow<T extends { date: string }>(rows: T[], targetIso: string): T | null {
  const target = Date.parse(targetIso);
  let best: T | null = null;
  let bestDiff = Infinity;
  for (const r of rows) {
    const diff = Math.abs(Date.parse(r.date) - target);
    if (diff < bestDiff) {
      best = r;
      bestDiff = diff;
    }
  }
  return best;
}

/** A row near the target date, or null if the nearest one is more than `toleranceDays` away
 *  (missing data shouldn't pass for "a month ago"). */
function rowNear(rows: YieldRow[], targetIso: string, toleranceDays = 10): YieldRow | null {
  const r = nearestRow(rows, targetIso);
  return r && Math.abs(Date.parse(r.date) - Date.parse(targetIso)) <= toleranceDays * DAY_MS ? r : null;
}

const diff = (a: number | null, b: number | null) => (a !== null && b !== null ? a - b : null);

export interface MarketSnapshot {
  date: string;
  kpis: Kpis;
  /** Change in each KPI since about a month earlier (same units), or null without that data. */
  monthChange: Kpis | null;
  curve: CurvePoint[];
  yearAgo: { date: string; curve: CurvePoint[] } | null;
}

export function marketSnapshot(rows: YieldRow[]): MarketSnapshot | null {
  const sorted = [...rows].sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  const latest = sorted.at(-1);
  if (!latest) return null;
  const kpis = computeCurveKpis(latest);
  const monthAgo = rowNear(sorted, shiftMonths(latest.date, -1));
  const yearAgo = rowNear(sorted, shiftMonths(latest.date, -12));
  const before = monthAgo && monthAgo !== latest ? computeCurveKpis(monthAgo) : null;
  return {
    date: latest.date,
    kpis,
    monthChange: before
      ? {
          tenYear: diff(kpis.tenYear, before.tenYear),
          twos10s: diff(kpis.twos10s, before.twos10s),
          threeM10Y: diff(kpis.threeM10Y, before.threeM10Y),
        }
      : null,
    curve: rowToCurve(latest),
    yearAgo: yearAgo && yearAgo !== latest ? { date: yearAgo.date, curve: rowToCurve(yearAgo) } : null,
  };
}
