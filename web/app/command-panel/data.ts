// DataRegistry for the command panel: the ONLY data generated widgets can read
// (via useMetric). Each load() runs server-side and calls the same pure lib/
// functions the app's own pages use — no HTTP hop through our own API routes.
// Chart-consumable sources return { label, value }[].

import { createDataRegistry } from "@nikolayvalev/command-panel";
import { fetchTreasuryYear } from "@/lib/treasury";
import { computeCurveKpis, BC_TO_TENOR, TENOR_YEARS } from "@/lib/finance";
import { bootstrapZeros } from "@/lib/curve";
import { carryRollDown } from "@/lib/carry";
import { toSpreadPoints } from "@/lib/signal";
import { TENOR_LABELS, TenorLabel, YieldRow } from "@/lib/types";

const TENOR_TO_BC: Record<TenorLabel, string> = Object.fromEntries(
  Object.entries(BC_TO_TENOR).map(([bc, tenor]) => [tenor, bc]),
) as Record<TenorLabel, string>;

/** Latest available daily par-yield row (falls back to the prior year in January). */
async function latestRow(): Promise<YieldRow> {
  const year = new Date().getUTCFullYear();
  let rows = await fetchTreasuryYear(year);
  if (rows.length === 0) rows = await fetchTreasuryYear(year - 1);
  if (rows.length === 0) throw new Error("Treasury par-yield feed unavailable");
  return rows[rows.length - 1];
}

/** One { label, years, value } point per tenor with a numeric yield in the row. */
function curvePoints(row: YieldRow): { label: TenorLabel; years: number; value: number }[] {
  return TENOR_LABELS.flatMap((label) => {
    const v = row[TENOR_TO_BC[label]];
    const n = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
    return Number.isFinite(n) ? [{ label, years: TENOR_YEARS[label], value: n }] : [];
  });
}

export const dataRegistry = createDataRegistry([
  {
    id: "yield.treasury.parCurve",
    description:
      "Latest U.S. Treasury daily par-yield curve: one point per tenor from 1M to 30Y, " +
      "yield in percent. Labels are tenor names ('1M', '3M', …, '10Y', '30Y') in maturity order.",
    resultSchema: {
      type: "array",
      items: { label: "string (tenor, e.g. '10Y')", value: "number (par yield, percent)" },
    },
    async load() {
      const row = await latestRow();
      return curvePoints(row).map(({ label, value }) => ({ label, value }));
    },
  },
  {
    id: "yield.treasury.kpis",
    description:
      "Headline curve KPIs from the latest Treasury close: 10-year yield (percent), " +
      "2s10s slope and 3m10y slope (percentage points; negative = inverted). Any field " +
      "may be null if the tenor was not published that day.",
    resultSchema: {
      type: "object",
      properties: {
        date: "string (ISO yyyy-mm-dd of the close)",
        tenYear: "number | null (10Y yield, percent)",
        twos10s: "number | null (10Y minus 2Y, percentage points)",
        threeM10Y: "number | null (10Y minus 3M, percentage points)",
      },
    },
    async load() {
      const row = await latestRow();
      return { date: row.date, ...computeCurveKpis(row) };
    },
  },
  {
    id: "yield.treasury.zeroCurve",
    description:
      "Zero-coupon (spot) curve bootstrapped from the latest par curve on a semiannual grid " +
      "(0.5Y steps out to 30Y). Values are annual-compounded zero rates in percent.",
    resultSchema: {
      type: "array",
      items: { label: "string (maturity, e.g. '7.5Y')", value: "number (zero rate, percent)" },
    },
    async load() {
      const row = await latestRow();
      const par = curvePoints(row).map((p) => ({ years: p.years, yieldPct: p.value }));
      const { grid, zero } = bootstrapZeros(par);
      return grid.map((t, i) => ({ label: `${t}Y`, value: zero[i] * 100 }));
    },
  },
  {
    id: "carry.rollDown",
    description:
      "Carry + roll-down return per tenor over a 3-month horizon on a static curve, in basis " +
      "points of horizon return (the same numbers as the app's Carry page). `value` is the " +
      "total; carry/roll components and the breakeven sell-off cushion ride along.",
    resultSchema: {
      type: "array",
      items: {
        label: "string (tenor, e.g. '10Y')",
        value: "number (carry + roll-down, bps of horizon return)",
        carryBps: "number",
        rollBps: "number",
        breakevenBps: "number (yield rise in bps that zeroes the return)",
      },
    },
    async load() {
      const row = await latestRow();
      const curve = curvePoints(row).map((p) => ({ label: p.label, years: p.years, yieldPct: p.value }));
      return carryRollDown(curve, 0.25).map((p) => ({
        label: p.label,
        value: p.totalBps,
        carryBps: p.carryBps,
        rollBps: p.rollBps,
        breakevenBps: p.breakevenBps,
      }));
    },
  },
  {
    id: "yield.spread2s10s.history",
    description:
      "2s10s slope (10Y minus 2Y) over the last two years, in basis points, sampled weekly " +
      "(every 5th trading day, always including the latest close). Negative = inverted.",
    resultSchema: {
      type: "array",
      items: { label: "string (ISO date)", value: "number (2s10s spread, bps)" },
    },
    async load() {
      const now = new Date();
      const startYear = now.getUTCFullYear() - 2;
      const years: number[] = [];
      for (let y = startYear; y <= now.getUTCFullYear(); y++) years.push(y);
      const perYear = await Promise.all(
        years.map((y) => fetchTreasuryYear(y).catch(() => [] as YieldRow[])),
      );
      const start = new Date(now.getTime() - 730 * 86_400_000).toISOString().slice(0, 10);
      const points = toSpreadPoints(perYear.flat().filter((r) => r.date >= start)).filter(
        (p) => p.s2s10s !== null,
      );
      return points
        .filter((_, i) => i % 5 === 0 || i === points.length - 1)
        .map((p) => ({ label: p.date, value: (p.s2s10s as number) * 100 }));
    },
  },
]);
