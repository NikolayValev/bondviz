import { CurvePoint, Kpis, TenorLabel, YieldRow } from "@/lib/types";

export const TENOR_YEARS: Record<TenorLabel, number> = {
  "1M": 1 / 12, "2M": 2 / 12, "3M": 3 / 12, "6M": 6 / 12,
  "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30,
};

export const BC_TO_TENOR: Record<string, TenorLabel> = {
  BC_1MONTH: "1M", BC_2MONTH: "2M", BC_3MONTH: "3M", BC_6MONTH: "6M",
  BC_1YEAR: "1Y", BC_2YEAR: "2Y", BC_3YEAR: "3Y", BC_5YEAR: "5Y",
  BC_7YEAR: "7Y", BC_10YEAR: "10Y", BC_20YEAR: "20Y", BC_30YEAR: "30Y",
};

/** Convert a spread/level in percentage points to basis points. */
export const toBps = (pp: number): number => pp * 100;

export function pvContinuous(face: number, coupon: number, yieldRate: number, years: number): number {
  const C = face * coupon;
  if (yieldRate === 0) return C * years + face;
  const d = Math.exp(-yieldRate * years);
  return (C * (1 - d)) / yieldRate + face * d;
}

export function discountFactors(yieldRate: number, tenors: number[]): { t: number; df: number }[] {
  return tenors.map((t) => ({ t, df: Math.exp(-yieldRate * t) }));
}

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && !Number.isNaN(v) ? v : null;
}

export function computeCurveKpis(row: Record<string, unknown>): Kpis {
  const y10 = numOrNull(row.BC_10YEAR);
  const y2 = numOrNull(row.BC_2YEAR);
  const y3m = numOrNull(row.BC_3MONTH);
  return {
    tenYear: y10,
    twos10s: y10 !== null && y2 !== null ? y10 - y2 : null,
    threeM10Y: y10 !== null && y3m !== null ? y10 - y3m : null,
  };
}

export function rowToCurve(row: Record<string, unknown>): CurvePoint[] {
  const pts: CurvePoint[] = [];
  for (const [bc, label] of Object.entries(BC_TO_TENOR)) {
    const v = numOrNull(row[bc]);
    if (v !== null) pts.push({ label, years: TENOR_YEARS[label], yield: v });
  }
  return pts.sort((a, b) => a.years - b.years);
}

export function spreadSeries(rows: YieldRow[]): {
  twos10s: [number, number][];
  threeM10Y: [number, number][];
} {
  const twos10s: [number, number][] = [];
  const threeM10Y: [number, number][] = [];
  for (const r of rows) {
    const t = new Date(r.date).getTime();
    const y10 = numOrNull(r.BC_10YEAR);
    const y2 = numOrNull(r.BC_2YEAR);
    const y3m = numOrNull(r.BC_3MONTH);
    if (y10 !== null && y2 !== null) twos10s.push([t, y10 - y2]);
    if (y10 !== null && y3m !== null) threeM10Y.push([t, y10 - y3m]);
  }
  return { twos10s, threeM10Y };
}

export function describeCurve(curve: CurvePoint[]): string {
  const front = curve.find((p) => ["3M", "6M", "1Y", "2Y"].includes(p.label));
  const long = [...curve].reverse().find((p) => ["10Y", "20Y", "30Y"].includes(p.label));
  if (!front || !long) return "Incomplete tenor coverage in this snapshot.";
  const slope = long.yield - front.yield;
  if (slope < -0.1) return `Inverted: ${long.label} (${long.yield.toFixed(2)}%) below ${front.label} (${front.yield.toFixed(2)}%).`;
  if (slope < 0.1) return `Roughly flat between ${front.label} and ${long.label}.`;
  return `Upward sloping: ${long.label} (${long.yield.toFixed(2)}%) above ${front.label} (${front.yield.toFixed(2)}%).`;
}
