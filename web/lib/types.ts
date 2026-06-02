export const TENOR_LABELS = [
  "1M", "2M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y",
] as const;
export type TenorLabel = (typeof TENOR_LABELS)[number];

export interface YieldRow {
  date: string; // ISO yyyy-mm-dd
  [bcColumn: string]: string | number | null;
}

export interface CurvePoint {
  label: TenorLabel;
  years: number;
  yield: number; // percent
}

export interface Kpis {
  tenYear: number | null; // percent
  twos10s: number | null; // percentage points (10Y - 2Y)
  threeM10Y: number | null; // percentage points (10Y - 3M)
}
