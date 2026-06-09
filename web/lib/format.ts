/** Shared number/date formatters used across the client pages. */

/** ISO yyyy-mm-dd (UTC) for a Date — used to build API query windows. */
export function iso(d: Date): string {
  return d.toISOString().slice(0, 10);
}

/** Currency-style number with two decimals (e.g. 1,234.50). */
export const money = (v: number) =>
  v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

/** Whole-number with thousands separators (e.g. volumes, face values). */
export const money0 = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 });

/** Signed percentage from a ratio, two decimals (e.g. +1.25% / −0.40%). */
export const signedPct = (v: number) =>
  `${v >= 0 ? "+" : "−"}${(Math.abs(v) * 100).toFixed(2)}%`;
