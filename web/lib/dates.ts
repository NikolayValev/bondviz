// Calendar dates as "YYYY-MM-DD" strings. Data dates are handled in UTC so they never shift with
// the viewer's time zone; "today" for file names is the viewer's own date. One home for the month
// names and the labels the charts and pages use.

export const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const pad = (n: number) => String(n).padStart(2, "0");

/** A date moved by whole calendar months, clamped to the target month's last day:
 *  "2026-03-31" − 1 → "2026-02-28" (not Mar 3, which is where Date's overflow lands). */
export function shiftMonths(iso: string, months: number): string {
  const total = Number(iso.slice(0, 4)) * 12 + Number(iso.slice(5, 7)) - 1 + months;
  const y = Math.floor(total / 12);
  const m = total - y * 12;
  const lastDay = new Date(Date.UTC(y, m + 1, 0)).getUTCDate();
  return `${y}-${pad(m + 1)}-${pad(Math.min(Number(iso.slice(8, 10)), lastDay))}`;
}

/** "2001-03-01" → "Mar 2001". */
export const monthYear = (iso: string) => `${MONTHS[Number(iso.slice(5, 7)) - 1]} ${iso.slice(0, 4)}`;

/** An ISO date, or epoch-ms at UTC midnight, as "Oct 14, 2026". */
export function dayLabel(v: string | number): string {
  const d = typeof v === "number" ? new Date(v) : new Date(`${v.slice(0, 10)}T00:00:00Z`);
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${d.getUTCFullYear()}`;
}

/** The viewer's local calendar date (for file names and stamps, not data). */
export function localIsoDate(d = new Date()): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}
