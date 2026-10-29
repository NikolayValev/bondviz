// CSV export, pure: builds the text and the file name. The download itself (Blob + link click)
// lives in components/ui/ExportButtons.tsx.

export type CsvCell = string | number | null | undefined;

export interface CsvTable {
  columns: string[];
  rows: CsvCell[][];
}

/**
 * One cell, RFC 4180 style: quoted when it holds a comma, quote or line break. Numbers are
 * written plainly to at most 6 decimals, which also trims float noise (0.1 + 0.2 → 0.3; well
 * past the precision of a yield in % or a spread in bps); missing or non-finite values are
 * empty. Text a spreadsheet would run as a formula (leading =, +, -, @) gets an apostrophe,
 * so pass numbers as numbers: a negative number stays numeric.
 */
export function csvCell(v: CsvCell): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isFinite(v) ? String(+v.toFixed(6)) : "";
  const s = /^[=+\-@\t\r]/.test(v) ? `'${v}` : v;
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** A whole table, header first, CRLF line endings. */
export function toCsv({ columns, rows }: CsvTable): string {
  return [columns, ...rows].map((r) => r.map(csvCell).join(",")).join("\r\n") + "\r\n";
}

/**
 * Line-chart series as one table: an x column, then one column per series, joined on x (a
 * series with no point at some x leaves that cell empty). Time x values are epoch-ms at UTC
 * midnight and are written as ISO dates. `unit` is appended to each series header: "Zero (%)".
 */
export function seriesTable(
  series: { label: string; points: [number, number][] }[],
  { xLabel = "x", xType = "linear", unit }: { xLabel?: string; xType?: "linear" | "time"; unit?: string } = {},
): CsvTable {
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const byX = series.map((s) => new Map(s.points));
  return {
    columns: [xLabel, ...series.map((s) => (unit ? `${s.label} (${unit})` : s.label))],
    rows: xs.map((x) => [
      xType === "time" ? new Date(x).toISOString().slice(0, 10) : x,
      ...byX.map((m) => m.get(x)),
    ]),
  };
}

/** Category charts and grids: one row per category, one column per series. */
export function categoryTable(
  categoryLabel: string,
  categories: string[],
  series: { label: string; values: (number | null)[] }[],
): CsvTable {
  return {
    columns: [categoryLabel, ...series.map((s) => s.label)],
    rows: categories.map((c, i) => [c, ...series.map((s) => s.values[i])]),
  };
}

/** The unit in an axis label's trailing parentheses: "Rate (%)" → "%"; none → undefined. */
export function axisUnit(axisLabel: string | undefined): string | undefined {
  return axisLabel ? /\(([^()]+)\)\s*$/.exec(axisLabel)?.[1] : undefined;
}

/** "Zero & forward curve" → "bondviz-zero-forward-curve-2026-10-27.png". */
export function exportFilename(title: string, ext: "csv" | "png", date = new Date()): string {
  const slug = title
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60)
    .replace(/-+$/, "");
  return `bondviz-${slug || "export"}-${date.toISOString().slice(0, 10)}.${ext}`;
}
