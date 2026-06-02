import { XMLParser } from "fast-xml-parser";
import { YieldRow } from "@/lib/types";

const YIELD_COLS = [
  "BC_1MONTH", "BC_2MONTH", "BC_3MONTH", "BC_6MONTH", "BC_1YEAR", "BC_2YEAR",
  "BC_3YEAR", "BC_5YEAR", "BC_7YEAR", "BC_10YEAR", "BC_20YEAR", "BC_30YEAR",
];

const TREASURY_URL = (year: number) =>
  "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml" +
  `?data=daily_treasury_yield_curve&field_tdr_date_value=${year}`;

export function parseTreasuryXml(xml: string): YieldRow[] {
  const parser = new XMLParser({ ignoreAttributes: true, removeNSPrefix: true });
  const doc = parser.parse(xml);
  const feed = doc?.feed;
  if (!feed?.entry) return [];
  const entries = Array.isArray(feed.entry) ? feed.entry : [feed.entry];

  const rows: YieldRow[] = [];
  for (const entry of entries) {
    const props = entry?.content?.properties;
    if (!props) continue;
    const rawDate = props.NEW_DATE ?? props.DATE ?? props.RecordDate;
    if (!rawDate) continue;
    const d = new Date(rawDate);
    if (Number.isNaN(d.getTime())) continue;

    const row: YieldRow = { date: d.toISOString().slice(0, 10) };
    for (const col of YIELD_COLS) {
      const v = props[col];
      row[col] = v === undefined || v === null || v === "" ? null : Number(v);
    }
    rows.push(row);
  }
  rows.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  return rows;
}

export async function fetchTreasuryYear(year: number): Promise<YieldRow[]> {
  const res = await fetch(TREASURY_URL(year), { next: { revalidate: 3600 } });
  if (!res.ok) throw new Error(`Treasury feed ${year} returned ${res.status}`);
  return parseTreasuryXml(await res.text());
}
