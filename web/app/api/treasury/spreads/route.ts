import { NextResponse } from "next/server";
import { fetchTreasuryYear } from "@/lib/treasury";
import { toSpreadPoints } from "@/lib/signal";
import { YieldRow } from "@/lib/types";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const start = searchParams.get("start");
  const end = searchParams.get("end");
  if (!start || !end) return NextResponse.json({ points: [] }, { status: 400 });

  try {
    const y0 = new Date(start).getUTCFullYear();
    const y1 = new Date(end).getUTCFullYear();
    const years: number[] = [];
    for (let y = y0; y <= y1; y++) years.push(y);
    // Fetch every year concurrently — a 35-year span was 35 sequential round
    // trips before. A failed year resolves to [] so one bad year is skipped.
    const perYear = await Promise.all(years.map((y) => fetchTreasuryYear(y).catch(() => [] as YieldRow[])));
    const rows = perYear.flat().filter((r) => r.date >= start && r.date <= end);
    return NextResponse.json({ points: toSpreadPoints(rows) });
  } catch {
    return NextResponse.json({ points: [] }, { status: 503 });
  }
}
