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
    const all: YieldRow[] = [];
    for (let y = y0; y <= y1; y++) {
      const rows = await fetchTreasuryYear(y).catch(() => [] as YieldRow[]);
      all.push(...rows);
    }
    const rows = all.filter((r) => r.date >= start && r.date <= end);
    return NextResponse.json({ points: toSpreadPoints(rows) });
  } catch {
    return NextResponse.json({ points: [] }, { status: 503 });
  }
}
