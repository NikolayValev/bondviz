import { NextResponse } from "next/server";
import { fetchTreasuryYear } from "@/lib/treasury";
import { YieldRow } from "@/lib/types";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const start = searchParams.get("start");
  const end = searchParams.get("end");
  if (!start || !end) return NextResponse.json({ rows: [] }, { status: 400 });

  try {
    const y0 = new Date(start).getUTCFullYear();
    const y1 = new Date(end).getUTCFullYear();
    const years = Array.from({ length: Math.max(0, y1 - y0 + 1) }, (_, i) => y0 + i);
    // One request per calendar year, fetched in parallel (same pattern as /spreads).
    const perYear = await Promise.all(
      years.map((y) => fetchTreasuryYear(y).catch(() => [] as YieldRow[])),
    );
    const rows = perYear.flat().filter((r) => r.date >= start && r.date <= end);
    return NextResponse.json({ rows });
  } catch {
    return NextResponse.json({ rows: [] }, { status: 503 });
  }
}
