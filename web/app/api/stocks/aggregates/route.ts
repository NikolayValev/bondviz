import { NextResponse } from "next/server";
import { fetchAggregates } from "@/lib/polygon";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const ticker = searchParams.get("ticker");
  const from = searchParams.get("from");
  const to = searchParams.get("to");
  if (!ticker || !from || !to) {
    return NextResponse.json({ configured: true, bars: [] }, { status: 400 });
  }

  const apiKey = process.env.POLYGON_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ configured: false, bars: [] });
  }

  try {
    const bars = await fetchAggregates(ticker, from, to, apiKey);
    return NextResponse.json({ configured: true, bars });
  } catch (e) {
    return NextResponse.json(
      { configured: true, bars: [], error: e instanceof Error ? e.message : "fetch failed" },
      { status: 502 },
    );
  }
}
