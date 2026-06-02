import { NextResponse } from "next/server";
import { fetchTreasuryYear } from "@/lib/treasury";

export async function GET() {
  try {
    const year = new Date().getUTCFullYear();
    let rows = await fetchTreasuryYear(year);
    if (rows.length === 0) rows = await fetchTreasuryYear(year - 1);
    if (rows.length === 0) return NextResponse.json({ row: null }, { status: 503 });
    return NextResponse.json({ row: rows[rows.length - 1] });
  } catch {
    return NextResponse.json({ row: null }, { status: 503 });
  }
}
