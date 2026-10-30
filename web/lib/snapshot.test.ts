import { describe, it, expect } from "vitest";
import { marketSnapshot, nearestRow } from "@/lib/snapshot";

const row = (date: string, y3m: number, y2: number, y10: number) => ({
  date,
  BC_3MONTH: y3m,
  BC_2YEAR: y2,
  BC_10YEAR: y10,
});

describe("nearestRow", () => {
  const rows = [{ date: "2026-09-18" }, { date: "2026-09-21" }, { date: "2026-09-22" }];
  it("finds the closest date, ignoring order", () => {
    expect(nearestRow(rows, "2026-09-20")?.date).toBe("2026-09-21");
    expect(nearestRow([...rows].reverse(), "2026-09-01")?.date).toBe("2026-09-18");
  });
  it("returns null for no rows", () => {
    expect(nearestRow([], "2026-09-20")).toBeNull();
  });
});

describe("marketSnapshot", () => {
  const rows = [
    row("2026-10-13", 4.26, 4.79, 5.21), // latest, out of order on purpose
    row("2025-10-14", 4.02, 3.48, 4.03),
    row("2026-09-12", 4.3, 4.6, 5.0),
    row("2026-10-10", 4.25, 4.78, 5.2),
  ];

  it("reports the latest KPIs and their change over a month", () => {
    const s = marketSnapshot(rows)!;
    expect(s.date).toBe("2026-10-13");
    expect(s.kpis.tenYear).toBe(5.21);
    expect(s.monthChange?.tenYear).toBeCloseTo(0.21, 10);
    expect(s.monthChange?.twos10s).toBeCloseTo(0.42 - 0.4, 10);
    expect(s.monthChange?.threeM10Y).toBeCloseTo(0.95 - 0.7, 10);
  });

  it("includes the curve a year earlier", () => {
    const s = marketSnapshot(rows)!;
    expect(s.yearAgo?.date).toBe("2025-10-14");
    expect(s.curve.map((p) => p.label)).toEqual(["3M", "2Y", "10Y"]);
  });

  it("leaves out comparisons it has no data near", () => {
    const s = marketSnapshot([row("2026-10-13", 4.26, 4.79, 5.21), row("2026-07-01", 4, 4, 4)])!;
    expect(s.monthChange).toBeNull();
    expect(s.yearAgo).toBeNull();
  });

  it("is null without rows", () => {
    expect(marketSnapshot([])).toBeNull();
  });
});
