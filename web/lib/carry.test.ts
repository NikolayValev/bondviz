import { describe, it, expect } from "vitest";
import { carryRollDown } from "@/lib/carry";

const FLAT = [
  { years: 1, yieldPct: 5, label: "1Y" },
  { years: 2, yieldPct: 5, label: "2Y" },
  { years: 5, yieldPct: 5, label: "5Y" },
  { years: 10, yieldPct: 5, label: "10Y" },
];

const UPWARD = [
  { years: 1, yieldPct: 1, label: "1Y" },
  { years: 2, yieldPct: 2, label: "2Y" },
  { years: 5, yieldPct: 3, label: "5Y" },
  { years: 10, yieldPct: 4, label: "10Y" },
];

const INVERTED = [
  { years: 1, yieldPct: 5, label: "1Y" },
  { years: 2, yieldPct: 4, label: "2Y" },
  { years: 5, yieldPct: 3, label: "5Y" },
  { years: 10, yieldPct: 2, label: "10Y" },
];

describe("carryRollDown", () => {
  it("returns empty for empty input", () => {
    expect(carryRollDown([], 0.25)).toEqual([]);
  });

  it("flat curve: roll-down ≈ 0 and total ≈ carry", () => {
    const out = carryRollDown(FLAT, 0.25);
    expect(out.length).toBeGreaterThan(0);
    for (const p of out) {
      expect(p.rollBps).toBeCloseTo(0, 6);
      expect(p.totalBps).toBeCloseTo(p.carryBps, 6);
    }
  });

  it("flat 5% curve: carry over 3M ≈ 125 bps", () => {
    const p = carryRollDown(FLAT, 0.25).find((x) => x.label === "10Y")!;
    expect(p.carryBps).toBeCloseTo(125, 6); // 0.05 * 0.25 = 0.0125 = 125 bps
  });

  it("upward curve: roll-down is positive", () => {
    for (const p of carryRollDown(UPWARD, 0.25)) expect(p.rollBps).toBeGreaterThan(0);
  });

  it("inverted curve: roll-down is negative", () => {
    for (const p of carryRollDown(INVERTED, 0.25)) expect(p.rollBps).toBeLessThan(0);
  });

  it("carry scales linearly with horizon", () => {
    const a = carryRollDown(FLAT, 0.25).find((x) => x.label === "10Y")!;
    const b = carryRollDown(FLAT, 0.5).find((x) => x.label === "10Y")!;
    expect(b.carryBps).toBeCloseTo(a.carryBps * 2, 6);
  });

  it("breakeven sign matches total sign", () => {
    for (const p of carryRollDown(UPWARD, 0.25)) {
      expect(Math.sign(p.breakevenBps)).toBe(Math.sign(p.totalBps));
    }
  });

  it("excludes tenors with maturity ≤ horizon", () => {
    const curve = [
      { years: 0.5, yieldPct: 5, label: "6M" },
      { years: 1, yieldPct: 5, label: "1Y" },
      { years: 2, yieldPct: 5, label: "2Y" },
    ];
    const labels = carryRollDown(curve, 1).map((p) => p.label);
    expect(labels).not.toContain("6M");
    expect(labels).not.toContain("1Y"); // 1Y == horizon, excluded
    expect(labels).toContain("2Y");
  });
});
