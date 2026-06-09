import { describe, it, expect } from "vitest";
import { iso, money, money0, signedPct } from "@/lib/format";

describe("iso", () => {
  it("formats a date as UTC yyyy-mm-dd", () => {
    expect(iso(new Date("2025-01-03T00:00:00Z"))).toBe("2025-01-03");
  });
});

describe("money", () => {
  it("shows two decimals with separators", () => {
    expect(money(1234.5)).toBe("1,234.50");
  });
});

describe("money0", () => {
  it("shows no decimals", () => {
    expect(money0(1234567)).toBe("1,234,567");
  });
});

describe("signedPct", () => {
  it("adds a + and percent for non-negative ratios", () => {
    expect(signedPct(0.0125)).toBe("+1.25%");
  });
  it("uses a unicode minus for negative ratios", () => {
    expect(signedPct(-0.004)).toBe("−0.40%");
  });
});
