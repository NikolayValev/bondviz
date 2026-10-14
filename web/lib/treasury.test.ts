import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseTreasuryXml, fetchTreasuryYear, toIsoDate } from "@/lib/treasury";

const xml = readFileSync(
  fileURLToPath(new URL("../test/fixtures/treasury-sample.xml", import.meta.url)),
  "utf-8",
);

describe("parseTreasuryXml", () => {
  it("extracts sorted rows with numeric BC_* fields", () => {
    const rows = parseTreasuryXml(xml);
    expect(rows).toHaveLength(2);
    expect(rows[0].date).toBe("2025-01-02");
    expect(rows[0].BC_10YEAR).toBe(4.5);
    expect(rows[1].date).toBe("2025-01-03");
  });
  it("returns [] for empty/garbage input", () => {
    expect(parseTreasuryXml("<feed></feed>")).toEqual([]);
  });
});

describe("timezone safety", () => {
  const original = process.env.TZ;
  afterEach(() => {
    process.env.TZ = original;
  });
  it.each(["Asia/Tokyo", "Pacific/Kiritimati", "America/Los_Angeles", "UTC"])(
    "keeps the feed's calendar date when the server runs in %s",
    (tz) => {
      process.env.TZ = tz;
      const rows = parseTreasuryXml(xml);
      expect(rows.map((r) => r.date)).toEqual(["2025-01-02", "2025-01-03"]);
    },
  );
});

describe("toIsoDate", () => {
  it("takes the calendar date from ISO-like strings", () => {
    expect(toIsoDate("2025-01-03T00:00:00")).toBe("2025-01-03");
    expect(toIsoDate("2024-12-31")).toBe("2024-12-31");
  });
  it("rejects impossible and unparseable dates", () => {
    expect(toIsoDate("2025-02-30T00:00:00")).toBeNull();
    expect(toIsoDate("not a date")).toBeNull();
  });
});

describe("fetchTreasuryYear", () => {
  afterEach(() => vi.restoreAllMocks());
  it("fetches and parses the year feed", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(xml, { status: 200 })));
    const rows = await fetchTreasuryYear(2025);
    expect(rows).toHaveLength(2);
  });
  it("throws on non-OK responses", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 500 })));
    await expect(fetchTreasuryYear(2025)).rejects.toThrow();
  });
});
